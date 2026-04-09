"""
MLX integration for KV cache compaction.

Compacts KV caches from mlx-lm models in-place, enabling longer context
with less memory on Apple Silicon.

Usage with mlx-lm:

    from mlx_lm import load, generate
    from mlx_lm.models.cache import make_prompt_cache
    from kv_compact.mlx import compact_cache

    model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
    cache = make_prompt_cache(model)

    # ... prefill the cache via generate or model.__call__ ...

    # Compact: keep 50% of tokens, attention-aware selection
    stats = compact_cache(model, cache, target_ratio=0.5)
    print(f"Compacted {stats.original_len} -> {stats.compacted_len} tokens")
    print(f"Avg cosine similarity: {stats.avg_cosine_sim:.4f}")

    # Continue generating with the compacted cache
    # generate(model, tokenizer, prompt="...", cache=cache)

Requires: mlx, mlx-lm, numpy
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional, Sequence

import numpy as np

from kv_compact._core import CompactParams, compact_layer


@dataclasses.dataclass
class CacheCompactStats:
    """Statistics from compacting an entire model's KV cache."""
    original_len: int           # original sequence length
    compacted_len: int          # sequence length after compaction
    n_layers_compacted: int     # number of layers that were compacted
    n_layers_skipped: int       # number of layers skipped (non-attention / SSM)
    avg_cosine_sim: float       # average cosine similarity across all compacted layers
    avg_mse: float              # average MSE across all compacted layers
    elapsed_ms: float           # total wall-clock time


def _is_kv_cache(cache_obj) -> bool:
    """Check if a cache object is a compactable KV cache (not SSM/Mamba)."""
    try:
        from mlx_lm.models.cache import KVCache, QuantizedKVCache
        return isinstance(cache_obj, (KVCache, QuantizedKVCache))
    except ImportError:
        # Fallback: check for .keys and .values attributes
        return hasattr(cache_obj, "keys") and hasattr(cache_obj, "values")


def _is_quantized_cache(cache_obj) -> bool:
    """Check if a cache object uses quantized storage."""
    try:
        from mlx_lm.models.cache import QuantizedKVCache
        return isinstance(cache_obj, QuantizedKVCache)
    except ImportError:
        return False


def _extract_kv(cache_obj) -> tuple:
    """Extract K, V as numpy float32 arrays from an MLX cache layer.

    Returns:
        (K, V) each shaped (n_kv_heads, seq_len, head_dim) as float32 numpy arrays.
        Returns (None, None) if the cache is empty.
    """
    import mlx.core as mx

    # Use .state property which returns trimmed tensors
    state = cache_obj.state
    if state is None:
        return None, None

    keys, values = state

    # Handle quantized caches: state returns (data, scales, biases) tuples
    if isinstance(keys, (tuple, list)):
        # QuantizedKVCache: dequantize first
        keys = mx.dequantize(*keys)
        values = mx.dequantize(*values)

    # MLX shape: (B, n_kv_heads, seq_len, head_dim)
    # We need: (n_kv_heads, seq_len, head_dim)
    mx.eval(keys, values)  # force lazy evaluation

    K_np = np.array(keys, copy=False)
    V_np = np.array(values, copy=False)

    # Squeeze batch dim (B=1 for single-sequence)
    if K_np.ndim == 4:
        K_np = K_np[0]  # (n_kv_heads, seq_len, head_dim)
        V_np = V_np[0]

    return K_np.astype(np.float32), V_np.astype(np.float32)


def _write_kv(cache_obj, K_new: np.ndarray, V_new: np.ndarray):
    """Write compacted K, V back into an MLX cache layer.

    Args:
        cache_obj: MLX KVCache object
        K_new: (n_kv_heads, t, head_dim) float32
        V_new: (n_kv_heads, t, head_dim) float32
    """
    import mlx.core as mx

    # Determine the original dtype from the existing cache
    old_keys = cache_obj.keys
    if isinstance(old_keys, (tuple, list)):
        dtype = mx.float16  # quantized caches store in float16
    else:
        dtype = old_keys.dtype

    # Add batch dimension: (n_kv_heads, t, head_dim) -> (1, n_kv_heads, t, head_dim)
    K_mx = mx.array(K_new[np.newaxis], dtype=dtype)
    V_mx = mx.array(V_new[np.newaxis], dtype=dtype)

    # Use the .state setter which handles offset tracking
    cache_obj.state = (K_mx, V_mx)


def compact_cache(
    model,
    cache: list,
    *,
    target_ratio: float = 0.5,
    target_count: int = 0,
    layers: Optional[Sequence[int]] = None,
    skip_beta: bool = True,
    ridge: float = 1e-6,
    score_method: int = 1,
    chunk_size: int = 0,
    verbose: bool = False,
) -> CacheCompactStats:
    """Compact an mlx-lm KV cache in-place.

    This is the main entry point for MLX users. It iterates over all layers,
    extracts K/V tensors, runs attention-based compaction, and writes the
    compacted cache back.

    Args:
        model: mlx-lm model (used only for metadata, not modified)
        cache: List of cache objects from make_prompt_cache(model)
        target_ratio: Fraction of tokens to keep (default 0.5 = 50%)
        target_count: Explicit target count (overrides ratio if > 0)
        layers: Optional list of layer indices to compact. If None, auto-detects
                compactable layers (skips SSM/Mamba layers in hybrid models).
        skip_beta: Skip NNLS beta solve (default True, LS refit is sufficient)
        ridge: Ridge regularization for LS solve
        score_method: Key importance scoring (0=max, 1=rms, 2=mean)
        chunk_size: Chunked compaction (0=auto, -1=disabled)
        verbose: Print per-layer stats

    Returns:
        CacheCompactStats with overall metrics
    """
    t0 = time.perf_counter()

    params = CompactParams(
        target_ratio=target_ratio,
        target_count=target_count,
        skip_beta=skip_beta,
        ridge=ridge,
        use_cheap_qref=True,
        score_method=score_method,
        chunk_size=chunk_size,
    )

    n_layers = len(cache)
    n_compacted = 0
    n_skipped = 0
    total_cos = 0.0
    total_mse = 0.0
    original_len = 0
    compacted_len = 0

    for layer_idx in range(n_layers):
        layer_cache = cache[layer_idx]

        # Skip non-KV layers (SSM/Mamba in hybrid models)
        if not _is_kv_cache(layer_cache):
            n_skipped += 1
            continue

        # Skip layers not in the explicit list
        if layers is not None and layer_idx not in layers:
            n_skipped += 1
            continue

        # Extract K, V as numpy arrays
        K, V = _extract_kv(layer_cache)
        if K is None or K.shape[1] == 0:
            n_skipped += 1
            continue

        if original_len == 0:
            original_len = K.shape[1]

        # Run compaction
        result = compact_layer(K, V, None, params)

        # Write compacted cache back
        _write_kv(layer_cache, result.K_selected, result.C_v)

        if compacted_len == 0:
            compacted_len = result.selected_indices.shape[0]

        total_cos += result.stats.avg_cosine_sim
        total_mse += result.stats.avg_mse
        n_compacted += 1

        if verbose:
            print(
                f"  Layer {layer_idx:3d}: "
                f"{K.shape[1]} -> {result.selected_indices.shape[0]} tokens, "
                f"cos_sim={result.stats.avg_cosine_sim:.4f}, "
                f"mse={result.stats.avg_mse:.6f}, "
                f"time={result.stats.elapsed_ms:.1f}ms"
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return CacheCompactStats(
        original_len=original_len,
        compacted_len=compacted_len,
        n_layers_compacted=n_compacted,
        n_layers_skipped=n_skipped,
        avg_cosine_sim=total_cos / max(n_compacted, 1),
        avg_mse=total_mse / max(n_compacted, 1),
        elapsed_ms=elapsed_ms,
    )


def compact_and_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int = 256,
    target_ratio: float = 0.5,
    verbose: bool = False,
    **generate_kwargs,
) -> str:
    """Convenience: prefill, compact, and generate in one call.

    Args:
        model: mlx-lm model
        tokenizer: mlx-lm tokenizer
        prompt: Input text
        max_tokens: Max tokens to generate after compaction
        target_ratio: KV cache compression ratio
        verbose: Print compaction stats
        **generate_kwargs: Passed to mlx_lm.generate()

    Returns:
        Generated text string
    """
    from mlx_lm import generate as mlx_generate
    from mlx_lm.models.cache import make_prompt_cache

    import mlx.core as mx

    # Create cache and prefill
    cache = make_prompt_cache(model)
    tokens = mx.array(tokenizer.encode(prompt))[None]  # (1, seq_len)

    # Prefill: run model forward to populate cache
    model(tokens, cache=cache)

    if verbose:
        seq_len = cache[0].offset if hasattr(cache[0], "offset") else "?"
        print(f"Prefilled {seq_len} tokens, compacting to {target_ratio:.0%}...")

    # Compact
    stats = compact_cache(model, cache, target_ratio=target_ratio, verbose=verbose)

    if verbose:
        print(
            f"Compacted: {stats.original_len} -> {stats.compacted_len} tokens "
            f"({stats.n_layers_compacted} layers, {stats.elapsed_ms:.0f}ms, "
            f"cos_sim={stats.avg_cosine_sim:.4f})"
        )

    # Generate with compacted cache
    result = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        cache=cache,
        **generate_kwargs,
    )

    return result


def compact_and_generate_speculative(
    model,
    draft_model,
    tokenizer,
    prompt: str,
    *,
    max_tokens: int = 256,
    target_ratio: float = 0.5,
    num_draft_tokens: int = 3,
    verbose: bool = False,
    **generate_kwargs,
) -> str:
    """Prefill, compact the target model's cache, then generate with speculative decoding.

    Speculative decoding uses a small draft model to propose tokens, which the
    large target model verifies in a single forward pass. Combined with cache
    compaction, this gives both memory savings AND faster generation.

    Pipeline:
      1. Prefill target model cache with full context
      2. Compact target cache (attention-based selection + LS value refit)
      3. Prefill draft model cache with same context (NOT compacted — it's tiny)
      4. Generate using speculative decoding with both caches

    Args:
        model: Target (large) mlx-lm model
        draft_model: Draft (small) mlx-lm model for speculation
        tokenizer: Shared tokenizer
        prompt: Input text
        max_tokens: Max tokens to generate
        target_ratio: KV cache compression ratio for target model
        num_draft_tokens: Tokens the draft model proposes per step (default 3)
        verbose: Print compaction stats
        **generate_kwargs: Passed to mlx_lm.generate()

    Returns:
        Generated text string
    """
    from mlx_lm import generate as mlx_generate
    from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache

    import mlx.core as mx

    tokens = mx.array(tokenizer.encode(prompt))[None]

    # 1. Prefill target model
    target_cache = make_prompt_cache(model)
    model(tokens, cache=target_cache)
    mx.eval([c.state for c in target_cache if hasattr(c, "state")])

    if verbose:
        seq_len = target_cache[0].offset if hasattr(target_cache[0], "offset") else "?"
        print(f"Target model prefilled: {seq_len} tokens")

    # 2. Compact target cache (the big one — this is where memory savings matter)
    stats = compact_cache(model, target_cache, target_ratio=target_ratio, verbose=verbose)

    if verbose:
        print(
            f"Target cache compacted: {stats.original_len} -> {stats.compacted_len} tokens "
            f"({stats.elapsed_ms:.0f}ms, cos_sim={stats.avg_cosine_sim:.4f})"
        )

    # Verify target cache is trimmable (required for speculative decoding)
    if not can_trim_prompt_cache(target_cache):
        raise ValueError(
            "Compacted target cache is not trimmable. "
            "Speculative decoding requires trimmable caches (KVCache or QuantizedKVCache)."
        )

    # 3. Prefill draft model (NOT compacted — draft model is small, cache is cheap)
    draft_cache = make_prompt_cache(draft_model)
    draft_model(tokens, cache=draft_cache)
    mx.eval([c.state for c in draft_cache if hasattr(c, "state")])

    if verbose:
        draft_len = draft_cache[0].offset if hasattr(draft_cache[0], "offset") else "?"
        print(f"Draft model prefilled: {draft_len} tokens (not compacted)")

    # 4. Combine caches: [target_layers..., draft_layers...]
    #    mlx-lm's speculative_generate_step expects this layout
    combined_cache = list(target_cache) + list(draft_cache)

    # 5. Generate with speculative decoding
    result = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        draft_model=draft_model,
        prompt_cache=combined_cache,
        num_draft_tokens=num_draft_tokens,
        **generate_kwargs,
    )

    return result

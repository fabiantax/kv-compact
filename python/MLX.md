# MLX Integration Guide

KV cache compaction for [MLX](https://github.com/ml-explore/mlx) models on Apple Silicon. Compresses the KV cache at inference time with minimal quality loss -- no model retraining required.

## Quick Start

```bash
# Install
cd python && pip install -e ".[mlx]"

# Run
python examples/mlx_example.py --model mlx-community/Qwen3-8B-4bit
```

## Why MLX + Compaction?

MLX outperforms llama.cpp on Apple Silicon for LLM inference. But long contexts still hit memory limits because the KV cache grows linearly with sequence length. Compaction shrinks the cache by selecting the most important positions (via attention scores) and refitting values (via least squares), preserving >99.9% cosine similarity with full-cache outputs.

| Context | Full KV cache | 50% compacted | 80% compacted |
|---------|---------------|---------------|---------------|
| 4K tokens, 8B model | ~1 GB | ~500 MB | ~200 MB |
| 32K tokens, 8B model | ~8 GB | ~4 GB | ~1.6 GB |
| 128K tokens, 8B model | ~32 GB | ~16 GB | ~6.4 GB |

## Usage

### Basic: Compact and Generate

```python
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache
from kv_compact.mlx import compact_cache
import mlx.core as mx

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")

# Prefill cache with long context
cache = make_prompt_cache(model)
tokens = mx.array(tokenizer.encode(long_prompt))[None]
model(tokens, cache=cache)

# Compact: keep 50% of tokens
stats = compact_cache(model, cache, target_ratio=0.5)
print(f"{stats.original_len} -> {stats.compacted_len} tokens")
print(f"Cosine similarity: {stats.avg_cosine_sim:.4f}")

# Generate with compacted cache
output = generate(model, tokenizer, prompt=long_prompt,
                  max_tokens=256, cache=cache)
```

### One-liner

```python
from kv_compact.mlx import compact_and_generate

output = compact_and_generate(
    model, tokenizer, long_prompt,
    target_ratio=0.5, max_tokens=256, verbose=True
)
```

### Compaction + Speculative Decoding

Combine cache compaction with speculative decoding for both memory savings AND faster generation. The draft model proposes tokens cheaply; the target model verifies them in a single forward pass.

```python
from mlx_lm import load
from kv_compact.mlx import compact_and_generate_speculative

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
draft_model, _ = load("mlx-community/Qwen3-0.6B-4bit")

output = compact_and_generate_speculative(
    model, draft_model, tokenizer, long_prompt,
    target_ratio=0.5,       # compact target model's KV cache
    num_draft_tokens=3,     # draft proposes 3 tokens per step
    max_tokens=256,
    verbose=True,
)
```

**How it works together:**

```
1. Prefill target model (8B)         -- builds full KV cache
2. compact_cache(target_ratio=0.5)   -- shrink cache 50%, ~100ms
3. Prefill draft model (0.6B)        -- tiny, not compacted
4. Speculative decode loop:
     draft proposes: "the" "cat" "sat"
     target verifies: [accept] [accept] [reject] -> "ran"
     trim_prompt_cache(1)            -- rewinds rejected token
     (works correctly on compacted cache)
```

The two optimizations are orthogonal:
- **Compaction**: reduces memory so longer contexts fit
- **Speculative decoding**: reduces latency via batched verification

Command line:

```bash
python examples/mlx_example.py \
  --model mlx-community/Qwen3-8B-4bit \
  --draft mlx-community/Qwen3-0.6B-4bit \
  --ratio 0.5 -v
```

### Hybrid Models (Qwen 3.5, Jamba)

Models mixing attention layers with SSM/Mamba layers are auto-detected. Only attention layers (which have KV caches) are compacted; SSM layers are passed through unchanged.

```python
stats = compact_cache(model, cache, target_ratio=0.5)
print(f"Compacted {stats.n_layers_compacted} layers, skipped {stats.n_layers_skipped}")
```

Or specify layers explicitly:

```python
# Only compact layers 3, 7, 11, ... (attention layers in Qwen 3.5)
attention_layers = list(range(3, 64, 4))
stats = compact_cache(model, cache, target_ratio=0.5, layers=attention_layers)
```

## API Reference

### `compact_cache(model, cache, **kwargs) -> CacheCompactStats`

Compact an mlx-lm KV cache in-place.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | mlx-lm model | required | Target model (used for metadata only) |
| `cache` | list | required | Cache from `make_prompt_cache(model)` |
| `target_ratio` | float | 0.5 | Fraction of tokens to keep |
| `target_count` | int | 0 | Explicit count (overrides ratio if > 0) |
| `layers` | list[int] | None | Layer indices to compact (None = auto-detect) |
| `skip_beta` | bool | True | Skip NNLS beta (LS refit alone is sufficient) |
| `ridge` | float | 1e-6 | Ridge regularization for LS solve |
| `score_method` | int | 1 | Key scoring: 0=max, 1=rms (paper default), 2=mean |
| `chunk_size` | int | 0 | Chunked compaction: 0=auto, -1=disabled |
| `verbose` | bool | False | Print per-layer stats |

Returns `CacheCompactStats`:
- `original_len` / `compacted_len` -- token counts before/after
- `n_layers_compacted` / `n_layers_skipped` -- layer counts
- `avg_cosine_sim` -- average cosine similarity (>0.999 typical at 50%)
- `avg_mse` -- average mean squared error
- `elapsed_ms` -- total wall-clock time

### `compact_and_generate_speculative(model, draft_model, tokenizer, prompt, **kwargs) -> str`

Prefill, compact, and generate with speculative decoding.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | mlx-lm model | required | Target (large) model |
| `draft_model` | mlx-lm model | required | Draft (small) model |
| `tokenizer` | tokenizer | required | Shared tokenizer |
| `prompt` | str | required | Input text |
| `target_ratio` | float | 0.5 | Compression ratio for target cache |
| `num_draft_tokens` | int | 3 | Tokens proposed per speculative step |
| `max_tokens` | int | 256 | Max generation length |

## Building from Source

The Python package uses our C library via ctypes for maximum speed, with a pure numpy fallback.

```bash
# With C library (fastest -- uses BLAS + OpenMP)
mkdir build && cd build
cmake .. && cmake --build . -j
cd ../python && pip install -e ".[mlx]"

# Without C library (numpy fallback, still reasonable on Apple Silicon)
cd python && pip install -e ".[mlx]"
```

The build auto-detects your platform:

| | MacBook (Apple Silicon) | AMD Strix Halo (Linux) |
|---|---|---|
| BLAS | Accelerate (always) | OpenBLAS (auto-detect) |
| GPU | Metal (via MLX) | ROCm/HIP (auto-detect) |
| OpenMP | Homebrew libomp | System libgomp |
| Shared lib | `libkv-compact.dylib` | `libkv-compact.so` |
| Python path | `kv_compact.mlx` | `kv_compact._bindings` (ctypes) |

## Architecture

```
kv-compact-math.h          Pure CPU math (header-only, zero deps)
    |
kv-compact-api.h/cpp       C API: kv_compact() takes float* arrays
    |
    +-- _bindings.py        ctypes FFI to libkv-compact.so/.dylib
    |
    +-- _numpy_impl.py      Pure numpy fallback (no compilation needed)
    |
    +-- _core.py            Dispatch: tries C library, falls back to numpy
    |
    +-- mlx.py              MLX integration: extract/compact/write-back KV caches
```

The core compaction takes raw `float*` arrays -- no framework dependency. The MLX module handles the conversion between `mx.array` tensors (shape `(B, n_kv_heads, seq_len, head_dim)`, typically float16) and the flat float32 arrays our C library expects.

## Supported Cache Types

| Cache type | Compactable | Notes |
|------------|-------------|-------|
| `KVCache` | Yes | Standard attention layers |
| `QuantizedKVCache` | Yes | Dequantized before compaction |
| `RotatingKVCache` | Yes | Trimmable, works with speculative decoding |
| `ArraysCache` (SSM/Mamba) | No | Auto-skipped |

## Recommended Model Pairings for Speculative Decoding

| Target model | Draft model | Notes |
|-------------|-------------|-------|
| Qwen3-8B-4bit | Qwen3-0.6B-4bit | Same family, shared tokenizer |
| Llama-3.3-70B-4bit | Llama-3.2-3B-Instruct-4bit | Same tokenizer |
| Qwen3-32B-4bit | Qwen3-1.7B-4bit | Good quality/speed balance |

The draft model should use the same tokenizer as the target. Smaller is faster but may have lower acceptance rate.

"""
Core compaction interface — dispatches to C library or numpy fallback.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class CompactParams:
    """Parameters for KV cache compaction."""
    target_ratio: float = 0.5       # fraction of tokens to keep
    target_count: int = 0           # explicit count (overrides ratio if > 0)
    skip_beta: bool = True          # skip NNLS beta (LS refit alone is sufficient)
    ridge: float = 1e-6             # ridge regularization for LS
    use_cheap_qref: bool = True     # generate Q_ref from K vectors
    score_method: int = 1           # 0=max, 1=rms (paper default), 2=mean
    chunk_size: int = 0             # 0=auto, -1=disabled, >0=explicit
    n_threads: int = 0              # 0=auto


@dataclasses.dataclass
class CompactStats:
    """Quality metrics from compaction."""
    avg_cosine_sim: float = 0.0
    avg_mse: float = 0.0
    avg_agreement: float = 0.0
    elapsed_ms: float = 0.0


@dataclasses.dataclass
class CompactResult:
    """Result of compacting a single layer's KV cache."""
    selected_indices: np.ndarray    # [t] int32 — which positions were kept
    C_v: np.ndarray                 # [n_kv_heads, t, head_dim] — refitted values
    K_selected: np.ndarray          # [n_kv_heads, t, head_dim] — selected keys
    stats: CompactStats


def _try_load_c_library():
    """Try to load the compiled C shared library.

    Returns the compact_c function if the shared library is available,
    or None if it can't be loaded (falls back to numpy).
    """
    try:
        from kv_compact._bindings import _get_lib, compact_c
        _get_lib()  # force load — raises OSError if .so/.dylib not found
        return compact_c
    except (ImportError, OSError):
        return None


def compact_layer(
    K: np.ndarray,
    V: np.ndarray,
    Q_ref: Optional[np.ndarray] = None,
    params: Optional[CompactParams] = None,
) -> CompactResult:
    """Compact a single layer's KV cache.

    Args:
        K: Key tensor, shape (n_kv_heads, seq_len, head_dim), float32
        V: Value tensor, shape (n_kv_heads, seq_len, head_dim), float32
        Q_ref: Reference queries, shape (n_kv_heads, n_q, head_dim), float32.
               If None and use_cheap_qref=True, generated from K.
        params: Compaction parameters. Defaults used if None.

    Returns:
        CompactResult with selected indices, refitted values, and stats.
    """
    if params is None:
        params = CompactParams()

    # Validate shapes
    assert K.ndim == 3, f"K must be (n_kv_heads, seq_len, head_dim), got {K.shape}"
    assert V.ndim == 3, f"V must be (n_kv_heads, seq_len, head_dim), got {V.shape}"
    n_kv_heads, T, d_k = K.shape
    _, T_v, d_v = V.shape
    assert T == T_v, f"K and V seq_len mismatch: {T} vs {T_v}"
    assert V.shape[0] == n_kv_heads

    # Ensure contiguous float32
    K = np.ascontiguousarray(K, dtype=np.float32)
    V = np.ascontiguousarray(V, dtype=np.float32)

    # Try C library first, fall back to numpy
    c_fn = _try_load_c_library()
    if c_fn is not None:
        return c_fn(K, V, Q_ref, params)

    # Numpy fallback
    from kv_compact._numpy_impl import compact_numpy
    return compact_numpy(K, V, Q_ref, params)


def compact(
    K: np.ndarray,
    V: np.ndarray,
    Q_ref: Optional[np.ndarray] = None,
    *,
    target_ratio: float = 0.5,
    target_count: int = 0,
    **kwargs,
) -> CompactResult:
    """Convenience wrapper — compact a single layer with keyword args.

    Args:
        K: Key tensor, shape (n_kv_heads, seq_len, head_dim)
        V: Value tensor, shape (n_kv_heads, seq_len, head_dim)
        Q_ref: Optional reference queries
        target_ratio: Fraction of tokens to keep (default 0.5)
        target_count: Explicit count (overrides ratio if > 0)
        **kwargs: Additional CompactParams fields

    Returns:
        CompactResult
    """
    params = CompactParams(target_ratio=target_ratio, target_count=target_count, **kwargs)
    return compact_layer(K, V, Q_ref, params)

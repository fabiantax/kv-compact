"""
ctypes bindings for the kv-compact C shared library.

Looks for libkv-compact.so/.dylib in:
  1. KV_COMPACT_LIB environment variable
  2. ../build/ relative to this package
  3. System library paths
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from kv_compact._core import CompactParams, CompactResult, CompactStats


# ============================================================================
# C struct definitions (mirror kv-compact-api.h)
# ============================================================================

class _CStats(ctypes.Structure):
    _fields_ = [
        ("avg_cosine_sim", ctypes.c_float),
        ("avg_mse", ctypes.c_float),
        ("avg_agreement", ctypes.c_float),
        ("elapsed_ms", ctypes.c_double),
        ("scoring_ms", ctypes.c_double),
        ("nnls_ms", ctypes.c_double),
    ]


class _CResult(ctypes.Structure):
    _fields_ = [
        ("t", ctypes.c_int),
        ("n_head_kv", ctypes.c_int),
        ("selected_indices", ctypes.POINTER(ctypes.c_int)),
        ("beta", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        ("C_v", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        ("stats", _CStats),
    ]


class _CParams(ctypes.Structure):
    _fields_ = [
        ("target_ratio", ctypes.c_float),
        ("target_count", ctypes.c_int),
        ("use_sensitivity", ctypes.c_int),
        ("ridge", ctypes.c_float),
        ("nnls_max_iter", ctypes.c_int),
        ("refine_rounds", ctypes.c_int),
        ("use_diversity", ctypes.c_int),
        ("diversity_strength", ctypes.c_float),
        ("n_shared_prefix", ctypes.c_int),
        ("use_cheap_qref", ctypes.c_int),
        ("skip_beta", ctypes.c_int),
        ("score_method", ctypes.c_int),
        ("use_omp", ctypes.c_int),
        ("omp_k_choice", ctypes.c_int),
        ("omp_refit_interval", ctypes.c_int),
        ("nnls_method", ctypes.c_int),
        ("nnls_pgd_iters", ctypes.c_int),
        ("chunk_size", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("layer_filter", ctypes.c_void_p),
        ("layer_filter_data", ctypes.c_void_p),
        ("reasoning", ctypes.c_void_p),
    ]


# ============================================================================
# Library loading
# ============================================================================

def _find_library() -> Optional[str]:
    """Search for libkv-compact shared library."""
    # 1. Explicit environment variable
    env_path = os.environ.get("KV_COMPACT_LIB")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Common build directories relative to package
    pkg_dir = Path(__file__).parent
    repo_root = pkg_dir.parent.parent
    suffixes = [".so", ".dylib"] if sys.platform == "darwin" else [".so"]

    search_dirs = [
        repo_root / "build",
        repo_root / "build" / "Release",
        repo_root / "build" / "Debug",
        pkg_dir,
    ]

    for d in search_dirs:
        for suffix in suffixes:
            candidate = d / f"libkv-compact{suffix}"
            if candidate.is_file():
                return str(candidate)

    # 3. System paths
    found = ctypes.util.find_library("kv-compact")
    if found:
        return found

    return None


def _load_library():
    """Load the shared library and set up function signatures."""
    path = _find_library()
    if path is None:
        raise OSError(
            "libkv-compact shared library not found. Build with:\n"
            "  cmake -DKV_COMPACT_BUILD_SHARED=ON .. && cmake --build . -j\n"
            "Or set KV_COMPACT_LIB=/path/to/libkv-compact.so"
        )

    lib = ctypes.CDLL(path)

    # kv_compact_params_default
    lib.kv_compact_params_default.restype = _CParams
    lib.kv_compact_params_default.argtypes = []

    # kv_compact
    lib.kv_compact.restype = ctypes.c_int
    lib.kv_compact.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # K_all
        ctypes.POINTER(ctypes.c_float),  # V_all
        ctypes.POINTER(ctypes.c_float),  # Q_ref_all
        ctypes.c_int,                    # T
        ctypes.c_int,                    # n_q
        ctypes.c_int,                    # n_head_kv
        ctypes.c_int,                    # d_k
        ctypes.c_int,                    # d_v
        ctypes.POINTER(_CParams),        # params
        ctypes.POINTER(_CResult),        # result
    ]

    # kv_compact_result_free
    lib.kv_compact_result_free.restype = None
    lib.kv_compact_result_free.argtypes = [ctypes.POINTER(_CResult)]

    return lib


# Lazy singleton
_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


# ============================================================================
# Python wrapper
# ============================================================================

def compact_c(
    K: np.ndarray,
    V: np.ndarray,
    Q_ref: Optional[np.ndarray],
    params: CompactParams,
) -> CompactResult:
    """Call the C library kv_compact() function.

    Args:
        K: (n_kv_heads, seq_len, head_dim) float32
        V: (n_kv_heads, seq_len, head_dim) float32
        Q_ref: (n_kv_heads, n_q, head_dim) float32 or None
        params: CompactParams

    Returns:
        CompactResult
    """
    lib = _get_lib()

    n_kv_heads, T, d_k = K.shape
    d_v = V.shape[2]

    # C API expects [T × n_kv_heads × d_k] — transpose from (heads, seq, dim)
    K_c = np.ascontiguousarray(K.transpose(1, 0, 2).reshape(T, n_kv_heads * d_k), dtype=np.float32)
    V_c = np.ascontiguousarray(V.transpose(1, 0, 2).reshape(T, n_kv_heads * d_v), dtype=np.float32)

    if Q_ref is not None:
        n_q = Q_ref.shape[1]
        Q_c = np.ascontiguousarray(
            Q_ref.transpose(1, 0, 2).reshape(n_q, n_kv_heads * d_k), dtype=np.float32
        )
        q_ptr = Q_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        n_q = 0
        q_ptr = None

    # Build C params from defaults, then override
    c_params = lib.kv_compact_params_default()
    c_params.target_ratio = params.target_ratio
    c_params.target_count = params.target_count
    c_params.skip_beta = 1 if params.skip_beta else 0
    c_params.ridge = params.ridge
    c_params.use_cheap_qref = 1 if params.use_cheap_qref else 0
    c_params.score_method = params.score_method
    c_params.chunk_size = params.chunk_size
    c_params.n_threads = params.n_threads

    c_result = _CResult()

    rc = lib.kv_compact(
        K_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        V_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q_ptr,
        T, n_q, n_kv_heads, d_k, d_v,
        ctypes.byref(c_params),
        ctypes.byref(c_result),
    )

    if rc != 0:
        raise RuntimeError(f"kv_compact() returned error code {rc}")

    t = c_result.t

    # Copy results to numpy before freeing
    indices = np.array([c_result.selected_indices[i] for i in range(t)], dtype=np.int32)

    C_v = np.zeros((n_kv_heads, t, d_v), dtype=np.float32)
    for h in range(n_kv_heads):
        for i in range(t * d_v):
            C_v[h, i // d_v, i % d_v] = c_result.C_v[h][i]

    # Extract selected keys
    K_selected = K[:, indices, :]  # (n_kv_heads, t, d_k)

    stats = CompactStats(
        avg_cosine_sim=c_result.stats.avg_cosine_sim,
        avg_mse=c_result.stats.avg_mse,
        avg_agreement=c_result.stats.avg_agreement,
        elapsed_ms=c_result.stats.elapsed_ms,
    )

    lib.kv_compact_result_free(ctypes.byref(c_result))

    return CompactResult(
        selected_indices=indices,
        C_v=C_v,
        K_selected=K_selected,
        stats=stats,
    )

"""
kv-compact: Fast KV Cache Compaction via Attention Matching

Python bindings for the kv-compact library. Supports two backends:
  1. Native C library (libkv-compact.so/.dylib) via ctypes — fastest
  2. Pure numpy fallback — no compilation needed, works everywhere

For MLX integration, see kv_compact.mlx module.

Usage:
    from kv_compact import compact, CompactParams

    result = compact(K, V, target_ratio=0.5)
    # result.selected_indices — which positions were kept
    # result.C_v — refitted value matrix
    # result.stats — quality metrics
"""

from kv_compact._core import (
    compact,
    compact_layer,
    CompactParams,
    CompactResult,
    CompactStats,
)

__version__ = "0.1.0"
__all__ = ["compact", "compact_layer", "CompactParams", "CompactResult", "CompactStats"]

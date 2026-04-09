"""
Pure numpy implementation of KV cache compaction.

Used as fallback when the C shared library is not available.
Implements the same algorithm as kv-compact-math.h:
  1. Key selection via attention scoring
  2. Least-squares value refitting

Performance: ~10-50x slower than the C library, but works everywhere.
On Apple Silicon, numpy uses Accelerate BLAS so it's still reasonable.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from kv_compact._core import CompactParams, CompactResult, CompactStats


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _score_keys_rms(attn: np.ndarray) -> np.ndarray:
    """RMS importance score: sqrt(mean_i(attn[i,j]^2)) for each key j."""
    return np.sqrt(np.mean(attn ** 2, axis=0))


def _score_keys_max(attn: np.ndarray) -> np.ndarray:
    """Max importance score: max_i(attn[i,j]) for each key j."""
    return np.max(attn, axis=0)


def _score_keys_mean(attn: np.ndarray) -> np.ndarray:
    """Mean importance score: mean_i(attn[i,j]) for each key j."""
    return np.mean(attn, axis=0)


def _least_squares(A: np.ndarray, b: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Solve min ||Ax - b||^2 + ridge*||x||^2 via normal equations.

    A: (m, n), b: (m, p) -> x: (n, p)
    """
    AtA = A.T @ A
    AtA[np.diag_indices_from(AtA)] += ridge
    Atb = A.T @ b
    try:
        L = np.linalg.cholesky(AtA)
        y = np.linalg.solve(L, Atb)
        x = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x


def compact_numpy(
    K: np.ndarray,
    V: np.ndarray,
    Q_ref: Optional[np.ndarray],
    params: CompactParams,
) -> CompactResult:
    """Pure numpy compaction of a single layer's KV cache.

    Args:
        K: (n_kv_heads, T, d_k) float32
        V: (n_kv_heads, T, d_v) float32
        Q_ref: (n_kv_heads, n_q, d_k) float32, or None
        params: CompactParams

    Returns:
        CompactResult
    """
    t0 = time.perf_counter()

    n_kv_heads, T, d_k = K.shape
    d_v = V.shape[2]
    scale = 1.0 / np.sqrt(d_k)

    # Determine target count
    if params.target_count > 0:
        t_keep = min(params.target_count, T)
    else:
        t_keep = max(1, int(T * params.target_ratio))

    if t_keep >= T:
        # Nothing to compact
        return CompactResult(
            selected_indices=np.arange(T, dtype=np.int32),
            C_v=V.copy(),
            K_selected=K.copy(),
            stats=CompactStats(avg_cosine_sim=1.0, avg_mse=0.0, avg_agreement=1.0,
                               elapsed_ms=0.0),
        )

    # Generate Q_ref from K if not provided (cheap proxy)
    if Q_ref is None or params.use_cheap_qref:
        # Use last min(t_keep, 64) keys as reference queries
        n_q = min(t_keep, 64)
        Q_ref = K[:, -n_q:, :]  # (n_kv_heads, n_q, d_k)

    n_q = Q_ref.shape[1]

    # === Step 1: Key selection via attention scoring ===
    t_score = time.perf_counter()

    # Aggregate importance scores across heads
    scores = np.zeros(T, dtype=np.float64)
    score_fn = {0: _score_keys_max, 1: _score_keys_rms, 2: _score_keys_mean}.get(
        params.score_method, _score_keys_rms
    )

    for h in range(n_kv_heads):
        # attn = softmax(Q_ref @ K^T / sqrt(d_k))  -> (n_q, T)
        logits = Q_ref[h] @ K[h].T * scale  # (n_q, T)
        attn = _softmax(logits, axis=-1)  # (n_q, T)
        scores += score_fn(attn).astype(np.float64)

    # Select top-t_keep positions
    selected = np.argsort(scores)[-t_keep:]
    selected.sort()
    selected = selected.astype(np.int32)

    scoring_ms = (time.perf_counter() - t_score) * 1000

    # === Step 2: Least-squares value refit ===
    t_ls = time.perf_counter()

    C_v = np.zeros((n_kv_heads, t_keep, d_v), dtype=np.float32)

    for h in range(n_kv_heads):
        # Build attention sub-matrix: A_sel = softmax(Q_ref @ K_sel^T / sqrt(d_k))
        K_sel = K[h, selected, :]  # (t_keep, d_k)
        logits_sel = Q_ref[h] @ K_sel.T * scale  # (n_q, t_keep)
        A_sel = _softmax(logits_sel, axis=-1)  # (n_q, t_keep)

        # Target: b = softmax(Q_ref @ K_all^T / sqrt(d_k)) @ V_all
        logits_all = Q_ref[h] @ K[h].T * scale  # (n_q, T)
        A_all = _softmax(logits_all, axis=-1)  # (n_q, T)
        b = A_all @ V[h]  # (n_q, d_v)

        # Solve: min ||A_sel @ C_v_h - b||^2
        C_v[h] = _least_squares(A_sel, b, ridge=params.ridge)

    ls_ms = (time.perf_counter() - t_ls) * 1000

    # === Quality metrics ===
    total_cos = 0.0
    total_mse = 0.0
    total_agree = 0.0

    for h in range(n_kv_heads):
        K_sel = K[h, selected, :]
        logits_sel = Q_ref[h] @ K_sel.T * scale
        A_sel = _softmax(logits_sel, axis=-1)
        out_compact = A_sel @ C_v[h]  # (n_q, d_v)

        logits_all = Q_ref[h] @ K[h].T * scale
        A_all = _softmax(logits_all, axis=-1)
        out_orig = A_all @ V[h]  # (n_q, d_v)

        # Cosine similarity
        for q in range(n_q):
            a, b_vec = out_orig[q], out_compact[q]
            dot = np.dot(a, b_vec)
            na, nb = np.linalg.norm(a), np.linalg.norm(b_vec)
            if na > 0 and nb > 0:
                total_cos += dot / (na * nb)
            else:
                total_cos += 1.0

        # MSE
        total_mse += np.mean((out_orig - out_compact) ** 2)

        # Agreement (argmax match)
        total_agree += np.mean(np.argmax(out_orig, axis=-1) == np.argmax(out_compact, axis=-1))

    n_total = n_kv_heads * n_q
    elapsed_ms = (time.perf_counter() - t0) * 1000

    stats = CompactStats(
        avg_cosine_sim=float(total_cos / n_total) if n_total > 0 else 1.0,
        avg_mse=float(total_mse / n_kv_heads) if n_kv_heads > 0 else 0.0,
        avg_agreement=float(total_agree / n_kv_heads) if n_kv_heads > 0 else 1.0,
        elapsed_ms=elapsed_ms,
    )

    K_selected = K[:, selected, :]

    return CompactResult(
        selected_indices=selected,
        C_v=C_v,
        K_selected=K_selected,
        stats=stats,
    )

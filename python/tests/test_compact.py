"""
Tests for kv_compact Python package.

Validates the numpy fallback implementation against known properties:
  - Compacted cache is smaller than original
  - Cosine similarity is high (>0.95 for moderate compression)
  - Selected indices are valid and sorted
  - Value refit improves upon naive selection
"""

import numpy as np
import pytest

from kv_compact import compact, compact_layer, CompactParams, CompactResult


# ============================================================================
# Test data generators (match C test patterns)
# ============================================================================

def gen_kv(n_heads=4, seq_len=128, head_dim=64, seed=42):
    """Generate deterministic K, V tensors using sine/cosine patterns."""
    rng = np.random.RandomState(seed)
    K = rng.randn(n_heads, seq_len, head_dim).astype(np.float32) * 0.1
    V = rng.randn(n_heads, seq_len, head_dim).astype(np.float32) * 0.1

    # Add structured signal (sine/cosine)
    for h in range(n_heads):
        for t in range(seq_len):
            freq = (t + 1) / seq_len * np.pi
            for d in range(head_dim):
                K[h, t, d] += 0.5 * np.sin(freq * (d + 1))
                V[h, t, d] += 0.5 * np.cos(freq * (d + 1))

    return K, V


# ============================================================================
# Basic functionality tests
# ============================================================================

class TestCompactBasic:
    def test_compact_returns_result(self):
        K, V = gen_kv(n_heads=2, seq_len=64, head_dim=32)
        result = compact(K, V, target_ratio=0.5)
        assert isinstance(result, CompactResult)

    def test_compact_reduces_length(self):
        K, V = gen_kv(seq_len=128)
        result = compact(K, V, target_ratio=0.5)
        assert result.selected_indices.shape[0] == 64
        assert result.C_v.shape[1] == 64
        assert result.K_selected.shape[1] == 64

    def test_compact_target_count(self):
        K, V = gen_kv(seq_len=100)
        result = compact(K, V, target_count=30)
        assert result.selected_indices.shape[0] == 30

    def test_indices_sorted(self):
        K, V = gen_kv(seq_len=200)
        result = compact(K, V, target_ratio=0.3)
        indices = result.selected_indices
        assert np.all(indices[:-1] <= indices[1:]), "Indices must be sorted"

    def test_indices_valid_range(self):
        K, V = gen_kv(seq_len=150)
        result = compact(K, V, target_ratio=0.5)
        assert np.all(result.selected_indices >= 0)
        assert np.all(result.selected_indices < 150)

    def test_indices_unique(self):
        K, V = gen_kv(seq_len=100)
        result = compact(K, V, target_ratio=0.5)
        assert len(np.unique(result.selected_indices)) == len(result.selected_indices)

    def test_shapes_correct(self):
        n_heads, seq_len, head_dim = 4, 128, 64
        K, V = gen_kv(n_heads=n_heads, seq_len=seq_len, head_dim=head_dim)
        result = compact(K, V, target_ratio=0.5)
        t = result.selected_indices.shape[0]
        assert result.C_v.shape == (n_heads, t, head_dim)
        assert result.K_selected.shape == (n_heads, t, head_dim)

    def test_no_compaction_when_ratio_1(self):
        K, V = gen_kv(seq_len=64)
        result = compact(K, V, target_ratio=1.0)
        assert result.selected_indices.shape[0] == 64


# ============================================================================
# Quality tests
# ============================================================================

class TestCompactQuality:
    def test_cosine_similarity_high(self):
        """At 50% compression, cosine sim should be >0.95."""
        K, V = gen_kv(seq_len=128)
        result = compact(K, V, target_ratio=0.5)
        assert result.stats.avg_cosine_sim > 0.95, (
            f"Cosine sim {result.stats.avg_cosine_sim:.4f} too low"
        )

    def test_cosine_sim_increases_with_ratio(self):
        """Higher keep ratio should give better quality."""
        K, V = gen_kv(seq_len=128)
        r30 = compact(K, V, target_ratio=0.3)
        r70 = compact(K, V, target_ratio=0.7)
        assert r70.stats.avg_cosine_sim >= r30.stats.avg_cosine_sim - 0.01

    def test_refit_better_than_naive(self):
        """Value refit should be better than just selecting original V rows."""
        K, V = gen_kv(n_heads=2, seq_len=128, head_dim=32)
        result = compact(K, V, target_ratio=0.5)

        # Compare: our refitted values vs naive (just the selected V rows)
        V_naive = V[:, result.selected_indices, :]

        # Compute reconstruction error for both
        scale = 1.0 / np.sqrt(K.shape[2])
        refit_err = 0.0
        naive_err = 0.0

        n_q = min(32, result.selected_indices.shape[0])
        Q_ref = K[:, -n_q:, :]

        for h in range(K.shape[0]):
            K_sel = result.K_selected[h]
            logits = Q_ref[h] @ K_sel.T * scale
            A = np.exp(logits - logits.max(axis=-1, keepdims=True))
            A = A / A.sum(axis=-1, keepdims=True)

            logits_all = Q_ref[h] @ K[h].T * scale
            A_all = np.exp(logits_all - logits_all.max(axis=-1, keepdims=True))
            A_all = A_all / A_all.sum(axis=-1, keepdims=True)
            target = A_all @ V[h]

            refit_out = A @ result.C_v[h]
            naive_out = A @ V_naive[h]

            refit_err += np.mean((target - refit_out) ** 2)
            naive_err += np.mean((target - naive_out) ** 2)

        assert refit_err <= naive_err + 1e-6, (
            f"Refit error ({refit_err:.6f}) should be <= naive ({naive_err:.6f})"
        )


# ============================================================================
# Score method tests
# ============================================================================

class TestScoreMethods:
    @pytest.mark.parametrize("method", [0, 1, 2])
    def test_all_score_methods_work(self, method):
        K, V = gen_kv(seq_len=64, n_heads=2, head_dim=16)
        result = compact(K, V, target_ratio=0.5, score_method=method)
        assert result.selected_indices.shape[0] == 32
        assert result.stats.avg_cosine_sim > 0.9


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_single_head(self):
        K, V = gen_kv(n_heads=1, seq_len=64, head_dim=32)
        result = compact(K, V, target_ratio=0.5)
        assert result.C_v.shape[0] == 1

    def test_small_sequence(self):
        K, V = gen_kv(n_heads=2, seq_len=8, head_dim=16)
        result = compact(K, V, target_ratio=0.5)
        assert result.selected_indices.shape[0] == 4

    def test_target_count_exceeds_seq_len(self):
        K, V = gen_kv(seq_len=32)
        result = compact(K, V, target_count=100)
        assert result.selected_indices.shape[0] == 32  # capped at seq_len

    def test_many_heads(self):
        K, V = gen_kv(n_heads=32, seq_len=64, head_dim=16)
        result = compact(K, V, target_ratio=0.5)
        assert result.C_v.shape == (32, 32, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

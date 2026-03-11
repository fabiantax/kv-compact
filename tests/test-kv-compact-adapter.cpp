// Tests for KV cache adapter abstraction
//
// Validates the adapter layer that sits between cache I/O and compaction math:
//   - GQA adapter: identity decode/encode round-trip
//   - MLA adapter: latent projection decode + least-squares encode
//   - Layer classifiers: uniform and hybrid
//   - Factory: correct adapter creation from architecture descriptors
//   - Interface contracts: geometry, storage, polymorphic dispatch

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "kv-compact-math.h"
#include "kv-compact-adapter.h"

static const float EPS = 1e-4f;

static bool approx_eq(float a, float b, float tol = EPS) {
    return fabsf(a - b) < tol;
}

static float max_abs_diff(const float * a, const float * b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// ============================================================================
// GQA adapter — geometry
// ============================================================================

static void test_gqa_geometry() {
    printf("  test_gqa_geometry...");
    gqa_adapter adapter(128, 128, 8);

    auto geom = adapter.geometry();
    assert(geom.d_k == 128);
    assert(geom.d_v == 128);
    assert(geom.n_head_kv == 8);

    auto stor = adapter.storage();
    assert(stor.dim == 128 * 8);
    printf(" OK\n");
}

static void test_gqa_geometry_gqa_ratio() {
    printf("  test_gqa_geometry_gqa_ratio...");
    // GQA with 4 KV heads, different d_k/d_v
    gqa_adapter adapter(64, 96, 4);

    auto geom = adapter.geometry();
    assert(geom.d_k == 64);
    assert(geom.d_v == 96);
    assert(geom.n_head_kv == 4);
    printf(" OK\n");
}

// ============================================================================
// GQA adapter — decode extracts correct head slices
// ============================================================================

static void test_gqa_decode_single_head() {
    printf("  test_gqa_decode_single_head...");

    const int d_k = 4, d_v = 4, n_heads = 2, T = 3;
    gqa_adapter adapter(d_k, d_v, n_heads);

    // Cache K: [T=3, n_heads*d_k=8]
    // Each row: [head0_k0..k3, head1_k0..k3]
    float cache_k[3 * 8];
    float cache_v[3 * 8];
    for (int i = 0; i < 3 * 8; i++) {
        cache_k[i] = (float)(i + 1);
        cache_v[i] = (float)(100 + i);
    }

    // Decode head 0
    float K0[3 * 4], V0[3 * 4];
    adapter.decode(cache_k, cache_v, T, 0, K0, V0);

    // Head 0 of row 0: cache_k[0..3] = {1,2,3,4}
    assert(approx_eq(K0[0], 1.0f));
    assert(approx_eq(K0[1], 2.0f));
    assert(approx_eq(K0[2], 3.0f));
    assert(approx_eq(K0[3], 4.0f));

    // Head 0 of row 1: cache_k[8..11] = {9,10,11,12}
    assert(approx_eq(K0[4], 9.0f));
    assert(approx_eq(K0[5], 10.0f));

    // Decode head 1
    float K1[3 * 4], V1[3 * 4];
    adapter.decode(cache_k, cache_v, T, 1, K1, V1);

    // Head 1 of row 0: cache_k[4..7] = {5,6,7,8}
    assert(approx_eq(K1[0], 5.0f));
    assert(approx_eq(K1[1], 6.0f));
    assert(approx_eq(K1[2], 7.0f));
    assert(approx_eq(K1[3], 8.0f));

    printf(" OK\n");
}

// ============================================================================
// GQA adapter — encode/decode round-trip
// ============================================================================

static void test_gqa_round_trip() {
    printf("  test_gqa_round_trip...");

    const int d_k = 4, d_v = 3, n_heads = 2, T = 5;
    gqa_adapter adapter(d_k, d_v, n_heads);

    // Build interleaved cache
    float cache_k[5 * 8]; // T * n_heads * d_k
    float cache_v[5 * 6]; // T * n_heads * d_v
    for (int i = 0; i < 5 * 8; i++) cache_k[i] = (float)(i * 0.1f);
    for (int i = 0; i < 5 * 6; i++) cache_v[i] = (float)(i * 0.2f);

    for (int h = 0; h < n_heads; h++) {
        // Decode
        float K[5 * 4], V[5 * 3];
        adapter.decode(cache_k, cache_v, T, h, K, V);

        // Encode back
        float cache_k_out[5 * 8] = {};
        float cache_v_out[5 * 6] = {};
        adapter.encode(K, V, T, h, cache_k_out, cache_v_out);

        // Verify head h's slice matches original
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < d_k; j++) {
                float orig = cache_k[i * n_heads * d_k + h * d_k + j];
                float rt   = cache_k_out[i * n_heads * d_k + h * d_k + j];
                assert(approx_eq(orig, rt));
            }
            for (int j = 0; j < d_v; j++) {
                float orig = cache_v[i * n_heads * d_v + h * d_v + j];
                float rt   = cache_v_out[i * n_heads * d_v + h * d_v + j];
                assert(approx_eq(orig, rt));
            }
        }
    }

    printf(" OK\n");
}

// ============================================================================
// GQA adapter — encode with compacted (fewer) tokens
// ============================================================================

static void test_gqa_encode_compacted() {
    printf("  test_gqa_encode_compacted...");

    const int d_k = 2, d_v = 2, n_heads = 1, t = 3;
    gqa_adapter adapter(d_k, d_v, n_heads);

    float K_compacted[] = {1, 2, 3, 4, 5, 6};
    float V_compacted[] = {10, 20, 30, 40, 50, 60};

    float cache_k_out[3 * 2] = {};
    float cache_v_out[3 * 2] = {};
    adapter.encode(K_compacted, V_compacted, t, 0, cache_k_out, cache_v_out);

    assert(approx_eq(cache_k_out[0], 1.0f));
    assert(approx_eq(cache_k_out[1], 2.0f));
    assert(approx_eq(cache_v_out[4], 50.0f));
    assert(approx_eq(cache_v_out[5], 60.0f));

    printf(" OK\n");
}

// ============================================================================
// MLA adapter — geometry
// ============================================================================

static void test_mla_geometry() {
    printf("  test_mla_geometry...");

    // DeepSeek-V3-like: d_c=512, d_rope=64, d_k_nope=128, d_v=128, 1 KV head
    float dummy_wuk[1], dummy_wuv[1];
    mla_adapter adapter(512, 64, 128, 128, 1, dummy_wuk, dummy_wuv);

    auto geom = adapter.geometry();
    assert(geom.d_k == 128 + 64);  // d_k_nope + d_rope
    assert(geom.d_v == 128);
    assert(geom.n_head_kv == 1);

    auto stor = adapter.storage();
    assert(stor.dim == 512 + 64);  // d_c + d_rope
    printf(" OK\n");
}

// ============================================================================
// MLA adapter — decode with identity projection
// ============================================================================

static void test_mla_decode_identity_projection() {
    printf("  test_mla_decode_identity_projection...");

    // Simple case: d_c=3, d_rope=1, d_k_nope=2, d_v=2, 1 head
    // W_uk = [[1,0,0],[0,1,0]]  (2x3) — takes first 2 dims of latent
    // W_uv = [[0,0,1],[0,1,0]]  (2x3) — takes dims 2,1 of latent
    const int d_c = 3, d_rope = 1, d_k_nope = 2, d_v = 2, n_heads = 1, T = 2;

    float W_uk[] = {1, 0, 0,  // row 0
                    0, 1, 0}; // row 1
    float W_uv[] = {0, 0, 1,  // row 0: picks dim 2
                    0, 1, 0}; // row 1: picks dim 1

    mla_adapter adapter(d_c, d_rope, d_k_nope, d_v, n_heads, W_uk, W_uv);

    // Cache K: [T=2, d_c + d_rope = 4]
    // Token 0: latent=[1,2,3], rope=[10]
    // Token 1: latent=[4,5,6], rope=[20]
    float cache_k[] = {1, 2, 3, 10,
                       4, 5, 6, 20};

    // Cache V: [T=2, d_c=3]
    float cache_v[] = {1, 2, 3,
                       4, 5, 6};

    float K_out[2 * 3], V_out[2 * 2]; // d_k = d_k_nope+d_rope = 3
    adapter.decode(cache_k, cache_v, T, 0, K_out, V_out);

    // K_nope for token 0: latent=[1,2,3] @ W_uk^T = [1*1+2*0+3*0, 1*0+2*1+3*0] = [1, 2]
    // K for token 0: [1, 2, 10] (nope + rope)
    assert(approx_eq(K_out[0], 1.0f));
    assert(approx_eq(K_out[1], 2.0f));
    assert(approx_eq(K_out[2], 10.0f));  // rope

    // K for token 1: latent=[4,5,6] → K_nope=[4, 5], rope=[20]
    assert(approx_eq(K_out[3], 4.0f));
    assert(approx_eq(K_out[4], 5.0f));
    assert(approx_eq(K_out[5], 20.0f));

    // V for token 0: latent=[1,2,3] @ W_uv^T → [0*1+0*2+1*3, 0*1+1*2+0*3] = [3, 2]
    assert(approx_eq(V_out[0], 3.0f));
    assert(approx_eq(V_out[1], 2.0f));

    // V for token 1: latent=[4,5,6] → [6, 5]
    assert(approx_eq(V_out[2], 6.0f));
    assert(approx_eq(V_out[3], 5.0f));

    printf(" OK\n");
}

// ============================================================================
// MLA adapter — encode projects V back to latent space
// ============================================================================

static void test_mla_encode_projects_back() {
    printf("  test_mla_encode_projects_back...");

    // d_c=2, d_rope=1, d_k_nope=2, d_v=2, 1 head
    // W_uk = I(2x2), W_uv = I(2x2) — identity projections
    const int d_c = 2, d_rope = 1, d_k_nope = 2, d_v = 2, n_heads = 1;

    float W_uk[] = {1, 0, 0, 1};
    float W_uv[] = {1, 0, 0, 1};

    mla_adapter adapter(d_c, d_rope, d_k_nope, d_v, n_heads, W_uk, W_uv);

    // Compacted K: [t=2, d_k=3]  (d_k_nope + d_rope)
    float K_comp[] = {1, 2, 10,   // nope=[1,2], rope=[10]
                      3, 4, 20};  // nope=[3,4], rope=[20]

    // Compacted V: [t=2, d_v=2]
    float V_comp[] = {5, 6,
                      7, 8};

    // With identity W_uv, the LS projection should recover V as latent
    float cache_k_out[2 * 3] = {}; // [t, d_c+d_rope]
    float cache_v_out[2 * 2] = {}; // [t, d_c]

    adapter.encode(K_comp, V_comp, 2, 0, cache_k_out, cache_v_out);

    // cache_v_out should be close to V_comp (identity projection)
    assert(approx_eq(cache_v_out[0], 5.0f, 0.1f));
    assert(approx_eq(cache_v_out[1], 6.0f, 0.1f));
    assert(approx_eq(cache_v_out[2], 7.0f, 0.1f));
    assert(approx_eq(cache_v_out[3], 8.0f, 0.1f));

    // cache_k_out should have latent (=V via identity) + rope
    assert(approx_eq(cache_k_out[2], 10.0f));  // rope preserved
    assert(approx_eq(cache_k_out[5], 20.0f));  // rope preserved

    printf(" OK\n");
}

// ============================================================================
// MLA adapter — decode/encode round-trip with non-trivial projection
// ============================================================================

static void test_mla_round_trip() {
    printf("  test_mla_round_trip...");

    // d_c=4, d_rope=2, d_k_nope=3, d_v=3, 1 head
    // W_uk: [3, 4] — 3 key dims from 4 latent dims
    // W_uv: [3, 4] — 3 val dims from 4 latent dims
    const int d_c = 4, d_rope = 2, d_k_nope = 3, d_v = 3, T = 3;

    // Random but deterministic weights
    float W_uk[3 * 4] = {
        0.5f, 0.1f, -0.3f, 0.2f,
        0.1f, 0.6f,  0.2f, -0.1f,
       -0.2f, 0.3f,  0.7f,  0.1f
    };
    float W_uv[3 * 4] = {
        0.4f, -0.2f, 0.1f,  0.5f,
        0.3f,  0.5f, -0.1f, 0.2f,
       -0.1f,  0.2f,  0.6f, 0.3f
    };

    mla_adapter adapter(d_c, d_rope, d_k_nope, d_v, 1, W_uk, W_uv);

    // Cache K: [T=3, d_c+d_rope=6]
    float cache_k[3 * 6] = {
        1, 2, 3, 4,  10, 11,    // latent=[1,2,3,4], rope=[10,11]
        5, 6, 7, 8,  20, 21,
        9, 10, 11, 12, 30, 31
    };

    // Cache V: [T=3, d_c=4]
    float cache_v[3 * 4] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    // Decode
    const int d_k_full = d_k_nope + d_rope; // 5
    float K_dec[3 * 5], V_dec[3 * 3];
    adapter.decode(cache_k, cache_v, T, 0, K_dec, V_dec);

    // Encode back
    float cache_k_rt[3 * 6] = {};
    float cache_v_rt[3 * 4] = {};
    adapter.encode(K_dec, V_dec, T, 0, cache_k_rt, cache_v_rt);

    // RoPE portion should be exactly preserved
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < d_rope; j++) {
            float orig = cache_k[i * 6 + d_c + j];
            float rt   = cache_k_rt[i * 6 + d_c + j];
            assert(approx_eq(orig, rt, 1e-5f));
        }
    }

    // The latent round-trip won't be exact (LS projection is lossy when d_v < d_c)
    // but the V values recovered from the round-tripped latent should match
    // Verify: decode(encode(decode(original))) ≈ decode(original) in V space
    float V_rt[3 * 3];
    adapter.decode(cache_k_rt, cache_v_rt, T, 0, K_dec, V_rt);

    float v_err = max_abs_diff(V_dec, V_rt, T * d_v);
    // With well-conditioned 3→4→3 projections, error should be small
    assert(v_err < 1.0f);  // relaxed — LS on underdetermined system

    printf(" OK (V round-trip error: %.6f)\n", v_err);
}

// ============================================================================
// MLA adapter — multi-head
// ============================================================================

static void test_mla_multi_head() {
    printf("  test_mla_multi_head...");

    const int d_c = 4, d_rope = 1, d_k_nope = 2, d_v = 2, n_heads = 2, T = 2;

    // W_uk: [n_heads * d_k_nope, d_c] = [4, 4]
    float W_uk[4 * 4] = {
        1, 0, 0, 0,  // head 0, k_nope dim 0
        0, 1, 0, 0,  // head 0, k_nope dim 1
        0, 0, 1, 0,  // head 1, k_nope dim 0
        0, 0, 0, 1   // head 1, k_nope dim 1
    };
    // W_uv: [n_heads * d_v, d_c] = [4, 4]
    float W_uv[4 * 4] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    mla_adapter adapter(d_c, d_rope, d_k_nope, d_v, n_heads, W_uk, W_uv);

    float cache_k[2 * 5] = { // [T, d_c+d_rope=5]
        1, 2, 3, 4, 10,
        5, 6, 7, 8, 20
    };
    float cache_v[2 * 4] = { // [T, d_c=4]
        1, 2, 3, 4,
        5, 6, 7, 8
    };

    // Decode head 0: K_nope picks dims 0,1; V picks dims 0,1
    float K0[2 * 3], V0[2 * 2];
    adapter.decode(cache_k, cache_v, T, 0, K0, V0);

    assert(approx_eq(K0[0], 1.0f));  // latent[0]
    assert(approx_eq(K0[1], 2.0f));  // latent[1]
    assert(approx_eq(K0[2], 10.0f)); // rope
    assert(approx_eq(V0[0], 1.0f));  // latent[0]
    assert(approx_eq(V0[1], 2.0f));  // latent[1]

    // Decode head 1: K_nope picks dims 2,3; V picks dims 2,3
    float K1[2 * 3], V1[2 * 2];
    adapter.decode(cache_k, cache_v, T, 1, K1, V1);

    assert(approx_eq(K1[0], 3.0f));  // latent[2]
    assert(approx_eq(K1[1], 4.0f));  // latent[3]
    assert(approx_eq(K1[2], 10.0f)); // rope (shared)
    assert(approx_eq(V1[0], 3.0f));  // latent[2]
    assert(approx_eq(V1[1], 4.0f));  // latent[3]

    printf(" OK\n");
}

// ============================================================================
// Uniform layer classifier
// ============================================================================

static void test_uniform_classifier() {
    printf("  test_uniform_classifier...");

    uniform_classifier cls(32);
    assert(cls.n_layers() == 32);
    assert(cls.has_kv_cache(0)  == true);
    assert(cls.has_kv_cache(15) == true);
    assert(cls.has_kv_cache(31) == true);
    assert(cls.has_kv_cache(-1) == true);  // no bounds check in uniform
    assert(cls.has_kv_cache(32) == true);

    printf(" OK\n");
}

// ============================================================================
// Hybrid layer classifier
// ============================================================================

static void test_hybrid_classifier() {
    printf("  test_hybrid_classifier...");

    // Qwen3.5-like: 24 layers, only layers 0,6,12,18 use full attention
    std::vector<bool> mask(24, false);
    mask[0] = true; mask[6] = true; mask[12] = true; mask[18] = true;

    hybrid_classifier cls(24, mask);
    assert(cls.n_layers() == 24);
    assert(cls.has_kv_cache(0)  == true);
    assert(cls.has_kv_cache(1)  == false);
    assert(cls.has_kv_cache(6)  == true);
    assert(cls.has_kv_cache(7)  == false);
    assert(cls.has_kv_cache(12) == true);
    assert(cls.has_kv_cache(18) == true);
    assert(cls.has_kv_cache(23) == false);

    // Out of bounds → false
    assert(cls.has_kv_cache(-1) == false);
    assert(cls.has_kv_cache(24) == false);

    printf(" OK\n");
}

static void test_hybrid_classifier_mismatched_size() {
    printf("  test_hybrid_classifier_mismatched_size...");

    bool threw = false;
    try {
        std::vector<bool> mask(10, true);
        hybrid_classifier cls(24, mask);  // should throw
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    printf(" OK\n");
}

// ============================================================================
// Factory — GQA
// ============================================================================

static void test_factory_gqa() {
    printf("  test_factory_gqa...");

    attention_arch arch;
    arch.type = "gqa";
    arch.d_k = 128;
    arch.d_v = 128;
    arch.n_head_kv = 8;

    auto adapter = make_adapter(arch);
    assert(adapter != nullptr);

    auto geom = adapter->geometry();
    assert(geom.d_k == 128);
    assert(geom.d_v == 128);
    assert(geom.n_head_kv == 8);

    printf(" OK\n");
}

static void test_factory_default_is_gqa() {
    printf("  test_factory_default_is_gqa...");

    attention_arch arch;
    arch.type = "mha";  // unknown type defaults to GQA
    arch.d_k = 64;
    arch.d_v = 64;
    arch.n_head_kv = 32;

    auto adapter = make_adapter(arch);
    assert(adapter != nullptr);

    auto geom = adapter->geometry();
    assert(geom.d_k == 64);
    assert(geom.n_head_kv == 32);

    printf(" OK\n");
}

// ============================================================================
// Factory — MLA
// ============================================================================

static void test_factory_mla() {
    printf("  test_factory_mla...");

    float W_uk[128 * 512];
    float W_uv[128 * 512];
    memset(W_uk, 0, sizeof(W_uk));
    memset(W_uv, 0, sizeof(W_uv));

    attention_arch arch;
    arch.type = "mla";
    arch.d_k = 192;   // d_k_nope + d_rope (informational)
    arch.d_v = 128;
    arch.n_head_kv = 1;
    arch.d_c = 512;
    arch.d_rope = 64;
    arch.d_k_nope = 128;
    arch.W_uk = W_uk;
    arch.W_uv = W_uv;

    auto adapter = make_adapter(arch);
    assert(adapter != nullptr);

    auto geom = adapter->geometry();
    assert(geom.d_k == 128 + 64);  // d_k_nope + d_rope
    assert(geom.d_v == 128);
    assert(geom.n_head_kv == 1);

    auto stor = adapter->storage();
    assert(stor.dim == 512 + 64);

    printf(" OK\n");
}

static void test_factory_mla_missing_weights() {
    printf("  test_factory_mla_missing_weights...");

    attention_arch arch;
    arch.type = "mla";
    arch.d_v = 128;
    arch.n_head_kv = 1;
    arch.d_c = 512;
    arch.d_rope = 64;
    arch.d_k_nope = 128;
    // Missing W_uk and W_uv

    bool threw = false;
    try {
        auto adapter = make_adapter(arch);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    printf(" OK\n");
}

// ============================================================================
// Factory — classifier
// ============================================================================

static void test_factory_classifier_uniform() {
    printf("  test_factory_classifier_uniform...");

    attention_arch arch;
    arch.type = "gqa";

    auto cls = make_classifier(arch, 32);
    assert(cls->n_layers() == 32);
    assert(cls->has_kv_cache(0) == true);
    assert(cls->has_kv_cache(31) == true);

    printf(" OK\n");
}

static void test_factory_classifier_hybrid() {
    printf("  test_factory_classifier_hybrid...");

    attention_arch arch;
    arch.type = "hybrid";
    arch.attention_layers = {true, false, false, true, false, false};

    auto cls = make_classifier(arch, 6);
    assert(cls->n_layers() == 6);
    assert(cls->has_kv_cache(0) == true);
    assert(cls->has_kv_cache(1) == false);
    assert(cls->has_kv_cache(3) == true);
    assert(cls->has_kv_cache(4) == false);

    printf(" OK\n");
}

// ============================================================================
// Polymorphic dispatch — adapter via base pointer
// ============================================================================

static void test_polymorphic_dispatch() {
    printf("  test_polymorphic_dispatch...");

    const int d_k = 4, d_v = 4, n_heads = 1, T = 2;

    // Create via factory
    attention_arch arch;
    arch.type = "gqa";
    arch.d_k = d_k;
    arch.d_v = d_v;
    arch.n_head_kv = n_heads;

    std::unique_ptr<kv_adapter> adapter = make_adapter(arch);

    float cache_k[2 * 4] = {1, 2, 3, 4,  5, 6, 7, 8};
    float cache_v[2 * 4] = {10, 20, 30, 40,  50, 60, 70, 80};

    float K_out[2 * 4], V_out[2 * 4];
    adapter->decode(cache_k, cache_v, T, 0, K_out, V_out);

    assert(approx_eq(K_out[0], 1.0f));
    assert(approx_eq(V_out[3], 40.0f));
    assert(approx_eq(K_out[4], 5.0f));

    printf(" OK\n");
}

// ============================================================================
// Integration: GQA adapter with compaction pipeline functions
// ============================================================================

static void test_gqa_with_compaction_pipeline() {
    printf("  test_gqa_with_compaction_pipeline...");

    // Verify that adapter-decoded K/V works correctly with compact_head_highest_attn
    const int d_k = 4, d_v = 4, n_heads = 2, T = 8, n_q = 4, t = 4;

    gqa_adapter adapter(d_k, d_v, n_heads);

    // Build a synthetic cache with known structure
    std::vector<float> cache_k(T * n_heads * d_k);
    std::vector<float> cache_v(T * n_heads * d_v);

    // Fill with structured data: head h, token i, dim j → h*1000 + i*10 + j
    for (int i = 0; i < T; i++) {
        for (int h = 0; h < n_heads; h++) {
            for (int j = 0; j < d_k; j++) {
                cache_k[i * n_heads * d_k + h * d_k + j] = (float)(h * 1000 + i * 10 + j);
            }
            for (int j = 0; j < d_v; j++) {
                cache_v[i * n_heads * d_v + h * d_v + j] = (float)(h * 1000 + i * 10 + j + 100);
            }
        }
    }

    // Decode head 0
    std::vector<float> K(T * d_k), V(T * d_v);
    adapter.decode(cache_k.data(), cache_v.data(), T, 0, K.data(), V.data());

    // Use last n_q tokens as reference queries
    const float * Q_ref = K.data() + (T - n_q) * d_k;

    // Run compaction via the standard function
    auto result = compact_head_highest_attn(
        K.data(), V.data(), Q_ref, T, n_q, d_k, d_v, t);

    assert((int)result.selected_indices.size() == t);
    assert((int)result.beta.size() == t);
    assert((int)result.C_v.size() == t * d_v);

    printf(" OK\n");
}

// ============================================================================
// Edge cases
// ============================================================================

static void test_gqa_single_token() {
    printf("  test_gqa_single_token...");

    gqa_adapter adapter(2, 2, 1);

    float cache_k[] = {1, 2};
    float cache_v[] = {3, 4};
    float K[2], V[2];

    adapter.decode(cache_k, cache_v, 1, 0, K, V);
    assert(approx_eq(K[0], 1.0f));
    assert(approx_eq(V[1], 4.0f));

    printf(" OK\n");
}

static void test_gqa_single_head_mqa() {
    printf("  test_gqa_single_head_mqa...");

    // MQA: 1 KV head, many Q heads
    gqa_adapter adapter(64, 64, 1);

    auto geom = adapter.geometry();
    assert(geom.n_head_kv == 1);
    assert(geom.d_k == 64);

    printf(" OK\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== KV Adapter Tests ===\n\n");

    printf("GQA adapter:\n");
    test_gqa_geometry();
    test_gqa_geometry_gqa_ratio();
    test_gqa_decode_single_head();
    test_gqa_round_trip();
    test_gqa_encode_compacted();
    test_gqa_single_token();
    test_gqa_single_head_mqa();

    printf("\nMLA adapter:\n");
    test_mla_geometry();
    test_mla_decode_identity_projection();
    test_mla_encode_projects_back();
    test_mla_round_trip();
    test_mla_multi_head();

    printf("\nLayer classifiers:\n");
    test_uniform_classifier();
    test_hybrid_classifier();
    test_hybrid_classifier_mismatched_size();

    printf("\nFactory:\n");
    test_factory_gqa();
    test_factory_default_is_gqa();
    test_factory_mla();
    test_factory_mla_missing_weights();
    test_factory_classifier_uniform();
    test_factory_classifier_hybrid();

    printf("\nPolymorphic dispatch:\n");
    test_polymorphic_dispatch();

    printf("\nIntegration:\n");
    test_gqa_with_compaction_pipeline();

    printf("\n=== All adapter tests passed ===\n");
    return 0;
}

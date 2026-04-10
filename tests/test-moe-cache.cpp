// Unit tests for kv-compact-moe-cache.h (standalone, no llama.cpp needed)

#include "../include/kv-compact-moe-cache.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void test_init() {
    moe_expert_cache cache;
    assert(!cache.enabled);

    cache.init(4, 32);
    assert(cache.enabled);
    assert(cache.n_layers == 4);
    assert(cache.n_experts == 32);
    assert(cache.ema.size() == 4);
    assert(cache.ema[0].size() == 32);

    // All EMA should be zero initially
    for (int l = 0; l < 4; l++) {
        for (int e = 0; e < 32; e++) {
            assert(cache.ema[l][e] == 0.0f);
        }
    }
    printf("  PASS: init\n");
}

static void test_update_ema() {
    moe_expert_cache cache;
    cache.init(1, 8);
    cache.alpha = 0.5f;

    // Update with experts {0, 2, 4}
    int32_t ids[] = {0, 2, 4};
    cache.update(0, ids, 3);

    // Selected experts should have ema = alpha = 0.5
    assert(std::abs(cache.ema[0][0] - 0.5f) < 1e-6);
    assert(std::abs(cache.ema[0][2] - 0.5f) < 1e-6);
    assert(std::abs(cache.ema[0][4] - 0.5f) < 1e-6);

    // Non-selected should have ema = 0
    assert(std::abs(cache.ema[0][1]) < 1e-6);
    assert(std::abs(cache.ema[0][3]) < 1e-6);

    // Update again with {0, 1}
    int32_t ids2[] = {0, 1};
    cache.update(0, ids2, 2);

    // Expert 0: was 0.5, decay to 0.25, add 0.5 = 0.75
    assert(std::abs(cache.ema[0][0] - 0.75f) < 1e-6);
    // Expert 1: was 0, decay stays 0, add 0.5 = 0.5
    assert(std::abs(cache.ema[0][1] - 0.5f) < 1e-6);
    // Expert 2: was 0.5, decay to 0.25
    assert(std::abs(cache.ema[0][2] - 0.25f) < 1e-6);
    // Expert 4: was 0.5, decay to 0.25
    assert(std::abs(cache.ema[0][4] - 0.25f) < 1e-6);

    printf("  PASS: update_ema\n");
}

static void test_compute_bias() {
    moe_expert_cache cache;
    cache.init(1, 8);
    cache.cache_size = 3;
    cache.bias_strength = 1.0f;
    cache.alpha = 1.0f;  // instant tracking for test

    // Make experts 1, 3, 5 hot
    int32_t ids[] = {1, 3, 5};
    cache.update(0, ids, 3);

    float bias[8];
    cache.compute_bias(0, bias);

    // Hot experts should get bias=1.0
    assert(std::abs(bias[1] - 1.0f) < 1e-6);
    assert(std::abs(bias[3] - 1.0f) < 1e-6);
    assert(std::abs(bias[5] - 1.0f) < 1e-6);

    // Cold experts should get bias=0.0
    assert(std::abs(bias[0]) < 1e-6);
    assert(std::abs(bias[2]) < 1e-6);
    assert(std::abs(bias[4]) < 1e-6);
    assert(std::abs(bias[6]) < 1e-6);
    assert(std::abs(bias[7]) < 1e-6);

    printf("  PASS: compute_bias\n");
}

static void test_bias_vec_convenience() {
    moe_expert_cache cache;
    cache.init(1, 4);
    cache.cache_size = 2;
    cache.bias_strength = 0.5f;
    cache.alpha = 1.0f;

    int32_t ids[] = {0, 3};
    cache.update(0, ids, 2);

    auto bias = cache.compute_bias_vec(0);
    assert(bias.size() == 4);
    assert(std::abs(bias[0] - 0.5f) < 1e-6);
    assert(std::abs(bias[3] - 0.5f) < 1e-6);
    assert(std::abs(bias[1]) < 1e-6);
    assert(std::abs(bias[2]) < 1e-6);

    printf("  PASS: bias_vec_convenience\n");
}

static void test_biased_top_k() {
    // 4 experts, top-2, uniform logits
    float logits[] = {0.0f, 0.0f, 0.0f, 0.0f};

    // No bias: any 2 experts valid (softmax is uniform)
    auto sel_no_bias = moe_expert_cache::biased_top_k(logits, nullptr, 4, 2);
    assert(sel_no_bias.size() == 2);

    // Bias strongly toward experts 2,3
    float bias[] = {0.0f, 0.0f, 10.0f, 10.0f};
    auto sel_biased = moe_expert_cache::biased_top_k(logits, bias, 4, 2);
    assert(sel_biased.size() == 2);
    // Should select experts 2 and 3
    assert((sel_biased[0] == 2 || sel_biased[0] == 3));
    assert((sel_biased[1] == 2 || sel_biased[1] == 3));
    assert(sel_biased[0] != sel_biased[1]);

    printf("  PASS: biased_top_k\n");
}

static void test_reset() {
    moe_expert_cache cache;
    cache.init(2, 4);
    cache.alpha = 1.0f;

    int32_t ids[] = {0, 1};
    cache.update(0, ids, 2);
    cache.update(1, ids, 2);

    cache.reset();
    for (int l = 0; l < 2; l++) {
        for (int e = 0; e < 4; e++) {
            assert(cache.ema[l][e] == 0.0f);
        }
    }

    printf("  PASS: reset\n");
}

static void test_multi_layer_independence() {
    moe_expert_cache cache;
    cache.init(3, 8);
    cache.alpha = 1.0f;

    int32_t ids0[] = {0, 1};
    int32_t ids1[] = {2, 3};
    int32_t ids2[] = {4, 5};

    cache.update(0, ids0, 2);
    cache.update(1, ids1, 2);
    cache.update(2, ids2, 2);

    // Layer 0 should only have experts 0,1
    assert(cache.ema[0][0] > 0.0f);
    assert(cache.ema[0][1] > 0.0f);
    assert(cache.ema[0][2] == 0.0f);

    // Layer 1 should only have experts 2,3
    assert(cache.ema[1][0] == 0.0f);
    assert(cache.ema[1][2] > 0.0f);
    assert(cache.ema[1][3] > 0.0f);

    // Layer 2 should only have experts 4,5
    assert(cache.ema[2][4] > 0.0f);
    assert(cache.ema[2][5] > 0.0f);
    assert(cache.ema[2][0] == 0.0f);

    printf("  PASS: multi_layer_independence\n");
}

static void test_ema_decay_convergence() {
    moe_expert_cache cache;
    cache.init(1, 4);
    cache.alpha = 0.1f;

    // Repeatedly select expert 0 for many steps
    int32_t ids[] = {0};
    for (int t = 0; t < 100; t++) {
        cache.update(0, ids, 1);
    }

    // EMA for expert 0 should converge to ~1.0 (alpha / alpha = 1.0 at steady state)
    assert(cache.ema[0][0] > 0.9f);

    // Other experts should decay to ~0
    assert(cache.ema[0][1] < 0.01f);
    assert(cache.ema[0][2] < 0.01f);
    assert(cache.ema[0][3] < 0.01f);

    printf("  PASS: ema_decay_convergence\n");
}

int main() {
    printf("=== MoE Expert Cache Unit Tests ===\n\n");

    test_init();
    test_update_ema();
    test_compute_bias();
    test_bias_vec_convenience();
    test_biased_top_k();
    test_reset();
    test_multi_layer_independence();
    test_ema_decay_convergence();

    printf("\nAll C++ MoE cache tests passed!\n");
    return 0;
}

#pragma once

// kv-compact-moe-cache.h — Cache-Aware MoE Expert Routing
//
// Implements EMA-based expert caching with dynamic routing bias for MoE models.
// Paper: "Cache-Aware Routing" (2412.00099) — 2x decode speedup, <0.1% quality loss.
//
// Architecture:
//   Per token decode:
//     1. set_inputs() → push EMA bias tensor into graph
//     2. build_moe_ffn() → add bias to selection_probs before top-k
//     3. graph_compute() → run inference with biased routing
//     4. post-compute → extract ffn_moe_topk tensors, update EMA
//
// This header is self-contained (no ggml/llama deps) for standalone testing.
// The llama.cpp integration uses patches/apply.sh to wire this into the graph.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

struct moe_expert_cache {
    bool  enabled        = false;
    int   n_layers       = 0;
    int   n_experts      = 0;
    int   cache_size     = 32;     // number of "hot" experts per layer
    float alpha          = 0.1f;   // EMA decay rate
    float bias_strength  = 0.3f;   // routing bias magnitude for cached experts

    // per-layer EMA scores: ema[layer][expert]
    std::vector<std::vector<float>> ema;

    void init(int nl, int ne) {
        n_layers  = nl;
        n_experts = ne;
        ema.resize(nl);
        for (int l = 0; l < nl; l++) {
            ema[l].assign(ne, 0.0f);
        }
        enabled = true;
    }

    void reset() {
        for (int l = 0; l < n_layers; l++) {
            std::fill(ema[l].begin(), ema[l].end(), 0.0f);
        }
    }

    // Update EMA for a single layer after observing which experts were selected.
    // expert_ids: array of n_used expert indices (from ffn_moe_topk tensor)
    void update(int layer, const int32_t * expert_ids, int n_used) {
        assert(layer >= 0 && layer < n_layers);
        auto & e = ema[layer];

        // Decay all experts
        const float decay = 1.0f - alpha;
        for (int i = 0; i < n_experts; i++) {
            e[i] *= decay;
        }

        // Boost selected experts
        for (int i = 0; i < n_used; i++) {
            const int eid = expert_ids[i];
            if (eid >= 0 && eid < n_experts) {
                e[eid] += alpha;
            }
        }
    }

    // Compute bias vector for a layer: top cache_size experts by EMA get +bias_strength, others 0.
    // out_bias must have space for n_experts floats.
    void compute_bias(int layer, float * out_bias) const {
        assert(layer >= 0 && layer < n_layers);
        const auto & e = ema[layer];

        // Find the cache_size-th largest EMA score (threshold)
        std::vector<float> sorted_ema(e.begin(), e.end());
        std::sort(sorted_ema.begin(), sorted_ema.end(), std::greater<float>());

        const int effective_cache = std::min(cache_size, n_experts);
        const float threshold = (effective_cache > 0 && effective_cache <= n_experts)
            ? sorted_ema[effective_cache - 1]
            : 0.0f;

        for (int i = 0; i < n_experts; i++) {
            out_bias[i] = (e[i] >= threshold && threshold > 0.0f) ? bias_strength : 0.0f;
        }
    }

    // Convenience: compute bias into a returned vector
    std::vector<float> compute_bias_vec(int layer) const {
        std::vector<float> bias(n_experts, 0.0f);
        if (enabled && n_experts > 0) {
            compute_bias(layer, bias.data());
        }
        return bias;
    }

    // Simulate cache-aware routing: given original logits and EMA state,
    // return the expert indices that would be selected with bias applied.
    // This is for offline analysis / testing only.
    static std::vector<int> biased_top_k(
            const float * logits,
            const float * bias,
            int n_experts,
            int n_expert_used) {
        // Apply softmax to logits first
        std::vector<float> probs(n_experts);
        float max_logit = *std::max_element(logits, logits + n_experts);
        float sum_exp = 0.0f;
        for (int i = 0; i < n_experts; i++) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum_exp += probs[i];
        }
        for (int i = 0; i < n_experts; i++) {
            probs[i] /= sum_exp;
        }

        // Add bias to selection probs (same as build_moe_ffn does)
        std::vector<float> selection(n_experts);
        for (int i = 0; i < n_experts; i++) {
            selection[i] = probs[i] + (bias ? bias[i] : 0.0f);
        }

        // Top-k selection
        std::vector<int> indices(n_experts);
        for (int i = 0; i < n_experts; i++) indices[i] = i;
        std::partial_sort(indices.begin(), indices.begin() + n_expert_used, indices.end(),
            [&selection](int a, int b) { return selection[a] > selection[b]; });

        return std::vector<int>(indices.begin(), indices.begin() + n_expert_used);
    }
};

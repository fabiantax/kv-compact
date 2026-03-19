#!/bin/bash
# Apply KV cache attention bias patch to llama.cpp
# This adds per-cell attention biases for KV cache compaction (beta from attention matching).
#
# Usage: ./patches/apply.sh <llama_cpp_source_dir>

set -euo pipefail

LLAMA_DIR="${1:?Usage: $0 <llama_cpp_source_dir>}"

echo "Applying attention bias patch to: $LLAMA_DIR"

# ============================================================================
# 1. llama-kv-cells.h — add bias vector and accessors
# ============================================================================
CELLS_H="$LLAMA_DIR/src/llama-kv-cells.h"

if grep -q 'get_bias' "$CELLS_H"; then
    echo "  llama-kv-cells.h: already patched, skipping"
else
    echo "  Patching llama-kv-cells.h..."

    # Add bias[i] = 0.0f in reset()
    sed -i '/pos\[i\]   = -1;/a\            bias[i]  = 0.0f;' "$CELLS_H"

    # Add bias.resize in resize()
    sed -i '/ext\.resize(n);/a\        bias.resize(n, 0.0f);' "$CELLS_H"

    # Add bias copy in cp(uint32_t i, uint32_t n)  — after res.ext[j] = ext[idx]
    # and in cp(const std::vector<uint32_t> & idxs) — same pattern
    sed -i 's/res\.ext\[j\] = ext\[idx\];/res.ext[j] = ext[idx];\n            res.bias[j] = bias[idx];/g' "$CELLS_H"

    # Add bias copy in set() — after ext[idx] = other.ext[j]
    sed -i 's/ext\[idx\] = other\.ext\[j\];/ext[idx] = other.ext[j];\n            bias[idx] = other.bias[j];/g' "$CELLS_H"

    # Add get_bias/set_bias accessors after ext_get
    sed -i '/const llama_kv_cell_ext & ext_get/i\
    float get_bias(uint32_t i) const {\
        return bias[i];\
    }\
\
    void set_bias(uint32_t i, float b) {\
        bias[i] = b;\
    }\
' "$CELLS_H"

    # Add bias vector declaration after pos vector
    sed -i '/std::vector<llama_pos> pos;/a\
\
    // per-cell additive attention bias (added to QK^T logits via the mask)\
    std::vector<float> bias;' "$CELLS_H"

    echo "  llama-kv-cells.h: done"
fi

# ============================================================================
# 2. llama-kv-cache.cpp — inject bias into mask + implement set_attn_bias
# ============================================================================
CACHE_CPP="$LLAMA_DIR/src/llama-kv-cache.cpp"

if grep -q 'get_bias' "$CACHE_CPP"; then
    echo "  llama-kv-cache.cpp: already patched, skipping"
else
    echo "  Patching llama-kv-cache.cpp..."

    # Replace "data[idst + j] = 0.0f;" with bias injection
    sed -i 's|data\[idst + j\] = 0\.0f;|data[idst + j] = cells.get_bias(j); // attention bias from KV compaction|' "$CACHE_CPP"

    # Add set_attn_bias implementation after total_size()
    sed -i '/size_t llama_kv_cache::total_size/i\
void llama_kv_cache::set_attn_bias(llama_seq_id seq_id, const float * bias_data, int32_t n) {\
    if (n <= 0 || !bias_data) {\
        return;\
    }\
    const uint32_t stream = seq_to_stream.at(seq_id);\
    auto \& cells = v_cells.at(stream);\
    const int32_t n_set = std::min(n, (int32_t)cells.size());\
    for (int32_t i = 0; i < n_set; i++) {\
        cells.set_bias(i, bias_data[i]);\
    }\
}\
' "$CACHE_CPP"

    echo "  llama-kv-cache.cpp: done"
fi

# ============================================================================
# 3. llama-kv-cache.h — declare set_attn_bias
# ============================================================================
CACHE_H="$LLAMA_DIR/src/llama-kv-cache.h"

if grep -q 'set_attn_bias' "$CACHE_H"; then
    echo "  llama-kv-cache.h: already patched, skipping"
else
    echo "  Patching llama-kv-cache.h..."

    # Add declaration in llama_kv_cache class (the first set_input_kq_mask, not the context one)
    # Use a more specific pattern: match the line with "void set_input_kq_mask" inside the public
    # section of llama_kv_cache (before "protected:")
    sed -i '0,/void set_input_kq_mask.*const;/{/void set_input_kq_mask.*const;/a\
\
    void set_attn_bias(llama_seq_id seq_id, const float * bias_data, int32_t n) override;
    }' "$CACHE_H"

    echo "  llama-kv-cache.h: done"
fi

# ============================================================================
# 4. llama-memory.h — add virtual set_attn_bias to interface
# ============================================================================
MEMORY_H="$LLAMA_DIR/src/llama-memory.h"

if grep -q 'set_attn_bias' "$MEMORY_H"; then
    echo "  llama-memory.h: already patched, skipping"
else
    echo "  Patching llama-memory.h..."

    # Add virtual method before the closing brace of llama_memory_i
    sed -i '/virtual bool get_can_shift/a\
\
    // attention bias support (default no-op for non-KV caches)\
    virtual void set_attn_bias(llama_seq_id /*seq_id*/, const float * /*bias_data*/, int32_t /*n*/) {}' "$MEMORY_H"

    echo "  llama-memory.h: done"
fi

# ============================================================================
# 5. include/llama.h — add public API
# ============================================================================
LLAMA_H="$LLAMA_DIR/include/llama.h"

if grep -q 'llama_memory_set_attn_bias' "$LLAMA_H"; then
    echo "  llama.h: already patched, skipping"
else
    echo "  Patching llama.h..."

    # Add after llama_memory_can_shift declaration
    sed -i '/LLAMA_API bool llama_memory_can_shift/a\
\
    // Set per-cell additive attention bias for KV cache compaction.\
    // The bias is added to QK^T logits before softmax (via the attention mask).\
    // bias_data: array of n floats, one per cell starting at cell 0.\
    // Call after llama_state_seq_set_data() to set biases for the compacted cache.\
    LLAMA_API void llama_memory_set_attn_bias(llama_memory_t mem, llama_seq_id seq_id, const float * bias_data, int32_t n);' "$LLAMA_H"

    echo "  llama.h: done"
fi

# ============================================================================
# 6. llama-context.cpp — implement public API
# ============================================================================
CONTEXT_CPP="$LLAMA_DIR/src/llama-context.cpp"

if grep -q 'llama_memory_set_attn_bias' "$CONTEXT_CPP"; then
    echo "  llama-context.cpp: already patched, skipping"
else
    echo "  Patching llama-context.cpp..."

    # Add implementation after llama_memory_can_shift
    sed -i '/bool llama_memory_can_shift/i\
void llama_memory_set_attn_bias(\
        llama_memory_t mem,\
          llama_seq_id seq_id,\
         const float * bias_data,\
             int32_t   n) {\
    if (!mem) {\
        return;\
    }\
    mem->set_attn_bias(seq_id, bias_data, n);\
}\
' "$CONTEXT_CPP"

    echo "  llama-context.cpp: done"
fi

echo "Attention bias patch applied successfully!"

# ============================================================================
# ============================================================================
#
# PART 2: Cache-Aware MoE Expert Routing
#
# Adds dynamic routing bias to build_moe_ffn() that nudges the router toward
# recently-used experts, reducing effective memory bandwidth without retraining.
# Paper: "Cache-Aware Routing" (2412.00099)
#
# ============================================================================
# ============================================================================

echo ""
echo "Applying MoE cache-aware routing patch to: $LLAMA_DIR"

# ============================================================================
# 7. include/llama.h — add moe_cache_aware bool to llama_context_params
# ============================================================================

if grep -q 'moe_cache_aware' "$LLAMA_H"; then
    echo "  llama.h (MoE): already patched, skipping"
else
    echo "  Patching llama.h (MoE cache-aware routing)..."

    # Add moe_cache_aware bool after kv_unified
    sed -i '/bool kv_unified;/a\
        bool moe_cache_aware; \/\/ enable cache-aware MoE expert routing (EMA-based bias)' "$LLAMA_H"

    echo "  llama.h (MoE): done"
fi

# ============================================================================
# 8. llama-context.cpp — add moe_cache_aware = false to default params
# ============================================================================

if grep -q 'moe_cache_aware' "$CONTEXT_CPP"; then
    echo "  llama-context.cpp (MoE defaults): already patched, skipping"
else
    echo "  Patching llama-context.cpp (MoE default params)..."

    # Add moe_cache_aware default after kv_unified default
    sed -i '/\/\*\.kv_unified.*=\*\//a\
        /*.moe_cache_aware              =*/ false,' "$CONTEXT_CPP"

    echo "  llama-context.cpp (MoE defaults): done"
fi

# ============================================================================
# 9. llama-context.h — add moe_expert_cache member struct and instance
# ============================================================================
CONTEXT_H="$LLAMA_DIR/src/llama-context.h"

if grep -q 'moe_expert_cache' "$CONTEXT_H"; then
    echo "  llama-context.h (MoE): already patched, skipping"
else
    echo "  Patching llama-context.h (MoE expert cache)..."

    # Add the moe_expert_cache struct definition before struct llama_context
    sed -i '/^struct llama_context {/i\
// Cache-Aware MoE Expert Routing (2412.00099)\
// Tracks per-layer expert usage via EMA and provides routing bias.\
struct moe_expert_cache {\
    bool  enabled        = false;\
    int   n_layers       = 0;\
    int   n_experts      = 0;\
    int   cache_size     = 32;\
    float alpha          = 0.1f;\
    float bias_strength  = 0.3f;\
\
    std::vector<std::vector<float>> ema;\
\
    void init(int nl, int ne) {\
        n_layers  = nl;\
        n_experts = ne;\
        ema.resize(nl);\
        for (int l = 0; l < nl; l++) {\
            ema[l].assign(ne, 0.0f);\
        }\
        enabled = true;\
    }\
\
    void update(int layer, const int32_t * expert_ids, int n_used) {\
        auto \& e = ema[layer];\
        const float decay = 1.0f - alpha;\
        for (int i = 0; i < n_experts; i++) {\
            e[i] *= decay;\
        }\
        for (int i = 0; i < n_used; i++) {\
            const int eid = expert_ids[i];\
            if (eid >= 0 \&\& eid < n_experts) {\
                e[eid] += alpha;\
            }\
        }\
    }\
\
    void compute_bias(int layer, float * out_bias) const {\
        const auto \& e = ema[layer];\
        std::vector<float> sorted_ema(e.begin(), e.end());\
        std::sort(sorted_ema.begin(), sorted_ema.end(), std::greater<float>());\
        const int effective_cache = std::min(cache_size, n_experts);\
        const float threshold = (effective_cache > 0 \&\& effective_cache <= n_experts)\
            ? sorted_ema[effective_cache - 1] : 0.0f;\
        for (int i = 0; i < n_experts; i++) {\
            out_bias[i] = (e[i] >= threshold \&\& threshold > 0.0f) ? bias_strength : 0.0f;\
        }\
    }\
};\
' "$CONTEXT_H"

    # Add moe_cache member to llama_context struct (after the cross member)
    sed -i '/llama_cross cross;/a\
\
    moe_expert_cache moe_cache;' "$CONTEXT_H"

    # Add <algorithm> include for std::sort used by moe_expert_cache
    sed -i '/#include <vector>/a\
#include <algorithm>' "$CONTEXT_H"

    echo "  llama-context.h (MoE): done"
fi

# ============================================================================
# 10. llama-context.cpp — init moe_cache in constructor + extract topk post-compute
# ============================================================================

if grep -q 'moe_cache\.init' "$CONTEXT_CPP"; then
    echo "  llama-context.cpp (MoE init): already patched, skipping"
else
    echo "  Patching llama-context.cpp (MoE cache init + post-compute)..."

    # Initialize moe_cache in the constructor, right after kv_unified log line.
    sed -i '/kv_unified.*true.*false/a\
\
    // Initialize MoE expert cache if enabled\
    if (params.moe_cache_aware \&\& hparams.n_expert > 0) {\
        moe_cache.init(hparams.n_layer, hparams.n_expert);\
        LLAMA_LOG_INFO("%s: MoE cache-aware routing enabled (layers=%d, experts=%d, cache=%d)\\n",\
                       __func__, moe_cache.n_layers, moe_cache.n_experts, moe_cache.cache_size);\
    }' "$CONTEXT_CPP"

    # Add post-compute EMA update in process_ubatch, after graph_compute returns successfully.
    # We insert after "ret = GGML_STATUS_SUCCESS;" in process_ubatch.
    sed -i '/ret = GGML_STATUS_SUCCESS;/{
        /return res;/!{
            a\
\
    // Update MoE expert cache after graph compute\
    if (moe_cache.enabled) {\
        auto * gf_post = res->get_gf();\
        for (int il = 0; il < moe_cache.n_layers; il++) {\
            char name[64];\
            snprintf(name, sizeof(name), "ffn_moe_topk-%d", il);\
            ggml_tensor * topk = ggml_graph_get_tensor(gf_post, name);\
            if (!topk) continue;\
\
            std::vector<int32_t> ids(ggml_nelements(topk));\
            ggml_backend_tensor_get(topk, ids.data(), 0, ggml_nbytes(topk));\
\
            int n_tok  = topk->ne[1];\
            int n_used = topk->ne[0];\
            moe_cache.update(il, ids.data() + (n_tok - 1) * n_used, n_used);\
        }\
    }
        }
    }' "$CONTEXT_CPP"

    echo "  llama-context.cpp (MoE init + post-compute): done"
fi

# ============================================================================
# 11. llama-graph.h — add llm_graph_input_moe_bias class
# ============================================================================
GRAPH_H="$LLAMA_DIR/src/llama-graph.h"

if grep -q 'llm_graph_input_moe_bias' "$GRAPH_H"; then
    echo "  llama-graph.h (MoE): already patched, skipping"
else
    echo "  Patching llama-graph.h (MoE bias input class)..."

    # Add the new input class before llm_graph_result section
    sed -i '/^\/\/\s*llm_graph_result/i\
class llm_graph_input_moe_bias : public llm_graph_input_i {\
public:\
    llm_graph_input_moe_bias(int n_layers, int n_experts, moe_expert_cache * cache)\
        : n_layers(n_layers), n_experts(n_experts), cache(cache) {\
        bias.resize(n_layers, nullptr);\
    }\
    virtual ~llm_graph_input_moe_bias() = default;\
\
    void set_input(const llama_ubatch * ubatch) override {\
        GGML_UNUSED(ubatch);\
        if (!cache || !cache->enabled) return;\
        std::vector<float> bias_data(n_experts);\
        for (int l = 0; l < n_layers; l++) {\
            if (!bias[l]) continue;\
            cache->compute_bias(l, bias_data.data());\
            ggml_backend_tensor_set(bias[l], bias_data.data(), 0, n_experts * sizeof(float));\
        }\
    }\
\
    std::vector<ggml_tensor *> bias;\
    int n_layers;\
    int n_experts;\
    moe_expert_cache * cache;\
};\
' "$GRAPH_H"

    # Forward-declare moe_expert_cache at top of file (after existing forward declarations)
    sed -i '/^struct llama_memory_i;/a\
struct moe_expert_cache;' "$GRAPH_H"

    echo "  llama-graph.h (MoE): done"
fi

# ============================================================================
# 12. llama-graph.cpp — add cache-aware routing bias before argsort_top_k
# ============================================================================
GRAPH_CPP="$LLAMA_DIR/src/llama-graph.cpp"

if grep -q 'ffn_moe_probs_cache_biased' "$GRAPH_CPP"; then
    echo "  llama-graph.cpp (MoE): already patched, skipping"
else
    echo "  Patching llama-graph.cpp (MoE routing bias)..."

    # Insert cache-aware bias addition right after the DeepSeek V3 exp_probs_b block,
    # before the expert group selection. We target the line after the exp_probs_b block.
    # The existing code has:
    #   if (exp_probs_b != nullptr) {
    #       selection_probs = ggml_add(ctx0, probs, exp_probs_b);
    #       cb(selection_probs, "ffn_moe_probs_biased", il);
    #   }
    #
    #   // llama4 doesn't have exp_probs_b ...
    #
    # We add our cache-aware bias check right before the "// llama4" comment.
    sed -i '/\/\/ llama4 doesn.t have exp_probs_b/i\
    // Cache-aware routing bias (2412.00099): nudge selection toward recently-used experts\
    if (moe_cache_bias) {\
        selection_probs = ggml_add(ctx0, selection_probs, moe_cache_bias);\
        cb(selection_probs, "ffn_moe_probs_cache_biased", il);\
    }\
' "$GRAPH_CPP"

    echo "  llama-graph.cpp (MoE routing bias): done"
fi

# ============================================================================
# 13. llama-graph.h — add moe_cache_bias param to build_moe_ffn signatures
# ============================================================================

if grep -q 'moe_cache_bias' "$GRAPH_H"; then
    echo "  llama-graph.h (build_moe_ffn sig): already patched, skipping"
else
    echo "  Patching llama-graph.h (build_moe_ffn moe_cache_bias param)..."

    # Add moe_cache_bias param to both overloads of build_moe_ffn.
    # Overload 1 (without bias tensors): add after gate_up_exps = nullptr
    sed -i '/ggml_tensor \* gate_up_exps = nullptr) const;/{
        s/) const;/,\
             ggml_tensor * moe_cache_bias = nullptr) const;/
    }' "$GRAPH_H"

    # Overload 2 (with bias tensors): add after gate_up_exps_b = nullptr
    sed -i '/ggml_tensor \* gate_up_exps_b = nullptr) const;/{
        s/) const;/,\
             ggml_tensor * moe_cache_bias = nullptr) const;/
    }' "$GRAPH_H"

    echo "  llama-graph.h (build_moe_ffn sig): done"
fi

# ============================================================================
# 14. llama-graph.cpp — add moe_cache_bias param to build_moe_ffn implementations
# ============================================================================

if grep -q 'ggml_tensor \* moe_cache_bias) const {' "$GRAPH_CPP"; then
    echo "  llama-graph.cpp (build_moe_ffn impl): already patched, skipping"
else
    echo "  Patching llama-graph.cpp (build_moe_ffn moe_cache_bias param)..."

    # Overload 1 (delegating): add param to signature and pass through
    # Before: ggml_tensor * gate_up_exps) const {
    #     return build_moe_ffn(
    sed -i '/ggml_tensor \* gate_up_exps) const {/{
        s/) const {/,\
         ggml_tensor * moe_cache_bias) const {/
    }' "$GRAPH_CPP"

    # Pass moe_cache_bias through in the delegating overload.
    # The passthrough call ends with "gate_up_exps\n    );" — replace with passthrough.
    sed -i '/return build_moe_ffn/,/);/{
        /gate_up_exps$/{
            s|gate_up_exps$|gate_up_exps, /* gate_up_exps_b */ nullptr,|
            n
            s|^    );|        moe_cache_bias);|
        }
    }' "$GRAPH_CPP"

    # Overload 2 (main impl): add param to signature
    # Before: ggml_tensor * gate_up_exps_b) const {
    sed -i '/ggml_tensor \* gate_up_exps_b) const {/{
        s/) const {/,\
         ggml_tensor * moe_cache_bias) const {/
    }' "$GRAPH_CPP"

    echo "  llama-graph.cpp (build_moe_ffn impl): done"
fi

echo "MoE cache-aware routing patch applied successfully!"

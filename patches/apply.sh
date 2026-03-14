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

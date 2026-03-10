#!/bin/bash
# Apply KV compaction patches to llama.cpp source
# Run from kv-compact build directory after cmake configure
#
# Usage:
#   cd build
#   cmake ..
#   bash ../patches/apply.sh
#   cmake --build . --target llama-server
#
# This patches:
#   1. server-context.cpp: KV compaction hooks in process_token() and update_slots()
#   2. server CMakeLists.txt: link kv-compact-lib
#   3. common.h: add kv_compact_ratio parameter
#   4. arg.cpp: add --kv-compact-ratio CLI argument

LLAMA_SRC="${1:-_deps/llama_cpp-src}"
PATCH_DIR="$(dirname "$0")"

if [ ! -d "$LLAMA_SRC/tools/server" ]; then
    echo "Error: llama.cpp source not found at $LLAMA_SRC"
    echo "Usage: $0 [llama_cpp_src_dir]"
    exit 1
fi

SERVER_CTX="$LLAMA_SRC/tools/server/server-context.cpp"
SERVER_CMAKE="$LLAMA_SRC/tools/server/CMakeLists.txt"
COMMON_H="$LLAMA_SRC/common/common.h"
ARG_CPP="$LLAMA_SRC/common/arg.cpp"

# --- 1. Patch server-context.cpp: add kv-compact-api.h include ---
if ! grep -q "kv-compact-api.h" "$SERVER_CTX"; then
    sed -i '/#include "server-context.h"/a #include "kv-compact-api.h"' "$SERVER_CTX"
    echo "Patched: added kv-compact-api.h include"
fi

# --- 2. Patch server CMakeLists: link kv-compact-lib ---
if ! grep -q "kv-compact-lib" "$SERVER_CMAKE"; then
    sed -i '/target_link_libraries.*PUBLIC common mtmd/a \
\n# KV compaction integration\nif(TARGET kv-compact-lib)\n    target_link_libraries(${TARGET} PUBLIC kv-compact-lib)\nendif()' "$SERVER_CMAKE"
    echo "Patched: linked kv-compact-lib to server"
fi

# --- 3. Patch common.h: add kv_compact_ratio parameter ---
if ! grep -q "kv_compact_ratio" "$COMMON_H"; then
    sed -i '/bool kv_unified.*=.*false;/a \
\n    float kv_compact_ratio = 0.0f;  \/\/ KV compaction ratio (0.0 = disabled, 0.5 = keep 50%, etc.)' "$COMMON_H"
    echo "Patched: added kv_compact_ratio to common_params"
fi

# --- 4. Patch arg.cpp: add --kv-compact-ratio CLI argument ---
if ! grep -q "kv-compact-ratio" "$ARG_CPP"; then
    python3 - "$ARG_CPP" << 'PYEOF'
import sys

arg_cpp = sys.argv[1] if len(sys.argv) > 1 else "_deps/llama_cpp-src/common/arg.cpp"

with open(arg_cpp, 'r') as f:
    content = f.read()

# Find the --context-shift registration and add --kv-compact-ratio after it
marker = '.set_env("LLAMA_ARG_CONTEXT_SHIFT"));'
if marker not in content:
    print("Warning: could not find LLAMA_ARG_CONTEXT_SHIFT marker in arg.cpp")
    sys.exit(0)

insert = '''
    add_opt(common_arg(
        {"--kv-compact-ratio"}, "N",
        string_format("KV cache compaction ratio when context is full (default: %.1f, 0.0 = disabled, 0.5 = keep 50%%)", (double)params.kv_compact_ratio),
        [](common_params & params, const std::string & value) {
            params.kv_compact_ratio = std::stof(value);
            if (params.kv_compact_ratio < 0.0f || params.kv_compact_ratio >= 1.0f) {
                throw std::invalid_argument("--kv-compact-ratio must be >= 0.0 and < 1.0");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_KV_COMPACT_RATIO"));'''

content = content.replace(marker, marker + insert)
with open(arg_cpp, 'w') as f:
    f.write(content)
print("Patched: added --kv-compact-ratio CLI argument")
PYEOF
fi

# --- 5. Patch server-context.cpp: add compaction hook in process_token() ---
#     (no-context-shift path: compact instead of stopping)
if ! grep -q "kv_compact_sequence" "$SERVER_CTX"; then
    python3 - "$SERVER_CTX" << 'PYEOF'
import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

# Replace the no-context-shift early stop with compaction
old = '''        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }'''

new = '''        // if context shifting is disabled: try KV compaction if enabled, otherwise stop
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx && params_base.kv_compact_ratio <= 0.0f) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, n_ctx = %d\\n",
                    slot.prompt.n_tokens(), slot.n_ctx);
        }

        if (!params_base.ctx_shift && params_base.kv_compact_ratio > 0.0f && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            SLT_WRN(slot, "context full (%d/%d tokens), attempting KV compaction (ratio=%.2f)\\n",
                    slot.prompt.n_tokens(), slot.n_ctx, params_base.kv_compact_ratio);

            kv_compact_params compact_params = kv_compact_params_default();
            compact_params.compact_ratio = params_base.kv_compact_ratio;
            compact_params.n_keep        = std::min(4, slot.n_ctx / 8);

            // Allocate buffer for kept positions so we can accurately rebuild tokens
            int max_kept = (int)(slot.n_ctx * params_base.kv_compact_ratio) + 16;
            std::vector<int32_t> kept_pos(max_kept);
            compact_params.kept_positions     = kept_pos.data();
            compact_params.kept_positions_cap = max_kept;

            int new_size = kv_compact_sequence(ctx, slot.id, compact_params);

            if (new_size > 0 && new_size < slot.prompt.n_tokens()) {
                SLT_WRN(slot, "KV compaction succeeded: %d -> %d tokens\\n",
                        slot.prompt.n_tokens(), new_size);

                // Rebuild token bookkeeping using the actual kept positions.
                // kept_pos[i] contains the original position that maps to new position i.
                GGML_ASSERT(!slot.prompt.tokens.has_mtmd);
                llama_tokens old_tokens = slot.prompt.tokens.get_text_tokens();
                llama_tokens new_tokens;
                new_tokens.reserve(new_size);
                for (int i = 0; i < new_size; i++) {
                    int orig_pos = kept_pos[i];
                    if (orig_pos >= 0 && orig_pos < (int)old_tokens.size()) {
                        new_tokens.push_back(old_tokens[orig_pos]);
                    } else {
                        new_tokens.push_back(old_tokens.back());
                    }
                }
                slot.prompt.tokens.clear();
                slot.prompt.tokens.insert(new_tokens);
                slot.truncated = true;
            } else {
                slot.truncated      = true;
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;
                SLT_DBG(slot, "KV compaction failed, stopped, prompt.n_tokens() = %d, n_ctx = %d\\n",
                        slot.prompt.n_tokens(), slot.n_ctx);
            }
        }'''

if old in content:
    content = content.replace(old, new)
    with open(sys.argv[1], 'w') as f:
        f.write(content)
    print("Patched: replaced context-full handler with KV compaction (process_token)")
else:
    print("Warning: could not find context-full handler to patch (may already be patched)")
PYEOF
fi

# --- 6. Patch server-context.cpp: add compaction hook in update_slots() ---
#     (context-shift path: try compaction first, fall back to context shift)
if ! grep -q "KV cache compaction.*attention matching" "$SERVER_CTX"; then
    python3 - "$SERVER_CTX" << 'PYEOF'
import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

# Find the context-shift comment block in update_slots and wrap it with compaction
old = '''        // apply context-shift if needed'''

# We need a bigger anchor. Find the context-shift for loop.
# The original llama.cpp code looks like:
#   for (server_slot & slot : slots) {
#       if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
#           ...original context shift...
#       }
#   }
# We replace the body of the context-shift block to try compaction first.

# Find the n_discard calculation and the seq_rm/seq_add block
old_shift = '''                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                llama_memory_seq_rm (llama_get_memory(ctx), slot.id, n_keep            , n_keep + n_discard);
                llama_memory_seq_add(llama_get_memory(ctx), slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_text_tokens();
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }'''

new_shift = '''                // Try KV cache compaction if enabled, otherwise fall back to context shift
                int new_size = 0;

                // Allocate buffer for kept positions so we can accurately rebuild token list
                std::vector<int32_t> kept_pos;

                if (params_base.kv_compact_ratio > 0.0f) {
                    kv_compact_params compact_params = kv_compact_params_default();
                    compact_params.compact_ratio = params_base.kv_compact_ratio;
                    compact_params.n_keep        = n_keep; // preserve sink tokens

                    int max_kept = (int)(slot.n_ctx * params_base.kv_compact_ratio) + 16;
                    kept_pos.resize(max_kept);
                    compact_params.kept_positions     = kept_pos.data();
                    compact_params.kept_positions_cap = max_kept;

                    SLT_WRN(slot, "slot KV compaction, n_tokens = %d, n_keep = %d, ratio = %.2f\\n",
                            slot.prompt.n_tokens(), n_keep, compact_params.compact_ratio);

                    new_size = kv_compact_sequence(ctx, slot.id, compact_params);
                }

                if (new_size > 0) {
                    const int n_discard = slot.prompt.n_tokens() - new_size;

                    SLT_WRN(slot, "KV compaction done: %d -> %d tokens (discarded %d)\\n",
                            slot.prompt.n_tokens(), new_size, n_discard);

                    // Rebuild token bookkeeping using the actual kept positions.
                    // kept_pos[i] contains the original position that maps to new position i.
                    {
                        GGML_ASSERT(!slot.prompt.tokens.has_mtmd);
                        llama_tokens old_tokens = slot.prompt.tokens.get_text_tokens();
                        llama_tokens new_tokens;
                        new_tokens.reserve(new_size);
                        for (int i = 0; i < new_size; i++) {
                            int orig_pos = kept_pos[i];
                            if (orig_pos >= 0 && orig_pos < (int)old_tokens.size()) {
                                new_tokens.push_back(old_tokens[orig_pos]);
                            } else {
                                new_tokens.push_back(old_tokens.back());
                            }
                        }

                        slot.prompt.tokens.clear();
                        slot.prompt.tokens.insert(new_tokens);
                    }
                } else {
                    // Compaction failed or disabled - fall back to original context shift
                    if (params_base.kv_compact_ratio > 0.0f) {
                        SLT_WRN(slot, "%s", "KV compaction failed, falling back to context shift\\n");
                    }

                    const int n_left    = slot.prompt.n_tokens() - n_keep;
                    const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                    llama_memory_seq_rm (llama_get_memory(ctx), slot.id, n_keep            , n_keep + n_discard);
                    llama_memory_seq_add(llama_get_memory(ctx), slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                    {
                        GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                        llama_tokens new_tokens = slot.prompt.tokens.get_text_tokens();
                        for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                            new_tokens[i - n_discard] = new_tokens[i];
                        }

                        new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                        slot.prompt.tokens.clear();
                        slot.prompt.tokens.insert(new_tokens);
                    }
                }'''

if old_shift in content:
    content = content.replace(old_shift, new_shift)
    with open(sys.argv[1], 'w') as f:
        f.write(content)
    print("Patched: added KV compaction to update_slots context-shift path")
else:
    print("Warning: could not find context-shift block in update_slots (may already be patched)")
PYEOF
fi

echo ""
echo "Done. Rebuild with: cmake --build . --target llama-server"
echo ""
echo "Usage: llama-server --kv-compact-ratio 0.5 --no-context-shift ..."
echo "  or:  llama-server --kv-compact-ratio 0.5  (with context shift as fallback)"

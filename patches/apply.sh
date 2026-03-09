#!/bin/bash
# Apply KV compaction patch to llama.cpp server source
# Run from kv-compact build directory after cmake configure
#
# Usage:
#   cd build
#   cmake ..
#   bash ../patches/apply.sh
#   cmake --build . --target llama-server

LLAMA_SRC="${1:-_deps/llama_cpp-src}"
PATCH_DIR="$(dirname "$0")"

if [ ! -d "$LLAMA_SRC/tools/server" ]; then
    echo "Error: llama.cpp source not found at $LLAMA_SRC"
    echo "Usage: $0 [llama_cpp_src_dir]"
    exit 1
fi

SERVER_CTX="$LLAMA_SRC/tools/server/server-context.cpp"
SERVER_CMAKE="$LLAMA_SRC/tools/server/CMakeLists.txt"

# Patch server-context.cpp: add kv-compact-api.h include
if ! grep -q "kv-compact-api.h" "$SERVER_CTX"; then
    sed -i '/#include "server-context.h"/a #include "kv-compact-api.h"' "$SERVER_CTX"
    echo "Patched: added kv-compact-api.h include"
fi

# Patch server CMakeLists: link kv-compact-lib
if ! grep -q "kv-compact-lib" "$SERVER_CMAKE"; then
    sed -i '/target_link_libraries.*PUBLIC common mtmd/a \
\n# KV compaction integration\nif(TARGET kv-compact-lib)\n    target_link_libraries(${TARGET} PUBLIC kv-compact-lib)\nendif()' "$SERVER_CMAKE"
    echo "Patched: linked kv-compact-lib to server-context"
fi

# Patch server-context.cpp: replace context-full handler with compaction
if ! grep -q "kv_compact_sequence" "$SERVER_CTX"; then
    # Replace the no-context-shift early stop with compaction
    python3 << 'PYEOF'
import re

with open("'"$SERVER_CTX"'", 'r') as f:
    content = f.read()

old = '''        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }'''

new = '''        // if context shifting is disabled, try KV compaction before giving up
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            SLT_WRN(slot, "context full (%d/%d tokens), attempting KV compaction\\n",
                    slot.prompt.n_tokens(), slot.n_ctx);

            kv_compact_params compact_params = kv_compact_params_default();
            compact_params.compact_ratio = 0.5f;
            compact_params.n_keep        = std::min(4, slot.n_ctx / 8);

            int new_size = kv_compact_sequence(ctx, slot.id, compact_params);

            if (new_size > 0 && new_size < slot.prompt.n_tokens()) {
                SLT_WRN(slot, "KV compaction succeeded: %d -> %d tokens\\n",
                        slot.prompt.n_tokens(), new_size);

                GGML_ASSERT(!slot.prompt.tokens.has_mtmd);
                llama_tokens old_tokens = slot.prompt.tokens.get_text_tokens();
                int n_keep_tokens = compact_params.n_keep;
                llama_tokens new_tokens;
                new_tokens.reserve(new_size);
                for (int i = 0; i < n_keep_tokens && i < (int)old_tokens.size(); i++)
                    new_tokens.push_back(old_tokens[i]);
                int n_recent = new_size - (int)new_tokens.size();
                int start = std::max((int)old_tokens.size() - n_recent, n_keep_tokens);
                for (int i = start; i < (int)old_tokens.size() && (int)new_tokens.size() < new_size; i++)
                    new_tokens.push_back(old_tokens[i]);
                while ((int)new_tokens.size() < new_size)
                    new_tokens.push_back(old_tokens.back());
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
    with open("'"$SERVER_CTX"'", 'w') as f:
        f.write(content)
    print("Patched: replaced context-full handler with KV compaction")
else:
    print("Warning: could not find context-full handler to patch (may already be patched)")

PYEOF
fi

echo "Done. Rebuild with: cmake --build . --target llama-server"

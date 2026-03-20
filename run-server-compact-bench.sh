#!/bin/bash
# Continuous batching: compacted vs full KV cache comparison
# Same total memory budget, compacted fits more slots
#
# Baseline: N slots × 2048 ctx (full 881-token prompts)
# Compacted: 5N slots × 410 ctx (simulating 5x-compacted ~176-token KV)
# Both use the same total ctx allocation

SERVER="/c/Users/fabia/Projects/llama.cpp/llama-flash-attn/build-win/bin/llama-server.exe"
MODEL="/c/Users/fabia/models/SmolLM3-3B-128K-Q4_K_M.gguf"
FULL_PROMPT="/c/Users/fabia/Projects/kv-compact/prompt-profile.txt"
SHORT_PROMPT="/tmp/short_prompt.txt"

HOST="127.0.0.1"
PORT=8091
N_PREDICT=50

# Encode prompts as JSON payloads
WIN_FULL=$(cygpath -w "$FULL_PROMPT")
WIN_SHORT=$(cygpath -w "$SHORT_PROMPT")

python3 -c "
import json
with open(r'${WIN_FULL}') as f:
    p = f.read()
with open(r'C:\Users\fabia\AppData\Local\Temp\full_req.json', 'w') as out:
    json.dump({'prompt': p, 'n_predict': ${N_PREDICT}, 'temperature': 0.0, 'cache_prompt': True}, out)
with open(r'${WIN_SHORT}') as f:
    p = f.read()
with open(r'C:\Users\fabia\AppData\Local\Temp\short_req.json', 'w') as out:
    json.dump({'prompt': p, 'n_predict': ${N_PREDICT}, 'temperature': 0.0, 'cache_prompt': True}, out)
"

FULL_REQ="/tmp/full_req.json"
SHORT_REQ="/tmp/short_req.json"

echo "=== Compacted vs Full KV: Continuous Batching Benchmark ==="
echo "Model: SmolLM3 3B Q4_K_M (Vulkan)"
echo "Generate: ${N_PREDICT} tokens per request"
echo ""

wait_for_server() {
    for i in $(seq 1 120); do
        if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
            return 0
        fi
        sleep 1
    done
    return 1
}

send_one() {
    curl -s -X POST "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d @"$1" > "$2" 2>/dev/null
}

extract_timings() {
    local f="$1"
    local wf=$(cygpath -w "$f" 2>/dev/null || echo "$f")
    python3 -c "
import json
try:
    d = json.load(open(r'${wf}'))
    t = d.get('timings', {})
    print(f\"{t.get('predicted_n',0)} {t.get('predicted_per_second',0):.2f} {t.get('prompt_n',0)} {t.get('prompt_per_second',0):.2f}\")
except:
    print('0 0.00 0 0.00')
" 2>/dev/null
}

run_scenario() {
    local label="$1"
    local n_slots="$2"
    local ctx_per_slot="$3"
    local total_ctx=$((n_slots * ctx_per_slot))
    local req_file="$4"

    echo "--- ${label}: ${n_slots} slots × ${ctx_per_slot} ctx (total: ${total_ctx}) ---"

    taskkill //F //IM llama-server.exe > /dev/null 2>&1 || true
    sleep 3
    while netstat -ano 2>/dev/null | grep -q ":${PORT} "; do sleep 1; done

    "$SERVER" -m "$MODEL" \
        -c "$total_ctx" -np "$n_slots" -cb -ngl 99 \
        --host "$HOST" --port "$PORT" \
        > "/tmp/sbc_log_${label}.txt" 2>&1 &
    local spid=$!

    echo "  Starting server (PID: $spid)..."
    if ! wait_for_server; then
        echo "  SKIP: Server failed to start"
        kill $spid 2>/dev/null || true
        return
    fi
    echo "  Server ready."

    # Warmup
    send_one "$req_file" "/tmp/sbc_warmup.json"

    # Fire all slots concurrently
    echo "  Sending ${n_slots} concurrent requests..."
    local t_start=$(date +%s%N)

    local pids=()
    for i in $(seq 1 $n_slots); do
        send_one "$req_file" "/tmp/sbc_${label}_${i}.json" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    local t_end=$(date +%s%N)
    local wall_ms=$(( (t_end - t_start) / 1000000 ))

    # Collect results
    local total_gen=0
    local min_tps=99999
    local max_tps=0
    for i in $(seq 1 $n_slots); do
        local f="/tmp/sbc_${label}_${i}.json"
        local timing=$(extract_timings "$f")
        local n_tok=$(echo "$timing" | awk '{print $1}')
        local tps=$(echo "$timing" | awk '{print $2}')
        total_gen=$((total_gen + n_tok))

        # Track min/max for range display
        python3 -c "
t = float('${tps}')
mn = float('${min_tps}')
mx = float('${max_tps}')
if t > 0 and t < mn: print(f'MIN {t:.2f}')
elif t > mx: print(f'MAX {t:.2f}')
else: print('SAME')
" 2>/dev/null | while read line; do
            case "$line" in
                MIN*) min_tps=$(echo "$line" | awk '{print $2}') ;;
                MAX*) max_tps=$(echo "$line" | awk '{print $2}') ;;
            esac
        done
    done

    local agg_tps="0"
    if [ "$wall_ms" -gt 0 ] && [ "$total_gen" -gt 0 ]; then
        agg_tps=$(python3 -c "print(f'{$total_gen * 1000.0 / $wall_ms:.2f}')" 2>/dev/null || echo "0")
    fi

    # Show first and last slot details
    echo "  Slot 1: $(extract_timings /tmp/sbc_${label}_1.json | awk '{printf "gen=%s tok (%s tok/s)", $1, $2}')"
    if [ "$n_slots" -gt 1 ]; then
        echo "  Slot ${n_slots}: $(extract_timings /tmp/sbc_${label}_${n_slots}.json | awk '{printf "gen=%s tok (%s tok/s)", $1, $2}')"
    fi
    echo "  Wall time: ${wall_ms} ms"
    echo "  Total generated: ${total_gen} tokens"
    echo "  **Aggregate throughput: ${agg_tps} tok/s**"
    echo ""

    taskkill //F //IM llama-server.exe > /dev/null 2>&1 || true
    sleep 2
}

# ============================================================
# Test matrix: same total memory, different slot configurations
# ============================================================

TOTAL_CTX=16384  # Fixed memory budget

echo "Memory budget: ${TOTAL_CTX} total context tokens"
echo ""

# Baseline scenarios (full 881-token prompts, 2048 ctx per slot)
run_scenario "full_4"  4  2048 "$FULL_REQ"   # 4×2048 = 8192
run_scenario "full_8"  8  2048 "$FULL_REQ"   # 8×2048 = 16384

# Compacted scenarios (simulated: ~200-token prompts, 512 ctx per slot)
run_scenario "compact_8"   8  512 "$SHORT_REQ"   # 8×512 = 4096
run_scenario "compact_16" 16  512 "$SHORT_REQ"   # 16×512 = 8192
run_scenario "compact_32" 32  512 "$SHORT_REQ"   # 32×512 = 16384

# Extreme compaction (10x: ~90-token KV, 256 ctx per slot)
run_scenario "compact10x_32" 32 256 "$SHORT_REQ"  # 32×256 = 8192
run_scenario "compact10x_64" 64 256 "$SHORT_REQ"  # 64×256 = 16384

echo "=== Benchmark Complete ==="

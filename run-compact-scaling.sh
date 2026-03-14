#!/bin/bash
# Compaction scaling benchmark
# Single server instance, vary prompt length to simulate compaction
# Token counts: 1K=881, 10K=10500

SERVER="/c/Users/fabia/Projects/llama.cpp/llama-flash-attn/build-win/bin/llama-server.exe"
MODEL="/c/Users/fabia/models/SmolLM3-3B-128K-Q4_K_M.gguf"
HOST="127.0.0.1"
PORT=9050
N_PREDICT=50
N_SLOTS=8

# Generate prompts
python3 "$(cygpath -w /c/Users/fabia/Projects/kv-compact/gen-prompts.py)" 2>&1

echo ""
echo "=== KV Compaction Scaling Benchmark ==="
echo "Model: SmolLM3 3B Q4_K_M (Vulkan)"
echo "Slots: ${N_SLOTS}, gen: ${N_PREDICT} tokens"
echo ""

# Start ONE server with large ctx for all tests
TOTAL_CTX=$((N_SLOTS * 12288))  # 8 * 12K = 96K total
echo "Starting server (ctx=${TOTAL_CTX}, slots=${N_SLOTS})..."

"$SERVER" -m "$MODEL" -c "$TOTAL_CTX" -np "$N_SLOTS" -cb -ngl 99 \
    --host "$HOST" --port "$PORT" > /tmp/cs_server.log 2>&1 &
SPID=$!

# Wait for ready
for i in $(seq 1 120); do
    if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
        echo "Server ready (PID: $SPID)"
        break
    fi
    sleep 1
done
if ! curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
    echo "Server failed to start"; kill $SPID 2>/dev/null; exit 1
fi

bench() {
    local label="$1" req="$2"

    # Clear any pending slots
    sleep 1

    # Fire N_SLOTS concurrent requests (use -o for output, not >)
    local t0=$(date +%s%N)
    for i in $(seq 1 $N_SLOTS); do
        curl -s --max-time 300 -X POST "http://${HOST}:${PORT}/completion" \
            -H "Content-Type: application/json" -d @"$req" \
            -o "/tmp/cs_${label}_${i}.json" &
    done
    wait
    local t1=$(date +%s%N)
    local ms=$(( (t1 - t0) / 1000000 ))

    # Collect
    local total=0 errs=0 sum_tps=0
    for i in $(seq 1 $N_SLOTS); do
        local wf=$(cygpath -w "/tmp/cs_${label}_${i}.json")
        local vals=$(python3 -c "
import json
try:
    d = json.load(open(r'${wf}'))
    t = d.get('timings',{})
    print(t.get('predicted_n',0), f\"{t.get('predicted_per_second',0):.1f}\")
except: print('0 0.0')
" 2>/dev/null)
        local n=$(echo "$vals" | awk '{print $1}')
        local tps=$(echo "$vals" | awk '{print $2}')
        total=$((total + n))
        [ "$n" -eq 0 ] && errs=$((errs + 1))
    done

    if [ "$total" -gt 0 ] && [ "$ms" -gt 0 ]; then
        local agg=$(python3 -c "print(f'{$total*1000.0/$ms:.1f}')")
        local mark=""
        [ "$errs" -gt 0 ] && mark=" (${errs}err)"
        echo "${agg}${mark}"
    else
        echo "FAIL"
    fi
}

for ps in 1k 10k; do
    if [ "$ps" = "1k" ]; then ft=881; else ft=10500; fi
    echo ""
    echo "--- Prompt: ${ps} (${ft} tokens) ---"
    printf "%-6s | %-6s | %s\n" "Ratio" "KV tok" "Agg tok/s (${N_SLOTS} slots)"
    printf "%-6s-+-%-6s-+-%s\n" "------" "------" "----------------------------"

    for entry in "1x:100" "1.25x:80" "1.67x:60" "2.5x:40" "5x:20" "10x:10" "20x:5" "50x:2"; do
        rl="${entry%%:*}"
        kp="${entry##*:}"
        kv=$(python3 -c "print(int(${ft}*${kp}/100))")
        req="/tmp/req_${ps}_keep${kp}.json"

        if [ ! -f "$req" ]; then
            printf "%-6s | %-6s | NO_FILE\n" "$rl" "$kv"
            continue
        fi

        # Warmup this prompt length
        curl -s --max-time 180 -X POST "http://${HOST}:${PORT}/completion" \
            -H "Content-Type: application/json" -d @"$req" > /dev/null 2>&1

        printf "%-6s | %-6s | " "$rl" "$kv"
        result=$(bench "${ps}_${rl}" "$req")
        echo "${result} tok/s"
    done
done

echo ""
echo "=== Done ==="
kill $SPID 2>/dev/null

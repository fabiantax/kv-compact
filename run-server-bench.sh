#!/bin/bash
# Continuous batching benchmark via llama-server
# Tests aggregate throughput with N parallel slots
# Usage: ./run-server-bench.sh [model: smol|gemma] [max_slots: 8]

SERVER="/c/Users/fabia/Projects/llama.cpp/llama-flash-attn/build-win/bin/llama-server.exe"
PROMPT_FILE="/c/Users/fabia/Projects/kv-compact/prompt-profile.txt"

MODEL_ARG="${1:-smol}"
MAX_SLOTS="${2:-8}"

if [ "$MODEL_ARG" = "gemma" ]; then
    MODEL="/c/Users/fabia/models/gemma-3-4b-it-heretic-v1.2-Q4_K_M.gguf"
    MODEL_NAME="Gemma 3 4B"
else
    MODEL="/c/Users/fabia/models/SmolLM3-3B-128K-Q4_K_M.gguf"
    MODEL_NAME="SmolLM3 3B"
fi

HOST="127.0.0.1"
PORT=8091
N_PREDICT=50
CTX=2048

# Pre-encode prompt as JSON string (convert MSYS path to Windows for python)
WIN_PROMPT_FILE=$(cygpath -w "$PROMPT_FILE" 2>/dev/null || echo "$PROMPT_FILE")
PROMPT_JSON=$(python3 -c "
import json
with open(r'${WIN_PROMPT_FILE}') as f:
    print(json.dumps(f.read()))
" 2>/dev/null)

if [ -z "$PROMPT_JSON" ]; then
    echo "ERROR: Failed to encode prompt"
    exit 1
fi

# Build request payload template (will be written to file for curl)
REQUEST_FILE="/tmp/server_bench_request.json"
cat > "$REQUEST_FILE" << JSONEOF
{
    "prompt": ${PROMPT_JSON},
    "n_predict": ${N_PREDICT},
    "temperature": 0.0,
    "cache_prompt": true
}
JSONEOF

echo "=== Continuous Batching Throughput Benchmark ==="
echo "Model: ${MODEL_NAME}"
echo "Prompt: ~881 tokens, generate ${N_PREDICT} tokens"
echo "Server: llama-server with continuous batching + Vulkan"
echo ""

# Build slot counts array up to MAX_SLOTS
SLOT_COUNTS=()
s=1
while [ $s -le "$MAX_SLOTS" ]; do
    SLOT_COUNTS+=($s)
    s=$((s * 2))
done

wait_for_server() {
    local max_wait=120
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

send_one() {
    local outfile="$1"
    curl -s -X POST "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d @"$REQUEST_FILE" > "$outfile" 2>/dev/null
}

# Track results for early-stop comparison
declare -a AGG_TPS_RESULTS

for n_slots in "${SLOT_COUNTS[@]}"; do
    echo "--- ${n_slots} parallel slot(s) ---"

    # Kill any existing server and wait for port to free
    taskkill //F //IM llama-server.exe > /dev/null 2>&1 || true
    sleep 3
    # Verify port is free
    while netstat -ano 2>/dev/null | grep -q ":${PORT} "; do
        sleep 1
    done

    # Start server
    "$SERVER" \
        -m "$MODEL" \
        -c $((CTX * n_slots)) \
        -np "$n_slots" \
        -cb \
        -ngl 99 \
        --host "$HOST" \
        --port "$PORT" \
        > "/tmp/server_bench_log_${n_slots}.txt" 2>&1 &
    SERVER_PID=$!

    echo "  Starting server (PID: $SERVER_PID, ctx: $((CTX * n_slots)), slots: $n_slots)..."
    if ! wait_for_server; then
        echo "  SKIP: Server failed to start within 120s"
        kill $SERVER_PID 2>/dev/null || true
        continue
    fi
    echo "  Server ready."

    # Warmup: single request (fills prompt cache)
    echo "  Warmup..."
    send_one "/tmp/server_bench_warmup.json"

    # Benchmark: fire N concurrent requests
    echo "  Sending ${n_slots} concurrent requests..."
    t_start=$(date +%s%N)

    pids=()
    for i in $(seq 1 $n_slots); do
        send_one "/tmp/server_bench_r${n_slots}_${i}.json" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    t_end=$(date +%s%N)
    wall_ms=$(( (t_end - t_start) / 1000000 ))

    # Extract per-slot timings from /completion response (has timings object)
    total_tokens=0
    sum_tps=0
    for i in $(seq 1 $n_slots); do
        f="/tmp/server_bench_r${n_slots}_${i}.json"
        if [ -f "$f" ]; then
            wf=$(cygpath -w "$f" 2>/dev/null || echo "$f")
            result=$(python3 -c "
import json
try:
    d = json.load(open(r'${wf}'))
    t = d.get('timings', {})
    tps = t.get('predicted_per_second', 0)
    n = t.get('predicted_n', 0)
    pp_tps = t.get('prompt_per_second', 0)
    pp_n = t.get('prompt_n', 0)
    print(f'{n} {tps:.2f} {pp_n} {pp_tps:.2f}')
except Exception as e:
    print(f'0 0 0 0')
" 2>/dev/null)
            n_tok=$(echo "$result" | awk '{print $1}')
            tps=$(echo "$result" | awk '{print $2}')
            pp_n=$(echo "$result" | awk '{print $3}')
            pp_tps=$(echo "$result" | awk '{print $4}')
            total_tokens=$((total_tokens + n_tok))
            echo "  Slot $i: prompt=${pp_n} tok (${pp_tps} tok/s), gen=${n_tok} tok (${tps} tok/s)"
        else
            echo "  Slot $i: no output"
        fi
    done

    # Aggregate throughput = total tokens generated / wall time
    agg_tps="0"
    if [ "$wall_ms" -gt 0 ] && [ "$total_tokens" -gt 0 ]; then
        agg_tps=$(python3 -c "print(f'{$total_tokens * 1000.0 / $wall_ms:.2f}')" 2>/dev/null || echo "0")
    fi

    echo "  Wall time: ${wall_ms} ms"
    echo "  Total tokens: ${total_tokens}"
    echo "  Aggregate tok/s: ${agg_tps}"
    AGG_TPS_RESULTS+=("$agg_tps")
    echo ""

    # Stop server
    taskkill //F //IM llama-server.exe > /dev/null 2>&1 || true
    kill $SERVER_PID 2>/dev/null || true
    sleep 2

    # Early stop check: if at 4+ slots and aggregate doesn't scale vs baseline
    if [ "$n_slots" -ge 4 ] && [ "${#AGG_TPS_RESULTS[@]}" -ge 2 ]; then
        baseline="${AGG_TPS_RESULTS[0]}"
        should_stop=$(python3 -c "
b = float('${baseline}')
c = float('${agg_tps}')
if b > 0 and c > 0:
    ratio = c / b
    print(f'Scaling: {ratio:.2f}x vs baseline')
    if ratio < 1.5:
        print('STOP')
else:
    print('continue')
" 2>/dev/null)
        echo "  $should_stop"
        if echo "$should_stop" | grep -q "STOP"; then
            echo "  Aggregate throughput not scaling — stopping."
            break
        fi
        echo ""
    fi
done

echo "=== Summary ==="
echo "Slots | Aggregate tok/s"
for idx in "${!SLOT_COUNTS[@]}"; do
    if [ "$idx" -lt "${#AGG_TPS_RESULTS[@]}" ]; then
        printf "  %3d  | %s\n" "${SLOT_COUNTS[$idx]}" "${AGG_TPS_RESULTS[$idx]}"
    fi
done

echo ""
echo "=== Benchmark Complete ==="

#!/bin/bash
# Parallel inference scaling test for kv-compact
# Tests if compacted KV cache enables more parallel instances

set -e

TOOL="/c/Users/fabia/Projects/kv-compact/build-tool/llama-kv-compact.exe"
export PATH="/c/Users/fabia/Projects/kv-compact/build-tool/bin:$PATH"

MODEL="/c/Users/fabia/models/SmolLM3-3B-128K-Q4_K_M.gguf"
PROMPT="/c/Users/fabia/Projects/kv-compact/prompt-profile.txt"

PARALLEL_COUNTS=(1 2 4)

echo "=== Parallel Inference Scaling Test ==="
echo "Model: SmolLM3 3B Q4_K_M"
echo "Prompt: ~881 tokens, generate 50 tokens"
echo "Compaction: 5x (ratio=0.2)"
echo ""

for n_parallel in "${PARALLEL_COUNTS[@]}"; do
    echo "--- ${n_parallel} parallel instance(s) ---"

    # Launch N instances in parallel
    pids=()
    for i in $(seq 1 $n_parallel); do
        "$TOOL" -m "$MODEL" -f "$PROMPT" -c 2048 -n 50 -ngl 99 \
            --compact-ratio 0.2 --no-eviction \
            > "/tmp/kv_par_${n_parallel}_${i}.txt" 2>&1 &
        pids+=($!)
    done

    # Wait for all to complete
    t_start=$(date +%s%N)
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    t_end=$(date +%s%N)
    wall_ms=$(( (t_end - t_start) / 1000000 ))

    # Extract per-instance tok/s for compacted generation
    total_tps=0
    for i in $(seq 1 $n_parallel); do
        f="/tmp/kv_par_${n_parallel}_${i}.txt"
        # Get the LAST "Generated" line (compacted generation)
        tps=$(grep "Generated.*tok/s" "$f" | tail -1 | grep -oP '[\d.]+(?= tok/s)')
        prefill=$(grep "Prefill.*tok/s" "$f" | grep -oP '[\d.]+(?= tok/s)')
        echo "  Instance $i: prefill=${prefill} tok/s, gen=${tps} tok/s"
        total_tps=$(echo "$total_tps + ${tps:-0}" | bc -l 2>/dev/null || echo "$total_tps")
    done
    echo "  Wall time: ${wall_ms} ms"
    echo "  Aggregate throughput: ~${total_tps} tok/s"
    echo ""

    # Early stop: if 4 instances are slower than expected
    if [ "$n_parallel" -ge 4 ]; then
        # Check if per-instance speed dropped significantly
        last_tps=$(grep "Generated.*tok/s" "/tmp/kv_par_${n_parallel}_1.txt" | tail -1 | grep -oP '[\d.]+(?= tok/s)')
        baseline_tps=$(grep "Generated.*tok/s" "/tmp/kv_par_1_1.txt" | tail -1 | grep -oP '[\d.]+(?= tok/s)')
        ratio=$(echo "scale=2; ${last_tps:-0} / ${baseline_tps:-1}" | bc -l 2>/dev/null || echo "0")
        echo "  Speed ratio vs baseline: ${ratio}x"
        # If per-instance speed dropped below 50% of baseline, stop scaling
        if [ "$(echo "${ratio:-0} < 0.5" | bc -l 2>/dev/null || echo 1)" = "1" ]; then
            echo "  Per-instance speed dropped below 50% — stopping scaling test."
            break
        fi
    fi
done

echo "=== Done ==="

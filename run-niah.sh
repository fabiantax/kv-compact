#!/bin/bash
# Needle-in-a-Haystack Benchmark for kv-compact
# Tests retrieval accuracy across compaction ratios and needle depths

set -e

TOOL="./llama-kv-compact.exe"
PROMPT_DIR="/tmp/niah_prompts"
RESULTS_DIR="/tmp/niah_results"
HAYSTACK="/c/Users/fabia/Projects/kv-compact/prompt-profile.txt"

NEEDLE_VALUE="7492"
NEEDLE_LINE="IMPORTANT NOTE: The secret number referenced in the classified research report is ${NEEDLE_VALUE}. Remember this number."
QUESTION="Based on everything above, what is the secret number from the classified research report? Answer with just the number: "

# Models to test
SMOL_MODEL="/c/Users/fabia/models/SmolLM3-3B-128K-Q4_K_M.gguf"
GEMMA_MODEL="/c/Users/fabia/models/gemma-3-4b-it-heretic-v1.2-Q4_K_M.gguf"

DEPTHS=(0 25 50 75 100)
RATIOS=(0.2 0.1 0.05 0.033 0.02)
RATIO_NAMES=("5x" "10x" "20x" "30x" "50x")

mkdir -p "$PROMPT_DIR" "$RESULTS_DIR"

# Count content lines (non-blank) in haystack
TOTAL_LINES=$(grep -c '.' "$HAYSTACK")

# Generate prompt files with needle at different depths
echo "Generating NIAH prompt files..."
for depth in "${DEPTHS[@]}"; do
    outfile="${PROMPT_DIR}/niah_depth_${depth}.txt"

    # Calculate insertion point (line number in original file)
    # Depth 0% = after first paragraph, 100% = before last paragraph
    if [ "$depth" -eq 0 ]; then
        insert_after=2   # after line 2 (first blank line after intro)
    elif [ "$depth" -eq 25 ]; then
        insert_after=8   # after ENIAC paragraph + blank
    elif [ "$depth" -eq 50 ]; then
        insert_after=14  # after Unix paragraph + blank
    elif [ "$depth" -eq 75 ]; then
        insert_after=18  # after WWW paragraph + blank
    else
        insert_after=22  # after AI paragraph + blank (near end)
    fi

    # Build prompt: haystack[0:insert] + needle + haystack[insert:end] + question
    head -n "$insert_after" "$HAYSTACK" > "$outfile"
    echo "" >> "$outfile"
    echo "$NEEDLE_LINE" >> "$outfile"
    echo "" >> "$outfile"
    tail -n "+$((insert_after + 1))" "$HAYSTACK" >> "$outfile"
    echo "" >> "$outfile"
    echo "$QUESTION" >> "$outfile"

    tokens=$(wc -w < "$outfile")
    echo "  depth=${depth}%: insert_after=${insert_after}, ~${tokens} words"
done

# Run benchmark for a single model
run_model_benchmark() {
    local model_path="$1"
    local model_name="$2"

    echo ""
    echo "=========================================="
    echo "  NIAH Benchmark: ${model_name}"
    echo "=========================================="

    # Header
    printf "%-8s" "Depth"
    for rn in "${RATIO_NAMES[@]}"; do
        printf "%-10s" "$rn"
    done
    echo ""
    printf "%-8s" "-----"
    for rn in "${RATIO_NAMES[@]}"; do
        printf "%-10s" "------"
    done
    echo ""

    for depth in "${DEPTHS[@]}"; do
        printf "%-8s" "${depth}%"
        prompt="${PROMPT_DIR}/niah_depth_${depth}.txt"

        for ri in "${!RATIOS[@]}"; do
            ratio="${RATIOS[$ri]}"
            rname="${RATIO_NAMES[$ri]}"
            outfile="${RESULTS_DIR}/${model_name}_d${depth}_r${ratio}.txt"

            # Run compaction tool (skip eviction for speed, generate 30 tokens)
            "$TOOL" -m "$model_path" -f "$prompt" -c 2048 -n 30 -ngl 99 \
                --compact-ratio "$ratio" --no-eviction \
                > "$outfile" 2>&1

            # Check if output contains the needle value
            am_output=$(grep "Attention Matching output" "$outfile" | head -1)
            if echo "$am_output" | grep -q "$NEEDLE_VALUE"; then
                printf "%-10s" "PASS"
            else
                # Also check the raw generation output (last lines)
                gen_output=$(tail -20 "$outfile")
                if echo "$gen_output" | grep -q "$NEEDLE_VALUE"; then
                    printf "%-10s" "PASS"
                else
                    printf "%-10s" "FAIL"
                fi
            fi
        done
        echo ""
    done

    # Also test full cache (no compaction) as baseline
    echo ""
    echo "Full cache baseline (no compaction):"
    printf "%-8s" "Depth"
    printf "%-10s\n" "Result"
    for depth in "${DEPTHS[@]}"; do
        prompt="${PROMPT_DIR}/niah_depth_${depth}.txt"
        outfile="${RESULTS_DIR}/${model_name}_d${depth}_full.txt"

        "$TOOL" -m "$model_path" -f "$prompt" -c 2048 -n 30 -ngl 99 \
            --compact-ratio 0.99 --no-eviction \
            > "$outfile" 2>&1

        am_output=$(grep "Attention Matching output" "$outfile" | head -1)
        full_output=$(grep "Full cache output" "$outfile" | head -1)
        if echo "$full_output $am_output" | grep -q "$NEEDLE_VALUE"; then
            printf "%-8s%-10s\n" "${depth}%" "PASS"
        else
            # Check raw output
            gen_output=$(tail -20 "$outfile")
            if echo "$gen_output" | grep -q "$NEEDLE_VALUE"; then
                printf "%-8s%-10s\n" "${depth}%" "PASS"
            else
                printf "%-8s%-10s\n" "${depth}%" "FAIL"
            fi
        fi
    done
}

# Parse model selection from args (default: both)
MODEL_ARG="${1:-both}"

if [ "$MODEL_ARG" = "smol" ] || [ "$MODEL_ARG" = "both" ]; then
    run_model_benchmark "$SMOL_MODEL" "SmolLM3"
fi

if [ "$MODEL_ARG" = "gemma" ] || [ "$MODEL_ARG" = "both" ]; then
    run_model_benchmark "$GEMMA_MODEL" "Gemma3"
fi

echo ""
echo "=== NIAH Benchmark Complete ==="
echo "Detailed outputs in: ${RESULTS_DIR}/"

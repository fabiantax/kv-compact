#!/bin/bash
# Bisect fork commits to find server regression source
# Tests llama-server tok/s at key commit boundaries

REPO="/c/Users/fabia/Projects/llama.cpp/llama-flash-attn"
MODEL="/c/Users/fabia/models/SmolLM3-3B-128K-Q4_K_M.gguf"
HOST="127.0.0.1"
PORT=9300
RESULTS="/tmp/bisect_results.txt"

export INCLUDE="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/include;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/ucrt;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/um;C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/winrt;C:/VulkanSDK/1.4.341.1/Include"
export LIB="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/lib/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/um/x64"
export PATH="$PATH:/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja:/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"

echo "=== Fork Bisect: Server Regression ===" | tee "$RESULTS"
echo "Model: SmolLM3-3B, single request, 50 tokens" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

test_commit() {
    local label="$1"
    local commit="$2"

    echo -n "[$label] $commit ... " | tee -a "$RESULTS"

    cd "$REPO"
    git checkout "$commit" --quiet 2>/dev/null

    # Build llama-server
    cd build-win
    ninja -j 16 bin/llama-server.exe > /tmp/bisect_build.log 2>&1
    if [ $? -ne 0 ]; then
        echo "BUILD_FAIL" | tee -a "$RESULTS"
        return
    fi

    taskkill //F //IM llama-server.exe > /dev/null 2>&1
    sleep 2

    # Start server
    bin/llama-server.exe -m "$MODEL" -c 4096 -ngl 99 \
        --host "$HOST" --port "$PORT" > /tmp/bisect_server.log 2>&1 &
    local PID=$!

    # Wait for ready
    local ready=0
    for i in $(seq 1 60); do
        if ! kill -0 $PID 2>/dev/null; then echo "SERVER_DIED" | tee -a "$RESULTS"; return; fi
        if curl -s "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
            ready=1; break
        fi
        sleep 1
    done
    if [ $ready -eq 0 ]; then echo "TIMEOUT" | tee -a "$RESULTS"; kill $PID 2>/dev/null; return; fi

    # Warmup
    curl -s --max-time 60 -X POST "http://${HOST}:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d '{"prompt":"Hello","n_predict":5,"temperature":0}' > /dev/null 2>&1

    # Benchmark (3 runs, take best)
    local best=0
    for run in 1 2 3; do
        local tps=$(curl -s --max-time 60 -X POST "http://${HOST}:${PORT}/completion" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"Write a Python function to check if a number is prime:","n_predict":50,"temperature":0}' 2>/dev/null | \
            python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(f'{d.get(\"timings\",{}).get(\"predicted_per_second\",0):.1f}')" 2>/dev/null)
        if [ -z "$tps" ]; then tps="0.0"; fi
        local cmp=$(python3 -c "print(1 if float('$tps') > float('$best') else 0)" 2>/dev/null)
        if [ "$cmp" = "1" ]; then best="$tps"; fi
    done

    echo "${best} tok/s" | tee -a "$RESULTS"
    taskkill //F //IM llama-server.exe > /dev/null 2>&1
    sleep 2
}

# Test points (chronological order within fork-specific commits)
test_commit "BASELINE (last upstream)"  "e22cd0aa1"
test_commit "G1: GDN op (#19504)"       "c5a778891"
test_commit "G2: CUDA UMA+FA tune"      "2034f53c7"
test_commit "G3: Backend sync"          "f229ccc90"
test_commit "G4: FA auto-tune"          "4bdaa8f32"
test_commit "G5: UMA config+APEX split" "c9c19480b"
test_commit "G6: APEX scheduling"       "e0bb11e38"
test_commit "G7: Vulkan RDNA3+FA+UMA"   "7d21e4397"
test_commit "G8: MoE prefetch"          "3f39d2124"
test_commit "G9: Fused SSM+batched"     "e69b7823b"
test_commit "HEAD"                       "686f36493"

# Restore HEAD
cd "$REPO" && git checkout - --quiet 2>/dev/null

echo "" | tee -a "$RESULTS"
echo "=== Done ===" | tee -a "$RESULTS"
cat "$RESULTS"

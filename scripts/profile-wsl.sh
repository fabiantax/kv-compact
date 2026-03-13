#!/bin/bash
# Profile kv-compact on Linux/WSL2 using perf and generate flamegraph
# This provides better visualization than Windows profiling tools

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== KV Compact Flamegraph Profiling ===${NC}"
echo ""

# Check if we're on WSL2 or Linux
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}Running on WSL2${NC}"
    IS_WSL=1
else
    echo -e "${GREEN}Running on native Linux${NC}"
    IS_WSL=0
fi

# Check dependencies
echo "Checking dependencies..."

if ! command -v perf &> /dev/null; then
    echo -e "${RED}ERROR: perf not found${NC}"
    echo "Install: sudo apt-get install linux-tools-common linux-tools-generic"
    exit 1
fi

if ! command -v flamegraph.pl &> /dev/null; then
    echo -e "${YELLOW}WARNING: flamegraph.pl not found${NC}"
    echo "Install from: https://github.com/brendangregg/FlameGraph"
    echo ""
    read -p "Install FlameGraph now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git clone https://github.com/brendangregg/FlameGraph.git /tmp/flamegraph
        export PATH="/tmp/flamegraph:$PATH"
    fi
fi

# Configuration
BUILD_DIR="${BUILD_DIR:-./build}"
OUTPUT_DIR="./profiling"
EXE="$BUILD_DIR/llama-kv-compact"
MODEL="${MODEL:-$HOME/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
PROMPT_FILE="/tmp/kv-compact-prompt.txt"
DURATION="${DURATION:-30}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if executable exists
if [ ! -f "$EXE" ]; then
    echo -e "${RED}ERROR: Executable not found: $EXE${NC}"
    echo "Build first with: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build"
    exit 1
fi

# Create prompt file
cat > "$PROMPT_FILE" << 'EOF'
The quick brown fox jumps over the lazy dog. This is a test of the KV cache compaction system.
We want to profile the performance and identify bottlenecks in the algorithm.
EOF

# Generate a timestamp for output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PERF_DATA="$OUTPUT_DIR/kv-compact_$TIMESTAMP.data"
FOLDED_DATA="$OUTPUT_DIR/kv-compact_$TIMESTAMP.folded"
FLAMEGRAPH="$OUTPUT_DIR/kv-compact_$TIMESTAMP.svg"

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Executable: $EXE"
echo "  Model: $MODEL"
echo "  Duration: ${DURATION}s"
echo "  Output: $FLAMEGRAPH"
echo ""

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo -e "${YELLOW}WARNING: Model not found, creating minimal test...${NC}"
    USE_MINIMAL=1
else
    USE_MINIMAL=0
fi

echo -e "${GREEN}Step 1: Starting perf recording...${NC}"
echo "Running: perf record -F 99 -g --call-graph dwarf -o $PERF_DATA"

if [ $IS_WSL -eq 1 ]; then
    echo -e "${YELLOW}NOTE: On WSL2, perf has limitations.${NC}"
    echo "If profiling fails, consider:"
    echo "  1. Running on native Linux"
    echo "  2. Using Visual Studio on Windows"
    echo "  3. Using the instrumented build instead"
    echo ""
fi

# Run perf record
if [ $USE_MINIMAL -eq 1 ]; then
    # Run with minimal parameters (no actual model)
    timeout ${DURATION}s perf record \
        -F 99 \
        -g \
        --call-graph dwarf \
        -o "$PERF_DATA" \
        "$EXE" --help 2>/dev/null || true
else
    timeout ${DURATION}s perf record \
        -F 99 \
        -g \
        --call-graph dwarf \
        -o "$PERF_DATA" \
        "$EXE" \
        -m "$MODEL" \
        -f "$PROMPT_FILE" \
        -n 50 || true
fi

echo ""
echo -e "${GREEN}Step 2: Processing perf data...${NC}"

# Extract call traces
perf script -i "$PERF_DATA" > "$FOLDED_DATA"

echo ""
echo -e "${GREEN}Step 3: Generating flamegraph...${NC}"

# Generate flamegraph
if command -v flamegraph.pl &> /dev/null; then
    flamegraph.pl \
        --title "KV Cache Compaction Flamegraph" \
        --width 1600 \
        --countname samples \
        "$FOLDED_DATA" > "$FLAMEGRAPH"

    echo -e "${GREEN}Flamegraph generated: $FLAMEGRAPH${NC}"
else
    echo -e "${YELLOW}flamegraph.pl not found, skipping SVG generation${NC}"
    echo "Raw data saved to: $FOLDED_DATA"
fi

echo ""
echo -e "${GREEN}Step 4: Generating report...${NC}"

# Generate a text report
REPORT="$OUTPUT_DIR/kv-compact_$TIMESTAMP.txt"
perf report -i "$PERF_DATA" --stdio > "$REPORT"

echo ""
echo -e "${GREEN}=== Profiling Complete ===${NC}"
echo ""
echo "Files generated:"
echo "  - Flamegraph: $FLAMEGRAPH"
echo "  - Report:     $REPORT"
echo "  - Folded:     $FOLDED_DATA"
echo "  - Perf data:  $PERF_DATA"
echo ""
echo -e "${YELLOW}To view the flamegraph:${NC}"
echo "  1. Open $FLAMEGRAPH in a web browser"
echo "  2. Hover over blocks to see function names"
echo "  3. Click to zoom in"
echo ""
echo -e "${YELLOW}To analyze further:${NC}"
echo "  perf report -i $PERF_DATA"
echo "  perf annotate -i $PERF_DATA --stdio"
echo ""

# Try to open the flamegraph automatically
if command -v xdg-open &> /dev/null; then
    xdg-open "$FLAMEGRAPH" 2>/dev/null &
elif command -v open &> /dev/null; then
    open "$FLAMEGRAPH" 2>/dev/null &
fi

echo -e "${GREEN}Done!${NC}"

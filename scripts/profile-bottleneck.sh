#!/usr/bin/env bash
# Profile kv-compact bottleneck using samply (macOS native profiler)
#
# Usage:
#   ./scripts/profile-bottleneck.sh [target] [extra args...]
#
# Examples:
#   ./scripts/profile-bottleneck.sh                     # profile bench-kv-compact-quality
#   ./scripts/profile-bottleneck.sh bench-kv-compact    # profile throughput benchmark
#   ./scripts/profile-bottleneck.sh bench-opt-expected-attention
#
# Requires: samply (cargo install samply)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

TARGET="${1:-bench-kv-compact-quality}"
shift || true

echo "=== kv-compact Bottleneck Profiler ==="
echo "Target: $TARGET"
echo ""

# Check samply is installed
if ! command -v samply &>/dev/null; then
    echo "ERROR: samply not found. Install with:"
    echo "  cargo install samply"
    exit 1
fi

# Build with debug info if not already
if [ ! -d "$BUILD_DIR" ]; then
    echo "Configuring build with RelWithDebInfo..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DKV_COMPACT_BUILD_TOOL=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
fi

echo "Building $TARGET..."
cd "$BUILD_DIR"
cmake --build . --target "$TARGET" -j"$(sysctl -n hw.ncpu)" 2>&1 | tail -5

BINARY="$BUILD_DIR/$TARGET"
if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY not found or not executable"
    exit 1
fi

echo ""
echo "Profiling $TARGET with samply..."
echo "Output will be saved to $PROJECT_DIR/profile-$TARGET.json"
echo ""

cd "$PROJECT_DIR"
samply record -- "$BINARY" "$@"

echo ""
echo "Done. Open the flamegraph HTML that samply opened in your browser."
echo ""
echo "=== Expected Bottleneck Breakdown ==="
echo "  Scoring (Q@K^T matmul):     ~30%"
echo "  State I/O (serialize/deser): ~35-60%"
echo "  Value refit (LS solve):      ~15%"
echo "  Softmax + importance fusion:  ~9%"
echo ""
echo "Compare actual flamegraph with these estimates."

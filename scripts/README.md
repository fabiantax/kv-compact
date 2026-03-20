# Profiling kv-compact

This directory contains scripts and tools for profiling the kv-compact library to identify performance bottlenecks.

## 🔥 Flamegraph Profiling

### Linux / WSL2 (Recommended)

The easiest way to generate flamegraphs is on Linux or WSL2 using `perf`:

```bash
cd /path/to/kv-compact
./scripts/profile-wsl.sh
```

This will:
1. Record performance data using `perf record`
2. Generate a flamegraph SVG
3. Create a detailed report
4. Open the flamegraph in your browser

**Requirements:**
- Linux or WSL2
- `perf` tool (install via `sudo apt-get install linux-tools-generic`)
- FlameGraph tools (auto-downloaded)

### Windows Native

Windows doesn't have native flamegraph support, but you can:

#### Option 1: Windows Performance Toolkit (WPT)

```powershell
cd C:\Users\fabia\Projects\kv-compact
.\scripts\profile-windows.ps1
```

This generates an ETL trace that can be viewed in Windows Performance Analyzer (WPA).

#### Option 2: Simple Instrumentation

Enable profiling instrumentation at build time:

```bash
cmake -B build -DKV_COMPACT_ENABLE_PROFILING=ON
cmake --build build
```

Then run the instrumented binary:

```bash
./profiled-kv-compact -m model.gguf -f prompt.txt -n 50
```

This will print timing information for each function at the end of execution.

## 📊 Understanding Flamegraphs

### Reading the Flamegraph

- **X-axis**: Population (width = time spent)
- **Y-axis**: Stack depth (call hierarchy)
- **Color**: Random (warm colors for frequently used functions)

### Interpreting Results

1. **Wide blocks** = Functions that take a lot of time
2. **Narrow towers** = Deep call stacks
3. **Wide bases** = Functions that call many slow sub-functions

### Common Patterns

```
# Good: Balanced distribution
compaction (50ms)
├── key_selection (10ms)
├── nnls_fitting (25ms)
└── value_refitting (15ms)

# Bad: Bottleneck in one function
compaction (500ms)
└── nnls_fitting (450ms)  ← 90% of time!
    └── matrix_inverse (400ms)  ← Problem here!
```

## 🎯 Expected Performance Characteristics

Based on our tests:

| Token Count | Expected Compaction Time | Bottleneck |
|-------------|-------------------------|------------|
| 100 | ~100 ms | Key selection |
| 1,000 | ~10 s | NNLS fitting (O(n²)) |
| 10,000 | ~100 s (est.) | NNLS fitting |

**Key Findings:**
- **NNLS fitting** dominates for large token counts
- **Key selection** is relatively fast
- **Value refitting** uses efficient least-squares

## 🛠️ Optimization Strategies

### If NNLS is the bottleneck:

1. **Use fewer iterations**:
   ```cpp
   NnlsConfig.max_iterations = 50;  // default: 100
   ```

2. **Use smaller reference queries**:
   ```cpp
   CompactConfig.num_ref_queries = 32;  // default: 64
   ```

3. **Enable early stopping**:
   ```cpp
   NnlsConfig.tolerance = 1e-3;  // default: 1e-4
   ```

### If key selection is slow:

1. **Use submodular selection** (better quality, slower):
   ```cpp
   CompactConfig.use_submodular = true;
   ```

2. **Reduce sampling**:
   ```cpp
   KeySelectorConfig.sample_heads = 4;  // default: all heads
   ```

## 📈 Profiling Checklist

- [ ] Run with different token counts (100, 1K, 10K)
- [ ] Profile both prefill and generation phases
- [ ] Compare WSL2 vs Windows native
- [ ] Check memory allocations
- [ ] Verify SIMD optimizations are enabled

## 🔧 Advanced Profiling

### CPU Counters (Linux)

```bash
perf stat -e cycles,instructions,cache-references,cache-misses \
    ./llama-kv-compact -m model.gguf -f prompt.txt -n 50
```

### Memory Profiling

```bash
valgrind --tool=massif \
    ./llama-kv-compact -m model.gguf -f prompt.txt -n 50

# Visualize:
ms_print massif.out.xxxxx
```

### Call Graph (Linux)

```bash
gprof -b profiled-kv-compact gmon.out > profile.txt
```

## 📚 Resources

- **FlameGraph**: https://github.com/brendangregg/FlameGraph
- **Perf Tutorial**: https://www.brendangregg.com/perf.html
- **WPA**: https://learn.microsoft.com/en-us/windows-hardware/test/wpa/
- **VTune**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html

## 🐛 Troubleshooting

### "perf not found" (WSL2)

```bash
sudo apt-get update
sudo apt-get install linux-tools-common linux-tools-generic
```

### "WSL2 perf doesn't work"

WSL2 has limited perf support. Options:
1. Use Windows Performance Toolkit instead
2. Build on native Linux
3. Use simple instrumentation (`-DKV_COMPACT_ENABLE_PROFILING=ON`)

### "Flamegraph shows no data"

- Ensure you ran the application with actual workload
- Check that perf captured data: `perf report -i perf.data`
- Try increasing sampling frequency: `-F 999` instead of `-F 99`

### "Build fails with profiling enabled"

Make sure you have a compatible compiler:
- GCC 7+ or Clang 5+
- On Windows, MSVC 19.14+

## 📝 Example Session

```bash
# 1. Enable profiling instrumentation
cmake -B build -DKV_COMPACT_ENABLE_PROFILING=ON

# 2. Build
cmake --build build --target profiled-kv-compact

# 3. Run with profiling
./build/profiled-kv-compact \
    -m ~/.lmstudio/models/.../Qwen3-4B-Q4_K_M.gguf \
    -f prompt.txt \
    -n 50

# Output:
# [Prefill] 1250ms
# [Compact] 96ms
# [key_selection] 12ms
# [nnls_fitting] 78ms
# [value_refitting] 6ms
# [Generate] 3200ms
#
# === Profiling Summary ===
# Function                  Calls      Total (ms)   Avg (us)
# ----------------------------------------------------------------
# nnls_solve                  36       78000.00    2166.67
# compute_attention          72       12000.00     166.67
# select_top_keys              1       12000.00   12000.00
# least_squares_solve        36        6000.00     166.67
```

This shows that NNLS is taking 81% of the compaction time, making it the prime target for optimization.

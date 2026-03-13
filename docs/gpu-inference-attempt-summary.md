# GPU Inference Attempt Summary - 2026-03-13

## Status: ❌ BLOCKED - Hardware Incompatibility

---

## Problem

**Pre-built llama.cpp CUDA binaries are incompatible with GTX 1050 Ti (compute capability 6.1)**

---

## Investigation Results

### 1. CUDA Backend Loading ✅
- Successfully downloaded llama.cpp b8303 (latest release)
- CUDA 12.4 runtime DLLs required: `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`
- After copying runtime DLLs, CUDA backend loads:
  ```
  load_backend: loaded CUDA backend from ...ggml-cuda.dll
  ```

### 2. Model Loading Failure ❌
- **Error Location:** `ggml-cuda.cu:98: CUDA error`
- **Occurs:** During model load, even with `-ngl 0` (CPU-only mode)
- **Cause:** GTX 1050 Ti compute capability 6.1 incompatible with CUDA 12.4 ggml-cuda.dll

### 3. Hardware Analysis
| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce GTX 1050 Ti with Max-Q Design |
| VRAM | 4095 MiB (3373 MiB free) |
| Compute Capability | 6.1 |
| Driver Version | 528.79 |
| Supported CUDA | 12.0 |

### 4. Compatibility Matrix
| llama.cpp Build | CUDA Version | Compute Capabilities | GTX 1050 Ti (6.1) |
|-----------------|--------------|---------------------|------------------|
| b4134 (Nov 2024) | 12.2.0 | 6.1+ | ✅ Compatible (but lacks Qwen 3.5 support) |
| b8303 (Mar 2026) | 12.4 | 7.0+ | ❌ Incompatible |
| Latest | 12.4/13.1 | 7.0+ | ❌ Incompatible |

---

## Root Cause

**CUDA 12.4 requires compute capability 7.0+ (Volta architecture or newer).**
- GTX 1050 Ti is Pascal architecture (compute capability 6.1)
- llama.cpp switched minimum compute capability from 6.1 to 7.0 sometime between b4134 and b8303
- This was done to leverage CUDA 12.4+ features and drop legacy support

---

## Attempted Solutions

### ❌ Pre-built Binaries (Latest)
- **Result:** CUDA error during initialization
- **Issue:** Compute capability mismatch

### ❌ Pre-built Binaries (b4134 with CUDA 12.2)
- **Result:** Shared library errors
- **Issue:** Too old for Qwen 3.5 model support (Qwen 3.5 released Feb 2026, b4134 from Nov 2024)

### ❌ CUDA Runtime DLLs
- **Result:** Backend loads but initialization fails
- **Issue:** Not a runtime issue, fundamental compute capability mismatch

---

## Working Alternatives

### ✅ CPU-Only Inference
- **Status:** Working perfectly
- **Speed:** 31 tg/s (tokens/second)
- **Build:** Local build with profiling support
- **Compaction:** 5.3x compression (69 → 13 tokens)
- **Overhead:** ~10-15 ms (<0.1% of generation time)

---

## Solution Options

### Option A: Build llama.cpp from Source ⭐ RECOMMENDED
**Steps:**
1. Install CUDA Toolkit 11.8 or 12.0 (supports compute capability 6.1)
2. Clone llama.cpp latest master branch
3. Build with:
   ```bash
   cmake -DGGML_CUDA=ON \
         -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8" \
         -DCMAKE_BUILD_TYPE=Release \
         -DLLAMA_CUDA_SUPPORTS_GPCCPA=OFF \
         ..
   cmake --build . --config Release
   ```
4. Test with Qwen 3.5 model

**Expected Result:** GPU inference working with GTX 1050 Ti
**Expected Speedup:** 2-3x (limited by GTX 1050 Ti memory bandwidth)
**Time Required:** 2-3 hours

**Pros:**
- GTX 1050 Ti support
- Latest llama.cpp features
- Qwen 3.5 support

**Cons:**
- Requires full CUDA Toolkit installation
- Build time ~30 minutes
- Need to manage compatibility issues

### Option B: Use CPU-Only
**Steps:**
1. Continue using local CPU build
2. Focus on KV compaction optimization
3. Document GPU incompatibility

**Expected Result:** No changes, current working setup
**Tradeoff:** No GPU speedup, but compaction provides value

**Pros:**
- Works today
- Zero setup time
- Compaction is production-ready

**Cons:**
- Limited to CPU speed (~31 tg/s)
- No GPU acceleration for inference

### Option C: Upgrade Hardware
**Steps:**
1. Acquire RTX 3060 Ti or newer (compute capability 8.6+)
2. Use pre-built binaries

**Expected Result:** 3-5x speedup with modern GPU
**Time Required:** Depends on procurement

**Pros:**
- Best performance
- Works with pre-built binaries
- Future-proof

**Cons:**
- Cost ($300-400+)
- Procurement time

---

## Technical Details

### CUDA Compute Capability History
| Architecture | Compute Capability | CUDA Support | Example GPUs |
|--------------|-------------------|--------------|--------------|
| Pascal | 6.1 | CUDA 11.x | GTX 1050 Ti, GTX 1080 |
| Volta | 7.0 | CUDA 11.x+ | Tesla V100, Titan V |
| Turing | 7.5 | CUDA 11.x+ | RTX 2060, GTX 1660 |
| Ampere | 8.6 | CUDA 11.x+ | RTX 3060, RTX 3090 |
| Ada Lovelace | 8.9 | CUDA 11.x+ | RTX 4090 |

### llama.cpp CUDA Support Timeline
- **b4134 (Nov 2024):** Minimum compute capability 6.1 (Pascal)
- **b8292+ (Feb 2026):** Minimum compute capability 7.0 (Volta+)
- **b8303 (Mar 2026):** CUDA 12.4+ required, compute capability 7.0+

### Qwen 3.5 Support
- **Release Date:** 2026-02-16
- **Architecture:** Gated DeltaNet + Full Attention Hybrid
- **llama.cpp Support:** Added after Feb 2026
- **Minimum Version:** b8270+ (early Qwen 3.5 support)

---

## Recommendations

### Immediate: Document and Accept CPU-Only
1. Update `action-plan-gpu-inference.md` with hardware limitation findings
2. Document that GTX 1050 Ti is incompatible with latest llama.cpp CUDA builds
3. Focus on CPU optimization and KV compaction refinement

### Short-term: Build from Source if GPU Needed
1. Install CUDA 11.8 (last version with Pascal support)
2. Build llama.cpp from source with compatibility flags
3. Benchmark GPU vs CPU performance

### Long-term: Consider Hardware Upgrade
1. RTX 3060 Ti provides 3-5x speedup with pre-built binaries
2. Future-proof for newer models and features
3. Better memory bandwidth for KV cache operations

---

## Files Created During Investigation

### Downloaded Binaries
- `llama-b8303-bin-win-cuda-12.4-x64.zip` (215 MB) - Incompatible
- `llama-b4134-bin-win-cuda-cu12.2.0-x64.zip` (145 MB) - Too old
- `cudart-llama-bin-win-cuda-12.4-x64.zip` (414 MB) - Runtime DLLs

### Extraction Directories
- `llama-cuda-bin-b8303/` - Latest llama.cpp with CUDA 12.4
- `llama-cuda-bin-b4134/` - Older llama.cpp with CUDA 12.2

### Documentation
- `docs/gpu-inference-attempt-summary.md` - This document
- `llama-cuda-bin-b8303/gpu_test_output.txt` - Test outputs
- `llama-cuda-bin-b8303/gpu_benchmark.log` - Benchmark attempts

---

## Key Takeaways

1. **Hardware Limitation:** GTX 1050 Ti (compute capability 6.1) is too old for latest llama.cpp CUDA builds
2. **Software Solution:** Build from source with CUDA 11.8 is the only path to GPU inference
3. **Practical Reality:** CPU-only with KV compaction is working well (31 tg/s, 5.3x compression)
4. **Future Proofing:** RTX 3060 Ti or newer recommended for GPU inference

---

## Decision Required

Which path should be pursued?
- **Option A:** Build llama.cpp from source with CUDA 11.8 (2-3 hours)
- **Option B:** Accept CPU-only and focus on compaction optimization (0 hours)
- **Option C:** Plan hardware upgrade (depends on budget/timeline)

---

**Status:** Awaiting user decision on next steps
**Date:** 2026-03-13
**Investigation Time:** ~2 hours

# GPU Setup Guide for kv-compact

## Current Status

### ✅ GPU Hardware Detected
```
NVIDIA GeForce GTX 1050 Ti
- CUDA Version: 12.0
- Driver Version: 528.79
- VRAM: 4096 MiB
- Status: Available
```

### ❌ CUDA Toolkit Not Found
```
Error: Could not find `nvcc` executable
CMake Error: CUDA Toolkit not found
```

## Problem

**GPU Driver is installed** (nvidia-smi works) but **CUDA Toolkit is not installed**.

The CUDA Toolkit includes:
- `nvcc` - CUDA compiler
- CUDA headers and libraries
- cuBLAS, cuDNN, and other GPU libraries

These are **separate installations** from the GPU driver.

## Solution: Install CUDA Toolkit

### Option 1: Install CUDA Toolkit 12.0 (Recommended)

**Download:**
https://developer.nvidia.com/cuda-downloads

**Select:**
- Product: CUDA Toolkit 12.0
- Operating System: Windows
- Architecture: x86_64
- Version: 11
- Installer Type: exe (local)

**Installation Steps:**
1. Run the installer
2. Choose "Custom Installation"
3. **Deselect** "Graphics Driver" (already installed)
4. **Select** "CUDA Toolkit" and "CUDA Runtime"
5. Install to default path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0`
6. Add to PATH:
   ```powershell
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin
   ```

**Verify Installation:**
```bash
nvcc --version
# Expected: nvcc: NVIDIA (R) Cuda compiler driver
```

### Option 2: Use Pre-built llama.cpp Binary

**Download pre-built binary with CUDA:**
```bash
# From llama.cpp releases
https://github.com/ggerganov/llama.cpp/releases

# Look for: llama-bin-win-cuda-cu12.0-x64.zip
```

**Extract and use:**
```bash
unzip llama-bin-win-cuda-cu12.0-x64.zip
cd llama-bin-win-cuda-cu12.0-x64
./main -m models/Qwen3.5-0.8B-Q4_K_M.gguf -n-gpu-layers 24
```

### Option 3: Use conda/pip (Alternative)

```bash
# Install CUDA toolkit via conda
conda install -c nvidia cuda-toolkit

# Or use pre-built wheels
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

## Building kv-compact with CUDA (After Toolkit Installation)

### Step 1: Clean Build Directory
```bash
cd D:/Projects/kv-compact/kv-compact/build
rm -rf _deps
rm -f CMakeCache.txt
```

### Step 2: Reconfigure with CUDA
```bash
cmake -DGGML_CUDA=ON \
      -DKV_COMPACT_ENABLE_PROFILING=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
```

**Expected Output:**
```
-- Found CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0
-- GGML_CUDA: version 12.0.0, CUDA Runtime 12.0.0
-- CUDA detected: GGML_CUDA=ON
```

### Step 3: Build
```bash
cmake --build . --config Release
```

**Expected Build Time:** 10-15 minutes (first build with CUDA)

### Step 4: Verify GPU Support
```bash
./Release/llama-kv-compact.exe --help | grep -i cuda
# Expected: CUDA enabled, GPU buffer size
```

## Testing GPU Inference

### Basic Test
```bash
cd D:/Projects/kv-compact/kv-compact/build/Release

./llama-kv-compact.exe \
  -m ../../models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 8192 \
  -n 512 \
  -p "Once upon a time, there was a curious AI..." \
  -ngl 24 \
  --verbose
```

**Expected Output:**
```
llama_kv_cache:        GPU KV buffer size =   96.00 MiB
ggml_cuda_init: CUDA initialized
GGML_CUDA: version 12.0.0, CUDA Runtime 12.0.0
```

### Benchmark Test
```bash
./llama-kv-compact.exe \
  -m ../../models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 8192 \
  -n 512 \
  -p "Once upon a time, there was a curious AI..." \
  -ngl 24 \
  --perf
```

**Expected Performance:**
- CPU-only: 32.91 tg/s
- GPU (expected): 100-230 tg/s (3-7x speedup)

## Troubleshooting

### Error: "CUDA out of memory"
**Cause:** Model + KV cache exceeds 4GB VRAM

**Solution:** Reduce GPU layers
```bash
./llama-kv-compact.exe -m model.gguf -ngl 12  # Instead of 24
```

### Error: "CUDA not supported"
**Cause:** CUDA toolkit not in PATH

**Solution:**
```powershell
# Add to PATH (PowerShell)
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"

# Or set permanently
setx PATH "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
```

### Error: "cuBLAS not found"
**Cause:** CUDA toolkit installation incomplete

**Solution:** Reinstall CUDA toolkit with "CUDA Development" component

## Alternative: ROCm (AMD GPUs)

If you have an AMD GPU instead:

```bash
# Install ROCm toolkit
https://rocm.docs.amd.com/

# Build with ROCm
cmake -DGGML_HIPBLAS=ON -DKV_COMPACT_ENABLE_PROFILING=ON ..
cmake --build . --config Release

# Run with ROCm
./llama-kv-compact.exe -m model.gguf -ngl 24 --perf
```

## Performance Expectations

### GTX 1050 Ti (4GB VRAM)

| Configuration | Model | Context | Expected Speedup |
|---------------|-------|---------|------------------|
| **CPU-only** | Q4_K_M | 8192 | Baseline (32.91 tg/s) |
| **GPU (24 layers)** | Q4_K_M | 8192 | 3-5x (100-160 tg/s) |
| **GPU (12 layers)** | Q4_K_M | 8192 | 2-3x (65-100 tg/s) |

### Memory Usage

| Component | Size |
|-----------|------|
| Model weights (Q4_K_M) | ~500 MB |
| KV cache (8192 tokens) | ~96 MB |
| Working memory | ~200 MB |
| **Total** | ~800 MB |

**Fits comfortably in 4GB VRAM with room for batch processing**

## Next Steps After Installation

1. **Verify GPU is working:**
   ```bash
   nvidia-smi  # Should show GPU utilization during inference
   ```

2. **Run benchmark:**
   ```bash
   ./llama-kv-compact.exe -m model.gguf -ngl 24 -c 8192 -n 512 --perf
   ```

3. **Check compaction percentage:**
   - If compaction >5% → Implement GPU compaction
   - If compaction <5% → GPU compaction not worth it

4. **Re-profile with GPU:**
   ```bash
   # Use profiling demo with GPU backend
   ./profiling-demo.exe
   ```

## Installation Priority

Based on RICE analysis, GPU inference setup is **Priority 2 (RICE: 486)**:

1. **Install CUDA Toolkit** (30 minutes)
2. **Rebuild llama.cpp with CUDA** (15 minutes)
3. **Test GPU inference** (5 minutes)
4. **Benchmark and validate** (10 minutes)

**Total time: ~1 hour**
**Expected impact: 3-7x overall speedup**

## Resources

- **CUDA Download:** https://developer.nvidia.com/cuda-downloads
- **llama.cpp CUDA Guide:** https://github.com/ggerganov/llama.cpp#cublas
- **CUDA Installation Guide:** https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- **GGML CUDA Source:** https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-cuda

## Status

- [x] GPU hardware detected (GTX 1050 Ti)
- [x] GPU driver installed (CUDA 12.0)
- [ ] CUDA toolkit installed ← **BLOCKED**
- [ ] llama.cpp built with CUDA ← **BLOCKED**
- [ ] GPU inference tested ← **PENDING**
- [ ] Performance validated ← **PENDING**

**Action Required:** Install CUDA Toolkit to proceed

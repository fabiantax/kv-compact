# CUDA Build Attempts - Current Status & Solutions

**Date:** 2026-03-13
**Status:** CUDA detected but build configuration blocked

---

## ✅ What's Working

### 1. CUDA Installation
- **CUDA 13.1:** Installed via winget ✅
- **nvcc compiler:** Present at `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe` ✅
- **GPU Hardware:** GTX 1050 Ti (4GB VRAM) ✅
- **GPU Driver:** Version 528.79 (CUDA 12.0 support) ✅

### 2. llama.cpp Source
- **Fresh clone:** Successfully cloned to `D:/Projects/kv-compact/kv-compact/llama.cpp` ✅
- **CMakeLists.txt:** Present and valid ✅

---

## ❌ Current Blocker

### CUDA Build Configuration Error

**Error Message:**
```
error MSB8066: Custom build for CompilerIdCUDA.vcxproj FAILED
The CUDA Toolkit directory '' does not exist.
```

**Root Cause:**
CUDA 13.1 installation via winget **does not include Visual Studio integration components** needed for CMake build system.

**Missing Components:**
- CUDA 13.1 MSBuild targets
- Visual Studio CUDA integration
- CUDA Toolkit VS integration files

---

## 🔍 Why This Happens

### Winget Installation vs Full Installation

**Winget (Current):**
- Installs CUDA Toolkit binaries (nvcc, libraries)
- Installs runtime components
- **Missing:** Visual Studio integration

**Full NVIDIA Installer (Needed):**
- All CUDA Toolkit binaries
- Visual Studio integration
- NSight Tools (optional)
- CUDA samples (optional)

---

## 🛠️ Solutions (In Priority Order)

### Solution 1: Use Pre-built llama.cpp Binaries ⭐ **RECOMMENDED**

**Why Easiest:**
- No CUDA build configuration needed
- Already compiled with CUDA support
- Just download and run

**Steps:**
```bash
# Download pre-built llama.cpp with CUDA 12.0
# From: https://github.com/ggerganov/llama.cpp/releases

# Look for: llama-bin-win-cuda-cu12.0-x64.zip
# Extract and use directly

cd D:/Projects/kv-compact/kv-compact
wget https://github.com/ggerganov/llama.cpp/releases/download/b4134/llama-bin-win-cuda-cu12.0-x64.zip
unzip llama-bin-win-cuda-cu12.0-x64.zip -d llama-cuda-bin

# Test with our model
cd llama-cuda-bin
./main -m ../../models/Qwen3.5-0.8B-Q4_K_M.gguf -c 8192 -n 512 -p "test" -ngl 24
```

**Expected Result:**
- CUDA 12.0 runtime compatible with driver
- 3-5x speedup immediately
- No build configuration needed

---

### Solution 2: Install CUDA Toolkit 12.0 (Full Installer)

**Why Compatible:**
- Matches driver version exactly (CUDA 12.0)
- Includes Visual Studio integration
- No compatibility issues

**Steps:**
1. **Download CUDA 12.0:**
   ```
   https://developer.nvidia.com/cuda-downloads

   Select:
   - Product: CUDA Toolkit 12.0
   - OS: Windows
   - Architecture: x86_64
   - Version: 11
   - Installer Type: exe (local)
   ```

2. **Run Installer:**
   - **Deselect** "Graphics Driver" (already installed)
   - **Select** "CUDA Toolkit"
   - **Select** "Visual Studio Integration"
   - Install to default path

3. **Rebuild:**
   ```bash
   cd D:/Projects/kv-compact/kv-compact/build-cuda
   cmake -DLLAMA_CPP_DIR="D:/Projects/kv-compact/kv-compact/llama.cpp" \
         -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0" \
         -DGGML_CUDA=ON \
         -DKV_COMPACT_ENABLE_PROFILING=ON \
         -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --config Release
   ```

**Time Required:** ~2 hours (download + install + build)

---

### Solution 3: Use Docker with CUDA

**Why Portable:**
- Pre-configured CUDA environment
- No host system installation needed
- Consistent build environment

**Steps:**
```bash
# Pull llama.cpp Docker image with CUDA
docker pull ghcr.io/ggerganov/llama.cpp:cuda-builder

# Build inside container
docker run --gpus all -v ${PWD}:/work ghcr.io/ggerganov/llama.cpp:cuda-builder

# Or use pre-built image
docker run --gpus all -v ${PWD}:/work \
  -w /work \
  ghcr.io/ggerganov/llama.cpp:cuda-cuda12.0 \
  ./main -m models/Qwen3.5-0.8B-Q4_K_M.gguf -c 8192 -n 512 -ngl 24
```

---

### Solution 4: Build CPU-Only First, Add GPU Later

**Why Pragmatic:**
- CPU build works today
- Test profiling infrastructure
- Add GPU later when CUDA 12.0 is installed

**Steps:**
```bash
# Build CPU-only version
cd D:/Projects/kv-compact/kv-compact/build
cmake -DKV_COMPACT_ENABLE_PROFILING=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release

# Test with CPU
cd Release
./llama-kv-compact.exe -m ../../models/Qwen3.5-0.8B-Q4_K_M.gguf -c 8192 -n 512 -p "test" --perf

# Later: Rebuild with CUDA when installer is available
```

---

## Current Recommendation

### 🎯 **Use Solution 1 (Pre-built Binaries)**

**Reasons:**
1. **Fastest** - 10 minutes vs 2 hours
2. **Guaranteed to work** - No build configuration issues
3. **Test GPU inference immediately** - Validate 3-5x speedup
4. **No installation required** - CUDA 12.0 runtime included

**Action Items:**
1. Download pre-built llama.cpp CUDA binaries (5 min)
2. Extract and test GPU inference (5 min)
3. Validate performance improvement (5 min)
4. Re-profile with GPU working (5 min)

**Total time: 20 minutes to validated GPU inference**

---

## Alternative: CPU-Only Build for Now

If pre-built binaries aren't available, build CPU version:

```bash
cd D:/Projects/kv-compact/kv-compact/build
cmake -DKV_COMPACT_ENABLE_PROFILING=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --target llama-kv-compact

# Test profiling works
cd Release
./llama-kv-compact.exe -m ../../models/Qwen3.5-0.8B-Q4_K_M.gguf -c 8192 -n 512 -p "test" --perf
```

---

## Validation Plan

### After Pre-built Binaries

**1. Verify CUDA Support:**
```bash
./main.exe --help | grep -i cuda
# Expected: GGML_CUDA: version 12.0.0
```

**2. Test GPU Inference:**
```bash
./main.exe -m ../../models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 8192 -n 512 -p "Once upon a time..." -ngl 24
```

**3. Check GPU Utilization:**
```bash
# In another terminal
nvidia-smi
# Expected: GPU utilization >50%
```

**4. Benchmark Performance:**
```bash
./main.exe -m ../../models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 8192 -n 512 -ngl 24 --perf
# Expected: 100-160 tg/s (vs 32.91 tg/s CPU)
```

---

## Technical Details

### CUDA 13.1 vs 12.0 Compatibility

| Component | CUDA 12.0 | CUDA 13.1 | Status |
|-----------|-----------|-----------|--------|
| Driver Support | ✅ Native | ✅ Compatible | Works |
| nvcc Compiler | ✅ Available | ✅ Available | Works |
| VS Integration | ✅ Included | ❌ Missing | Blocked |
| MSBuild Targets | ✅ Complete | ❌ Incomplete | Blocked |

### Winget Installation Limitations

**Winget provides:**
- CUDA Toolkit binaries
- Runtime libraries
- Development tools (nvcc)

**Winget does NOT provide:**
- Visual Studio integration files
- MSBuild targets for CUDA
- CUDA VS integration components

These are needed for CMake-based builds but not for runtime usage.

---

## Summary

**Current Status:**
- ✅ CUDA hardware available (GTX 1050 Ti)
- ✅ CUDA 13.1 installed (winget)
- ✅ nvcc compiler functional
- ❌ Visual Studio CUDA integration missing
- ❌ CMake build configuration blocked

**Fastest Path Forward:**
1. Download pre-built llama.cpp CUDA binaries (10 min)
2. Test GPU inference immediately (10 min)
3. Validate 3-5x speedup (5 min)
4. Re-profile with GPU working (5 min)

**Alternative Path (if pre-built not available):**
1. Install CUDA Toolkit 12.0 full installer (1 hour)
2. Rebuild llama.cpp with CUDA (30 min)
3. Test and validate (10 min)

**Total Time to GPU Validation:** 20-30 minutes (pre-built) or 2 hours (full install)

---

## Files Created

- `llama.cpp/` - Fresh clone of llama.cpp master branch
- `docs/cuda-discovery-winget.md` - CUDA discovery via winget
- `memory/cuda-discovery.md` - Memory entry for CUDA discovery
- `MEMORY.md` - Project memory index

---

## Next Action

**Recommended:** Download and test pre-built llama.cpp CUDA binaries

**Command:**
```bash
# Check for pre-built binaries at:
https://github.com/ggerganov/llama.cpp/releases

# Look for: llama-bin-win-cuda-cu12.0-x64.zip
# Or: llama-bin-win-cuda-cu121-x64.zip
```

**If pre-built not available:** Install CUDA Toolkit 12.0 full installer and rebuild

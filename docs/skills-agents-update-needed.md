# Skills & Agents Update Needed

Based on GPU inference investigation findings, the following skills/agents need updates to reflect GTX 1050 Ti incompatibility with latest llama.cpp CUDA builds.

---

## Skills to Update

### 1. `skills/gpu-optimization.md`
**Current Issues:**
- Assumes pre-built binaries work for all CUDA GPUs
- Doesn't mention compute capability requirements
- Missing llama.cpp CUDA version compatibility matrix

**Updates Needed:**
```markdown
## Hardware Compatibility

### llama.cpp CUDA Support
- **Latest (b8303+):** CUDA 12.4+, requires compute capability 7.0+ (Volta+)
- **b4134 (Nov 2024):** CUDA 12.2, supports compute capability 6.1+ (Pascal+)

### GPU Compatibility Matrix
| Architecture | Min Compute Capability | Min GPU | CUDA Support |
|--------------|----------------------|---------|--------------|
| Pascal | 6.1 | GTX 1050 Ti | CUDA 11.x only |
| Volta | 7.0 | Tesla V100 | CUDA 11.x+ |
| Turing | 7.5 | RTX 2060 | CUDA 11.x+ |
| Ampere | 8.6 | RTX 3060 | CUDA 11.x+ |

### Critical Finding
**GTX 1050 Ti and older Pascal GPUs CANNOT use latest llama.cpp pre-built binaries.**

Solutions:
1. Build from source with CUDA 11.8
2. Use CPU-only mode
3. Upgrade to RTX 3060 Ti or newer
```

---

### 2. `skills/llama-gpu-integration.md`
**Current Issues:**
- Assumes `-ngl 24` works for all GPUs
- Doesn't mention VRAM constraints for different GPU generations
- Missing error handling for compute capability mismatches

**Updates Needed:**
```markdown
## Troubleshooting

### Error: "CUDA error at ggml-cuda.cu:98"
**Cause:** Compute capability mismatch

**Diagnosis:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

**Expected Output:**
```
"NVIDIA GeForce GTX 1050 Ti", "6.1"  ← Too old for CUDA 12.4
"NVIDIA GeForce RTX 3060", "8.6"     ← Compatible
```

**Solution by GPU:**

**Pascal (GTX 1050 Ti, GTX 1080):**
- Build llama.cpp with CUDA 11.8
- OR use CPU-only mode

**Volta+ (RTX 2060+, RTX 3060+):**
- Use pre-built binaries
- No build required

### VRAM Requirements
| GPU VRAM | Recommended -ngl | Model Size |
|----------|-----------------|------------|
| 4GB | 5-10 | 0.8B-3B Q4 |
| 8GB | 20-30 | 3B-7B Q4 |
| 12GB+ | 30+ | 7B+ Q4 |

**GTX 1050 Ti (4GB):** Use `-ngl 5` or lower to avoid OOM
```

---

## Agents to Update

### 1. `agents/cuda-optimization-advisor.md`
**Current Issues:**
- Assumes CUDA Toolkit 12.0+ is always better
- Doesn't check GPU compute capability before recommending
- Missing hardware compatibility checks

**Updates Needed:**
```markdown
## Step 1: Check GPU Compatibility

**CRITICAL:** Always check compute capability first.

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version,cuda_version --format=csv
```

**Decision Tree:**

**If compute_capability < 7.0:**
```
→ GTX 1050 Ti, GTX 1080, older Pascal GPUs
→ SOLUTION: Build llama.cpp with CUDA 11.8
→ NOT RECOMMENDED: Pre-built binaries (won't work)
```

**If compute_capability >= 7.0:**
```
→ RTX 2060+, RTX 3060+, newer GPUs
→ SOLUTION: Use pre-built binaries
→ RECOMMENDED: llama.cpp b8303+ with CUDA 12.4
```

## Step 2: Verify llama.cpp Version

**Qwen 3.5 Support:**
- Requires llama.cpp b8270+ (released Feb 2026+)
- Check model compatibility before choosing version

**Trade-off:**
- Old llama.cpp (b4134): CUDA 12.2, Pascal support, NO Qwen 3.5
- New llama.cpp (b8303): CUDA 12.4, Qwen 3.5 support, NO Pascal support
```

---

### 2. `agents/rocm-specialist.md`
**Current Issues:**
- May recommend AMD GPU alternatives without considering llama.cpp ROCm support
- Missing CUDA fallback guidance

**Updates Needed:**
```markdown
## NVIDIA GPU Recommendations

### Budget-Friendly
- ❌ **AVOID:** GTX 1050 Ti, GTX 1080 (too old for latest llama.cpp)
- ✅ **RECOMMENDED:** RTX 3060 Ti (8GB, $350)
- ✅ **ALTERNATIVE:** RX 7600 (AMD ROCm support)

### Performance-Focused
- ✅ **BEST VALUE:** RTX 4060 Ti (16GB, $450)
- ✅ **PREMIUM:** RTX 4070 (12GB, $600)

### For KV Cache Workloads
Prioritize memory bandwidth over compute:
- RTX 3060 Ti: 360 GB/s
- RTX 4060 Ti: 288 GB/s
- RTX 4070: 504 GB/s

### Compatibility Checklist
Before purchasing:
1. [ ] Compute capability 7.0+ (Volta or newer)
2. [ ] VRAM >= 8GB (for 7B models)
3. [ ] Memory bandwidth > 300 GB/s
4. [ ] ROCm support (if AMD GPU)
```

---

## New Skills to Create

### `skills/gpu-compatibility-check.md`
```markdown
# GPU Compatibility Check for llama.cpp

## Purpose
Check if your GPU is compatible with llama.cpp CUDA builds before attempting setup.

## Quick Check

### Step 1: Identify Your GPU
```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

### Step 2: Check Compute Capability
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

### Step 3: Compare to Matrix

| Compute Capability | Architecture | Compatible | Min llama.cpp | Notes |
|-------------------|--------------|------------|---------------|-------|
| 6.1 | Pascal (GTX 1050 Ti) | ⚠️ PARTIAL | b4134 or custom build | Needs CUDA 11.8 |
| 7.0 | Volta (Tesla V100) | ✅ YES | Any recent | Use pre-built |
| 7.5 | Turing (RTX 2060) | ✅ YES | Any recent | Use pre-built |
| 8.6 | Ampere (RTX 3060) | ✅ YES | Any recent | Best value |
| 8.9 | Ada (RTX 4090) | ✅ YES | Any recent | Premium |

## Action by GPU

### GTX 1050 Ti (6.1)
```
❌ Latest pre-built: NO
✅ Custom build (CUDA 11.8): YES
✅ CPU-only: YES (working alternative)
```

### RTX 3060 (8.6)
```
✅ Latest pre-built: YES
✅ All features: YES
✅ Qwen 3.5: YES
```

## Test Your Setup

### Quick Test (5 minutes)
```bash
# Download pre-built binaries
wget https://github.com/ggerganov/llama.cpp/releases/download/b8303/llama-b8303-bin-win-cuda-12.4-x64.zip

# Extract and test
unzip llama-b8303-bin-win-cuda-12.4-x64.zip
cd llama-b8303-bin-win-cuda-12.4-x64
./llama-cli.exe --version

# Expected output:
# load_backend: loaded CUDA backend  ← GOOD
# ggml-cuda.cu:98: CUDA error         ← BAD (compute capability too low)
```

## Error Resolution

### Error: "CUDA error at ggml-cuda.cu:98"
**Meaning:** Your GPU is too old for this llama.cpp build

**Solution:**
1. Build llama.cpp from source with CUDA 11.8, OR
2. Use CPU-only mode, OR
3. Upgrade GPU to RTX 3060 Ti or newer

## References
- llama.cpp releases: https://github.com/ggerganov/llama.cpp/releases
- CUDA compute capabilities: https://developer.nvidia.com/cuda-gpus
- GPU comparison: https://www.techpowerup.com/gpu-specs/
```

---

## Documentation Updates

### `docs/gpu-setup-guide.md`
Add section:
```markdown
## ⚠️ Hardware Compatibility Warning

**Before starting GPU setup, verify your GPU is compatible:**

### Incompatible GPUs (Pascal architecture, compute capability 6.1)
- GTX 1050 Ti ❌
- GTX 1060 ❌
- GTX 1070 ❌
- GTX 1080 ❌

**For these GPUs:**
1. Build llama.cpp from source with CUDA 11.8, OR
2. Use CPU-only mode (works well with KV compaction)

### Compatible GPUs (Volta architecture and newer, compute capability 7.0+)
- RTX 2060+ ✅
- RTX 3060+ ✅
- RTX 4060+ ✅
- Tesla V100+ ✅

**For these GPUs:**
1. Download pre-built binaries
2. Follow standard setup guide

[Continue with existing setup guide...]
```

---

## Priority Order

1. **HIGHEST:** Update `skills/llama-gpu-integration.md` (directly impacts user workflow)
2. **HIGH:** Create `skills/gpu-compatibility-check.md` (prevents future issues)
3. **MEDIUM:** Update `skills/gpu-optimization.md` (reference material)
4. **MEDIUM:** Update `agents/cuda-optimization-advisor.md` (improves recommendations)
5. **LOW:** Update `agents/rocm-specialist.md` (edge case)

---

## Testing Checklist

After updates, verify:
- [ ] Skills mention compute capability 7.0+ requirement
- [ ] GTX 1050 Ti specifically called out as incompatible
- [ ] Alternative solutions provided (CUDA 11.8 build, CPU-only, upgrade)
- [ ] Links to investigation documentation
- [ ] Decision tree for GPU selection
- [ ] VRAM recommendations per GPU generation

---

**Status:** Update plan created, awaiting implementation
**Date:** 2026-03-13
**Related Documents:**
- `docs/gpu-inference-attempt-summary.md` - Full investigation details
- `docs/action-plan-gpu-inference.md` - Updated action plan with findings

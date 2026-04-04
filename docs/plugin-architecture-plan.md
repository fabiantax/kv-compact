# Plugin Architecture Plan: Tracking Upstream llama.cpp

How to keep using upstream llama.cpp improvements while integrating
kv-compact features, with minimal maintenance burden.

---

## Current State: Already a Plugin

The `optimize-moe-rocm` branch has proven that a **fully external library**
architecture works. The codebase is structured as three independent layers:

```
┌─ kv-compact library (fully standalone, no llama.cpp dependency) ─┐
│                                                                   │
│  kv-compact-math.h      Header-only math (NNLS, LS, scoring)    │
│  kv-compact-api.h/cpp   C API: kv_compact(), quantized, R-KV    │
│  kv-compact-accel.h     GPU interface (conditional HIP/CPU)      │
│  kv-compact-hip.hip     ROCm kernels + rocBLAS (optional)        │
│  kv-compact-state.h     Parses llama.cpp KV state binary format  │
│                                                                   │
│  Input:  float arrays (K, V, Q_ref)                              │
│  Output: selected_indices[], beta[], C_v[]                       │
│  Tests:  test-kv-compact-math, test-kv-compact-api (standalone)  │
└───────────────────────────┬───────────────────────────────────────┘
                            │ state serialization APIs only
┌───────────────────────────┴───────────────────────────────────────┐
│  llama.cpp (STOCK, unmodified)                                    │
│                                                                   │
│  llama_state_seq_get_data()    read KV cache to binary buffer    │
│  llama_state_seq_set_data()    write KV cache from binary buffer │
│  llama_memory_seq_rm()         clear cache positions             │
│  llama_decode()                standard inference                 │
│  llama_model_n_layer/head/embd model metadata queries            │
└───────────────────────────────────────────────────────────────────┘
```

### What's Already Working (optimize-moe-rocm branch)

| Feature | Status | llama.cpp changes? |
|---------|--------|-------------------|
| Full C library API (`kv_compact()`) | Done | No |
| Quantized KV round-trip (Q4_0, Q8_0) | Done | No |
| ROCm GPU scoring (gfx1151 / Strix Halo) | Done | No |
| rocBLAS strided-batched GEMM | Done | No |
| Chunked compaction for 1M+ contexts | Done | No |
| OpenMP parallel per-layer | Done | No |
| Reasoning-aware compression (R-KV) | Done | No |
| Hybrid model layer filters (Qwen 3.5) | Done | No |
| Skip-NNLS mode (6.75x faster) | Done | No |
| Diversity-aware key selection | Done | No |
| Iterative key refinement | Done | No |
| Quality benchmarks (PPL, KL, needle) | Done | No |
| Throughput benchmarks | Done | No |
| Per-head sensitivity budgets | Done | No |
| Multi-round progressive compaction | Done | No |

**Every feature above runs against stock llama.cpp with zero patches.**

---

## The One Remaining Question: Is Beta Injection Worth Patching?

### Evidence That Skipping Beta Works

The `optimize-moe-rocm` branch discovered an empirical result:

> "Skip NNLS beta by default: better quality AND 3-7x faster"

This means the C_v least-squares refit alone (without beta bias injection)
produces quality equal to or better than beta + C_v in many cases. The
explanation: when the LS solver has enough reference queries, it compensates
for missing beta by adjusting C_v values to absorb the mass distribution
error.

### When Beta Injection Still Matters

Beta becomes important at **extreme compression** (>20x) where:
- Very few keys remain (t < 50 for a 60k context)
- The LS solver is underdetermined (fewer keys than query dimensions)
- Mass mismatch causes systematic over/under-attention to cached context

### Decision Framework

```
Compression ≤ 10x  →  Skip beta. LS-only is sufficient. No patch needed.
Compression 10-50x →  Beta helps marginally. Test case-by-case.
Compression > 50x  →  Beta is critical. Patch needed for best quality.
```

For practical home use (Strix Halo, M5 Max, 128GB):
- 60k context at 10x = 6k tokens retained → 12GB → 240MB. Easy.
- 60k context at 50x = 1.2k tokens retained → 12GB → 24MB. Aggressive.

**Most home workloads won't need >10x compression**, making beta injection
a nice-to-have rather than a must-have.

---

## Strategy: Three Layers of Integration

### Layer 1: External Library (current — no llama.cpp changes)

```
User workflow:
  1. llama-server or llama-cli runs stock llama.cpp
  2. kv-compact library compacts via state serialization:
     - llama_state_seq_get_data() → parse → compact → write → llama_state_seq_set_data()
  3. C_v values written back via state serialization
  4. No beta injection (skip-beta mode, which benchmarks show is fine)
```

**When to use:** All normal workloads. 10x compression with excellent quality.

**Maintenance cost:** Zero. Track upstream by bumping FetchContent tag or
submodule pointer.

**Already proven:** The optimize-moe-rocm branch has ~12,800 lines of code
running this way with quality benchmarks, GPU acceleration, quantized KV
support, and 1M+ context handling — all against stock llama.cpp.

---

### Layer 2: Minimal Patch for Beta Injection (if needed)

Only pursue this if benchmarks show beta injection provides meaningful quality
gains at compression ratios you actually use.

llama.cpp's `ggml_flash_attn_ext` already supports an **attention mask**
parameter. The approach:

1. **Store beta as a GGML tensor** alongside K/V in the KV cache.
   Add one tensor per layer to `llama_kv_cache`:
   ```c
   struct ggml_tensor * beta[n_layer];  // [n_kv_max] float32
   ```

2. **Inject beta into the attention mask** (one line in graph builder):
   ```c
   attn_mask = ggml_add(ctx, attn_mask, kv_cache.beta[il]);
   ```

3. **Expose beta via API** (one new function):
   ```c
   void llama_kv_cache_set_bias(llama_context * ctx, int layer,
                                 const float * beta, int n_tokens);
   ```

**Total llama.cpp diff: ~30-50 lines across 2-3 files.**

**Maintenance:**
- Keep as a `.patch` file in `patches/` directory
- Touches narrow, stable surface (KV cache struct + attention graph build)
- Conflicts are rare and mechanical to fix (~5 minutes)

---

### Layer 3: Upstream PR (zero maintenance, best case)

Push beta injection as a **general-purpose feature** to ggml-org/llama.cpp:
"per-position attention biases in KV cache."

Useful beyond kv-compact:
- ALiBi-style positional encoding
- Attention sink weighting
- Any system that needs per-key score adjustments

**If accepted:** kv-compact is fully external forever. Zero patches.
**If rejected:** Fall back to Layer 2 (maintained patch).

---

## Build System: How It Works Today

The CMakeLists.txt already cleanly separates library from tools:

```cmake
# ---- Standalone library (no llama.cpp) ----
add_library(kv-compact-math INTERFACE)           # header-only math
add_library(kv-compact-api STATIC ...)           # C API library

# ---- Optional GPU acceleration ----
if(KV_COMPACT_HIP)
    add_library(kv-compact-hip STATIC ...)       # ROCm kernels
endif()

# ---- Standalone tests (no llama.cpp) ----
add_executable(test-kv-compact-math ...)
add_executable(test-kv-compact-api ...)
add_executable(bench-kv-compact-quality ...)

# ---- Tools requiring llama.cpp (optional) ----
if(KV_COMPACT_BUILD_TOOL)
    # FetchContent or -DLLAMA_CPP_DIR=...
    add_executable(llama-kv-compact ...)          # CLI tool
    add_executable(bench-kv-compact-model ...)    # model benchmarks
    add_executable(test-kv-compact-e2e ...)       # E2E tests
endif()
```

**Three build modes:**

| Mode | Command | Needs llama.cpp? | Use case |
|------|---------|-------------------|----------|
| **Library only** | `cmake .. -DKV_COMPACT_BUILD_TOOL=OFF` | No | Embed in other apps |
| **With local llama.cpp** | `cmake .. -DLLAMA_CPP_DIR=/path` | Yes (your copy) | Development |
| **Auto-fetch** | `cmake ..` | Fetched automatically | Quick start |

---

## Tracking Upstream: Practical Workflow

### Recommended: Git Submodule + Optional Patch

```
kv-compact/
  ├── deps/
  │   └── llama.cpp/          # git submodule → upstream
  ├── patches/
  │   └── llama-kv-beta.patch # only if Layer 2 is needed (~40 lines)
  ├── scripts/
  │   ├── update-deps.sh      # fetch upstream + re-apply patches
  │   └── check-patches.sh    # CI: verify patches still apply cleanly
  ├── include/                # standalone headers
  ├── src/                    # standalone library + tools
  └── CMakeLists.txt
```

**Update workflow:**
```bash
#!/bin/bash
# scripts/update-deps.sh
cd deps/llama.cpp
git fetch origin
git checkout <new-tag-or-master>

# Apply patches if any exist
if [ -d ../../patches ]; then
    for p in ../../patches/*.patch; do
        echo "Applying $p..."
        git apply "$p" || {
            echo "CONFLICT in $p — manual fix needed"
            exit 1
        }
    done
fi

cd ../..
cmake --build build
echo "Updated to $(cd deps/llama.cpp && git describe --tags)"
```

**Frequency:** Update when llama.cpp adds features you want (new model
support, performance improvements, quantization formats). No urgency —
the state serialization API is stable.

---

## Integration Points: Full Inventory

### APIs Used from llama.cpp (all stable, public)

| API | Category | Stability |
|-----|----------|-----------|
| `llama_state_seq_get_data()` | State serialization | High — used by llama-server |
| `llama_state_seq_set_data()` | State serialization | High — used by llama-server |
| `llama_state_seq_get_size()` | State serialization | High |
| `llama_memory_seq_rm()` | Memory management | High |
| `llama_memory_seq_pos_max()` | Memory management | High |
| `llama_get_memory()` | Memory management | High |
| `llama_decode()` | Inference | Very high |
| `llama_batch_init/free()` | Batch management | Very high |
| `llama_model_n_layer/head/embd()` | Model queries | Very high |
| `llama_model_rope_type()` | Model queries | High |
| `common_tokenize()` | Tokenization | High |
| `common_sampler_*()` | Sampling | Medium (API evolving) |
| `common_params_parse()` | CLI parsing | Medium (API evolving) |

**High-risk surface:** Only `common_*` utilities, which are llama.cpp's own
internal helpers. These change more often than the core `llama_*` APIs.

**Mitigation:** The `common_*` functions are only used in the CLI tool
(`kv-compact.cpp`), not in the library. If they break, only the CLI tool
needs updating — the library and all standalone tests remain unaffected.

---

## What Stays External Forever

| Component | Lines | Dependencies | Touches llama.cpp? |
|-----------|-------|--------------|-------------------|
| kv-compact-math.h | ~960 | None | Never |
| kv-compact-api.h/cpp | ~1,870 | None | Never |
| kv-compact-accel.h | ~120 | HIP (optional) | Never |
| kv-compact-hip.hip | ~700 | ROCm + rocBLAS | Never |
| kv-compact-state.h | ~500 | None | Never (parses binary format) |
| Unit tests | ~3,200 | None | Never |
| Quality benchmarks | ~2,600 | None | Never |
| **Total standalone** | **~9,950** | | **Never** |
| | | | |
| kv-compact.cpp (CLI) | ~850 | llama.cpp | Uses stable public APIs |
| Model benchmarks | ~750 | llama.cpp | Uses stable public APIs |
| E2E tests | ~300 | llama.cpp | Uses stable public APIs |
| **Total llama.cpp-dependent** | **~1,900** | | **Public APIs only** |

**~84% of code has zero llama.cpp dependency. 100% runs without patches.**

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| KV state binary format changes | Low | High | Pin to tags, test in CI, format is stable (used by server) |
| `common_*` API changes | Medium | Low | Only affects CLI tool (~850 lines), not library |
| Core `llama_*` API changes | Low | Medium | These APIs are public contracts, rarely break |
| Flash attention mask format changes | Medium | Low | Only matters if using Layer 2 patch |
| New quantization formats added | Medium | None | Add dequant/requant support in kv-compact-api |
| llama.cpp adds native KV compaction | Low | Positive | Can deprecate our tool or contribute upstream |
| llama.cpp drops state serialization | Very low | Very high | Won't happen — llama-server depends on it |
| ROCm/HIP API changes | Low | Medium | Isolated in kv-compact-hip.hip, easy to update |

---

## Speculative Decoding Integration (Future)

The spec-decode-synergy analysis identified three paths. All remain compatible
with the plugin architecture:

| Path | Requires llama.cpp patch? | Why |
|------|--------------------------|-----|
| **A: Compaction-aware verification** | No (Layer 1) | Compact before decode, use state APIs |
| **B: Hierarchical speculation cache** | Maybe | Depends on SSD implementation details |
| **C: Aggressive draft compaction** | No (Layer 1) | Just run kv_compact() on draft model cache |

Path A and C work today with the existing external library approach. Path B
would require understanding how SSD manages its speculation cache, but the
compaction itself remains external.

---

## Summary

```
                      What we have today
                      ──────────────────
                      ┌──────────────────────┐
                      │  kv-compact library   │  9,950 lines
                      │  (standalone)         │  Zero llama.cpp deps
                      │                       │  C API, GPU optional
                      │  ✓ Key selection      │  ✓ Quantized KV
                      │  ✓ NNLS beta          │  ✓ ROCm GPU accel
                      │  ✓ LS value refit     │  ✓ 1M+ contexts
                      │  ✓ Skip-beta mode     │  ✓ R-KV reasoning
                      │  ✓ Multi-round        │  ✓ Hybrid layers
                      └──────────┬───────────┘
                                 │ state serialization (stable API)
                      ┌──────────┴───────────┐
                      │  llama.cpp (stock)    │  Unmodified
                      │  FetchContent or      │  Submodule
                      │  local checkout       │  Pin to tags
                      └──────────────────────┘

                      What we might add later
                      ──────────────────────
                      patches/llama-kv-beta.patch  (~40 lines)
                      Only if >20x compression needs beta injection.
                      Evidence so far: skip-beta works well enough.
```

**Bottom line:** kv-compact is already a plugin. The `optimize-moe-rocm`
branch proved this with 12,800 lines of working code against stock llama.cpp.
Keep it this way. Only patch llama.cpp if extreme compression benchmarks
prove beta injection is essential — and even then, it's ~40 lines.

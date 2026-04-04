# Plugin Architecture Plan: Tracking Upstream llama.cpp

How to keep using upstream llama.cpp improvements while integrating
kv-compact features, with minimal maintenance burden.

---

## Current State: What Works Today

The current architecture is already **zero-touch** on llama.cpp:

```
kv-compact (external tool)
    ├── kv-compact-math.h    (standalone, no deps)
    ├── kv-compact-state.h   (standalone, parses binary format)
    └── kv-compact.cpp       (links against llama.cpp as library)
         uses: llama_state_seq_get_data() → compact → llama_state_seq_set_data()
```

This works for: key selection, NNLS beta computation, C_v value refitting,
and writing compacted state back.

**What breaks this clean boundary:** Beta injection during `llama_decode()`.
The attention computation needs `score_ij = q @ k / sqrt(d) + beta_j` but
llama.cpp has no hook for per-position attention biases in its KV cache.

---

## The Problem: Where We Must Touch llama.cpp

| Feature | Requires llama.cpp change? | Why |
|---------|---------------------------|-----|
| Key selection | No | External math on serialized state |
| Beta computation | No | External NNLS solver |
| C_v refitting | No | External least squares |
| C_v write-back | **Borderline** | `llama_state_seq_set_data` works but is coarse |
| **Beta injection** | **Yes** | Must modify attention score computation |
| **Incremental compaction** | **Maybe** | Need hook after each decode step |
| **Spec decode integration** | **Maybe** | Need to intercept verify step |

Only **beta injection** is a hard requirement for touching llama.cpp internals.
Everything else can be done externally or through existing APIs.

---

## Strategy: Three Layers of Integration

### Layer 1: External Tool (current — no llama.cpp changes)

```
User workflow:
  1. llama-server runs stock llama.cpp
  2. kv-compact runs as sidecar, periodically:
     - saves KV state via API
     - compacts externally
     - writes compacted state back
  3. No beta injection (accept quality loss)
```

**When to use:** Quick experiments, proving the concept, benchmarking
compression ratios vs. quality without beta.

**Maintenance cost:** Zero. Track upstream by bumping FetchContent tag.

---

### Layer 2: GGML Custom Op / Attention Mask Injection (minimal patch)

llama.cpp's `ggml_flash_attn_ext` already supports an **attention mask**
parameter. The idea:

```
Instead of modifying attention computation code, inject beta as a
KV-cache-aligned bias tensor that gets added to attention scores
via the existing mask pathway.
```

**How it works:**

1. **Store beta as a GGML tensor** alongside K/V in the KV cache.
   This requires adding one tensor per layer to `llama_kv_cache`:
   ```c
   // In llama_kv_cache (llama.cpp)
   struct ggml_tensor * beta[n_layer];  // [n_kv_max] float32
   ```

2. **Inject beta into the attention mask.** `ggml_flash_attn_ext` takes a
   mask tensor. Modify the graph builder to add beta to the mask:
   ```c
   // In llama_build_graph (one line change)
   attn_mask = ggml_add(ctx, attn_mask, kv_cache.beta[il]);
   ```

3. **Expose beta via API.** Add one new function:
   ```c
   void llama_kv_cache_set_bias(llama_context * ctx, int layer,
                                 const float * beta, int n_tokens);
   ```

**Total llama.cpp diff: ~30-50 lines across 2-3 files.**

**Maintenance strategy:**
- Keep the patch as a `.patch` file in this repo
- On upstream update: `git fetch upstream && git rebase` or re-apply patch
- The patch touches a narrow, stable surface (KV cache struct + graph build)
- If it conflicts, the fix is mechanical (same 3 lines in new locations)

---

### Layer 3: Upstream PR (zero maintenance, best case)

Push beta injection as a **general-purpose feature** to upstream llama.cpp:
"per-position attention biases in KV cache." This is useful beyond kv-compact:
- ALiBi-style positional encoding
- Attention sink weighting
- Any system that needs per-key score adjustments

**If accepted:** kv-compact becomes fully external again. Zero patches.
**If rejected:** Fall back to Layer 2 (maintained patch).

---

## Recommended Implementation Plan

### Phase 1: Prove Value Without Patching (now)

Use Layer 1. Demonstrate compaction quality and tg/s improvements using
only existing APIs. Build the full external pipeline:

```
cmake .. -DLLAMA_CPP_DIR=/path/to/stock/llama.cpp
```

Deliverables:
- [ ] Full all-layer/all-head compaction via state serialization
- [ ] Quality benchmarks (perplexity, KL, needle-in-haystack)
- [ ] tg/s benchmarks at various compression ratios
- [ ] Comparison: with vs. without C_v write-back (both work today)

### Phase 2: Minimal Patch for Beta Injection

Create a small, rebasing-friendly patch:

```
patches/
  llama-kv-beta.patch       # The actual diff (~40 lines)
  apply-patches.sh           # git apply + verify script
  README-patches.md          # What each patch does, how to resolve conflicts
```

Build script becomes:
```bash
#!/bin/bash
# build-with-patches.sh
git clone https://github.com/ggml-org/llama.cpp.git --depth 1
cd llama.cpp
git apply ../patches/llama-kv-beta.patch
cd ..
cmake .. -DLLAMA_CPP_DIR=./llama.cpp
cmake --build .
```

**Conflict resolution playbook:**
- llama.cpp releases roughly weekly
- Test patch application in CI against `master` nightly
- When patch fails: fix takes 5-10 minutes (it's ~40 lines in stable code)
- Track the specific functions/structs we touch in `README-patches.md`

### Phase 3: Upstream Contribution

Once beta injection is proven valuable with benchmarks:
- Open PR to ggml-org/llama.cpp proposing per-position KV attention biases
- Include benchmark data from Phase 1+2 showing quality + speed gains
- Frame as general infrastructure, not kv-compact-specific

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│                  kv-compact                      │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ kv-compact-  │  │ kv-compact-state.h       │ │
│  │ math.h       │  │ (parse/write KV binary)  │ │
│  │ (standalone) │  │ (standalone)             │ │
│  └──────┬───────┘  └──────────┬───────────────┘ │
│         │                     │                  │
│  ┌──────┴─────────────────────┴───────────────┐ │
│  │ kv-compact.cpp (or libkv-compact.so)       │ │
│  │                                            │ │
│  │  compact() → selection + NNLS + LS         │ │
│  │  inject()  → llama_kv_cache_set_bias()  ←──┼─┼── only API needing patch
│  └────────────────────┬───────────────────────┘ │
│                       │                          │
└───────────────────────┼──────────────────────────┘
                        │ links against
┌───────────────────────┼──────────────────────────┐
│  llama.cpp (stock + optional ~40-line patch)     │
│                       │                          │
│  llama_state_seq_get_data()    (existing API)    │
│  llama_state_seq_set_data()    (existing API)    │
│  llama_memory_seq_rm()         (existing API)    │
│  llama_kv_cache_set_bias()     (NEW, from patch) │
│  ggml_flash_attn_ext(mask+β)   (patched, 1 line) │
└──────────────────────────────────────────────────┘
```

---

## Tracking Upstream: Practical Workflow

### Option A: Git Submodule + Patch (recommended)

```
kv-compact/
  ├── llama.cpp/              # git submodule pointing to upstream
  ├── patches/
  │   └── llama-kv-beta.patch
  ├── scripts/
  │   ├── update-llama.sh     # fetch upstream + re-apply patches
  │   └── check-patch.sh      # CI: verify patch still applies
  └── CMakeLists.txt          # -DLLAMA_CPP_DIR=./llama.cpp
```

Update workflow:
```bash
cd llama.cpp
git fetch origin
git checkout <new-tag>
git apply ../patches/llama-kv-beta.patch
# If conflict: manually fix ~40 lines, regenerate patch
cd .. && cmake --build build
```

### Option B: Fork with Rebase Branch

Maintain a fork with a single `kv-compact-patches` branch:
```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream
git rebase upstream/master   # rebase our 1-2 commits on top
```

**Downside:** fork divergence is harder to reason about than a .patch file.

### Option C: Runtime Injection (LD_PRELOAD / dylib)

Override the attention function at runtime without modifying source:
```bash
LD_PRELOAD=libkv-compact-attn.so llama-server ...
```

**Downside:** Fragile, ABI-dependent, breaks on any internal refactor.
Not recommended for ongoing use.

**Recommendation: Option A** (submodule + patch). It's explicit, debuggable,
and the patch is small enough to maintain manually.

---

## What Stays External Forever

These components will **never** need to touch llama.cpp:

| Component | Why it's safe |
|-----------|--------------|
| Math library | Pure CPU float32, zero dependencies |
| State parser | Parses documented binary format |
| Key selection | External algorithm on parsed data |
| NNLS solver | External optimization |
| LS value fitting | External optimization |
| Quality benchmarks | Uses standard llama.cpp decode API |
| Compaction scheduling | External orchestration |

**>90% of kv-compact code is and will remain fully external.**

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| llama.cpp changes KV state binary format | Low (stable API) | High | Pin to known-good tags, test in CI |
| Flash attention API changes mask format | Medium | Medium | Patch is 1 line, easy to update |
| KV cache struct changes | Medium | Low | Patch touches 1 field addition |
| Beta injection PR accepted upstream | Hopeful | Eliminates all patches | Best case scenario |
| llama.cpp drops state serialization API | Very low | Very high | This API is used by llama-server; won't be dropped |

---

## Summary

1. **Today:** Fully external, zero patches. Good enough for compaction without
   beta injection.
2. **Soon:** ~40-line patch for beta injection. Maintain as `.patch` file in
   a `patches/` directory. Rebase-friendly, narrow surface.
3. **Goal:** Upstream PR to make beta injection a first-class llama.cpp
   feature. Then kv-compact goes back to fully external.
4. **>90% of code never touches llama.cpp** regardless of strategy.

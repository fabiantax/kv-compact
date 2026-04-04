# Session Handover — 2026-04-04

Branch: `claude/spec-decode-research-hk3d4`
Base: `master`
Commits: 4 (bc99437 → f82f10e)

---

## What Was Done

### 1. Speculative Decoding Research (`docs/spec-decode-synergy.md`)

Analyzed two sources on speculative decoding:
- **Gimlet Labs Corsair blog:** Heterogeneous inference with SRAM accelerator
  (150 TB/s) for draft model. 2-10x speedup over GPU-only spec decode.
- **arXiv 2603.03251 "Speculative Speculative Decoding"** (ICLR 2026):
  Saguaro algorithm parallelizes drafting and verification via speculation
  cache with geometric fan-out. 30% faster than optimized SD.

**Key finding:** KV compaction and speculative decoding are *multipliers*,
not alternatives. Compaction frees memory → enables spec decode → both
improve tg/s. Five concrete synergies documented, three implementation
paths prioritized.

### 2. Plugin Architecture Plan (`docs/plugin-architecture-plan.md`)

Investigated how to track upstream llama.cpp while using our improvements.

**Key finding from `optimize-moe-rocm` branch:** kv-compact is already a
fully external plugin. 12,800 lines of working code (C API, ROCm GPU accel,
quantized KV, 1M+ context chunking) all run against **stock unmodified
llama.cpp**. Only beta injection would ever need a patch (~40 lines), and
the branch also discovered that skipping beta gives equal/better quality
while being 3-7x faster.

**Strategy:** Three layers — (1) external library today, (2) ~40-line patch
only if extreme compression proves it necessary, (3) upstream PR to
eliminate patches entirely. Recommended workflow: git submodule + .patch.

### 3. CSR Bridge Detection Prototype (code)

Implemented spectral bridge detection for key selection using TRIZ and
Axiomatic Design cross-domain thinking:

**New code in `include/kv-compact-math.h` (~250 lines):**
- `csr_matrix` struct + `csr_from_threshold()` — sparse attention graph
- `csr_transpose()` — CSR ↔ CSC conversion
- `csr_spmv()` — sparse matrix-vector multiply
- `fiedler_vector()` — 2nd eigenvector of graph Laplacian via power iteration
- `bridge_scores_from_fiedler()` — structural importance scoring
- `select_keys_bridge_aware()` — combined attention + bridge selection
- `compact_head_bridge_aware()` — full bridge-aware compaction pipeline

**Tests:** 8 new tests (30 total), all passing. Includes cluster separation
test (Fiedler separation=0.7071), bridge identification, quality regression.

**Benchmark (`tests/bench-bridge-selection.cpp`):** Compares standard vs
bridge-aware across random, bridge-heavy, and extreme compression scenarios.

**Honest results:**
- 1.4-1.6x faster than standard (because it uses skip-beta mode)
- Quality identical — C_v least-squares refit compensates for key selection
  differences. Bridge detection neither helped nor hurt.
- CSR achieves 90% sparsity consistently, construction <5ms at T=4096
- **Assessment: ~20-30% chance of practical quality improvement.** The CSR
  infrastructure is valuable regardless. Bridge detection needs validation
  on real model attention patterns to prove its worth.

---

## Branch State

```
docs/spec-decode-synergy.md        NEW   234 lines  Research analysis
docs/plugin-architecture-plan.md   NEW   301 lines  Architecture plan
include/kv-compact-math.h          MOD   +250 lines CSR + Fiedler + bridge-aware compaction
tests/test-kv-compact-math.cpp     MOD   +170 lines 8 new tests (30 total, all pass)
tests/bench-bridge-selection.cpp   NEW   270 lines  Benchmark comparing selection methods
CMakeLists.txt                     MOD   +3 lines   bench-bridge-selection target
```

Build: `cmake .. -DKV_COMPACT_BUILD_TOOL=OFF && cmake --build .`
Tests: `./test-kv-compact-math` (30/30 pass)
Bench: `./bench-bridge-selection`

---

## Key Decisions & Context

### Hardware Context
User has:
- **ASUS ROG Strix Halo 395** — 128GB unified memory, 256 GB/s, RDNA 3.5 iGPU
  (primary, home, long runs)
- **MacBook M5 Max** — 128GB unified memory, 546 GB/s (work laptop, dev/testing)

### Model Targets
- **Qwen 3.5** family: 72B Q6 target + 3B Q8 draft. Confirmed compatible.
- **Gemma 4:** Too new for llama.cpp GGUF support (as of 2026-04-04).

### Technical Insights Discovered
1. **Beta can be skipped** — C_v value refit has ~d_v× more degrees of freedom
   than beta (t×d_v vs t parameters). At moderate compression, C_v absorbs
   the mass error. Beta only matters at extreme compression (>20x) where
   C_v becomes underdetermined.
2. **kv-compact is already a plugin** — the optimize-moe-rocm branch proved
   full external operation with 12.8k lines against stock llama.cpp.
3. **Bridge detection is theoretically sound but practically unproven** — the
   Fiedler vector correctly identifies cluster boundaries and bridge keys,
   but C_v compensation masks any quality difference in benchmarks.

---

## Existing Branches (for context)

| Branch | Date | Lines Changed | Status |
|--------|------|--------------|--------|
| `optimize-moe-rocm` | Mar 10-14 | +12,880 | Most mature — ROCm, quantized KV, benchmarks |
| `arxiv-mcp-integration` | Mar 10-15 | +21,661 | Most recent — MoE expert caching, Vulkan profiling |
| `list-branch-changes` | Mar 12-13 | +21,826 | Profiling, closed-form beta, batch prefill |
| `review-recent-changes` | Mar 11 | +1,754 | Docs, bug fixes |
| `spec-decode-research` (this) | Apr 4 | +889 | Research + bridge detection prototype |

`optimize-moe-rocm` and `arxiv-mcp-integration` are the big feature branches.
They may overlap significantly. Relationship between them is unclear — a
future session should investigate whether they can be merged or if one
supersedes the other.

---

## Prioritized Next Steps

1. **Merge `optimize-moe-rocm` to master** — it has the validated pipeline
2. **Run on Strix Halo with Qwen 3.5** — first real tg/s numbers
3. **Test spec decode + compaction together** — measure combined improvement
4. **Test aggressive draft compaction (100x)** — measure acceptance rate
5. **Needle-in-a-haystack at extreme compression** — validates bridge detection
6. **Upstream PR for per-position KV bias** — only if extreme compression benchmarks warrant it

---

## Files to Read First (Next Session)

- `docs/plugin-architecture-plan.md` — integration strategy and full API inventory
- `docs/spec-decode-synergy.md` — how compaction + spec decode compose
- `docs/rationale.md` — why the algorithm works, improvement opportunities
- `docs/adjacent-concepts.md` — cross-domain techniques mapped to the pipeline
- `include/kv-compact-math.h` — all math primitives including new CSR/bridge code

# Handover: kv-compact

**Date:** 2026-03-11
**Branch:** `claude/arxiv-mcp-integration-t956A`
**Last commit:** `3a74c9a` — Add CLAUDE.md with Qwen 3.5 architecture reference

---

## What is kv-compact?

A C++ implementation of "Fast KV Compaction via Attention Matching" (arXiv:2602.16284, Zweiger et al., MIT, Feb 2026). Achieves **50x KV cache compression** with minimal quality loss using a closed-form 3-step algorithm:

1. **Key Selection** — Select top-t keys by cumulative attention score
2. **NNLS Beta Fitting** — Solve for attention mass biases (beta) to match original attention distribution
3. **Least Squares Value Refitting** — Compute optimal compacted values (C_v) via ridge regression

The key insight: it works in continuous latent space, not token space. Compacted values can be weighted combinations of many original values.

---

## Project Structure

```
include/
  kv-compact-math.h      — Header-only math library (1497 lines, zero deps)
  kv-compact-adapter.h   — Attention-type abstraction (GQA, MLA, hybrid)
  kv-compact-state.h     — llama.cpp binary state parser/writer
src/
  kv-compact.cpp          — CLI tool (requires llama.cpp)
tests/
  test-kv-compact-math.cpp     — 22+ math unit tests
  test-kv-compact-adapter.cpp  — 20+ adapter tests
  test-kv-compact-e2e.cpp      — End-to-end integration tests
  bench-synthetic.cpp          — Synthetic benchmarks
docs/                          — 7 design/reference docs
patches/
  attn-bias.patch              — llama.cpp patch for attention bias injection
```

---

## What's Done

### Core Algorithm (100% of paper's convex steps)
- Steps 1-3 fully implemented
- 4 key selection modes: MAX_ATTN, SUBMODULAR, TOKEN_MERGE, KMEANS
- 2 beta fitting modes: NNLS, SINKHORN
- Ridge regression value refitting via Cholesky decomposition
- Head sensitivity weighting + Caratheodory budget allocation
- Alternating minimization for joint beta/C_v optimization

### Infrastructure
- **69+ unit tests** — all passing
- **Adapter abstraction** — GQA adapter (identity), MLA adapter (latent projection + LSQ recovery), noop adapter, hybrid layer classifier
- **State I/O** — Parse/write all GGML quant types (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
- **Attention bias injection** — llama.cpp patch applied, beta injected via attention mask
- **Synthetic benchmarks** — Tested up to 50x compression at T=4096
- **CLI tool** — Full pipeline: load model → prefill → compact → write back → generate → compare

### Documentation
- Full paper breakdown (`docs/attention-matching-paper.md`)
- Algorithm reference (`docs/algorithms.md`)
- 24 adjacent compression techniques surveyed (`docs/adjacent-concepts.md`)
- Implementation status matrix (`docs/improvement-tracker.md`)
- 5-phase streaming roadmap (`plan.md`)
- Adapter state machine diagrams (`docs/adapter-state-machine.md`)
- Beta injection data flow (`docs/attn-bias-flow.md`)

---

## What's NOT Done

### High Priority (from paper)
1. **Greedy budget exchange (§5)** — Paper's most impactful ablation. Per-model precomputed head budgets via calibration data
2. **Online compaction (§7)** — Compress-during-generation for reasoning/agentic workloads. Requires inference loop hooks
3. **Reference query generation (§4)** — Repeat-prefill for high-quality Q_ref. Currently using caller-provided Q_ref
4. **OMP key selection (§3 Step 1B)** — Better keys but 100-500x slower. OMP-fast variant (§8) as compromise

### High Priority (from roadmap — plan.md)
5. **Streaming compaction (Phase 1)** — Incremental chunk-based compaction for 200K contexts. The `streaming_compactor` class is designed but not implemented
6. **Token pinning (Phase 3)** — Protect system prompt, tool boundaries, recent window from compaction
7. **Cumulative error control (Phase 4)** — Error monitoring + adaptive trigger + re-anchoring for multi-round compaction
8. **Qwen3.5-0.8B integration (Phase 5)** — Hybrid layer detection, GQA-aware sensitivity, E2E validation

### Medium Priority (from adjacent concepts)
9. **CSSP & leverage scores** — Principled replacement for top-t key scoring
10. **CUR decomposition** — Joint key+value factorization
11. **Frank-Wolfe** — Sparse beta fitting with convergence guarantees

---

## Qwen 3.5 Architecture (Critical Context)

Qwen 3.5 does **NOT** use GQA or MLA. It uses a **hybrid Gated DeltaNet + full attention** architecture:
- **3 out of 4 layers**: Gated DeltaNet (linear attention, recurrent hidden state — no KV cache)
- **Every 4th layer**: Standard full attention (has normal KV cache)
- Qwen3.5-0.8B: 24 layers total, only 6 have full attention (layers 3,7,11,15,19,23)

**Implications:**
- Only full-attention layers (6/24) need compaction — the rest have no KV cache to compact
- The `hybrid_classifier` in `kv-compact-adapter.h` handles this: it classifies layers as FULL_ATTENTION or RECURRENT
- The `noop_adapter` passes through recurrent layers unchanged
- With only 2 KV heads (GQA ratio=4), per-head budget is trivial but GQA-aware sensitivity matters

**Models:** 0.8B, 2B, 4B, 9B, 27B (dense) + 35B-A3B, 122B-A10B, 397B-A17B (MoE)
**Unsloth:** Day-zero GGUF quants. QLoRA not recommended. Custom Mamba Triton kernels.

---

## Build & Test

```bash
# Test-only build (no model needed, no llama.cpp)
mkdir build && cd build
cmake .. -DKV_COMPACT_BUILD_TOOL=OFF
cmake --build .
./test-kv-compact-math          # 60+ tests
./test-kv-compact-adapter       # 20+ tests

# Full build with llama.cpp (auto-fetched)
cmake ..
cmake --build .
./llama-kv-compact -m model.gguf -p "context" --compact-ratio 0.2
```

---

## Key Design Decisions

1. **Header-only math library** — Zero dependencies, testable independently, portable C++17
2. **Adapter pattern** — Open/closed principle: add new attention types (MLA, hybrid) without modifying core math
3. **State-based I/O** — Uses llama.cpp `llama_state_seq_get/set_data` for zero-copy KV access
4. **Scalar-only** — No SIMD yet. Profiling shows NNLS and LSQ are fast enough for chunk sizes up to 16K
5. **Patch-based llama.cpp integration** — `attn-bias.patch` adds `llama_memory_set_attn_bias()` API

---

## Recommended Next Session Focus

**Option A: Streaming compaction (Phase 1)**
Highest impact for the target use case (200K-context agents). The design is in `plan.md` Steps 1.1-1.3. Implement `streaming_compactor` class, chunk algorithm, and streaming tests.

**Option B: Greedy budget exchange (§5)**
Paper's most impactful ablation. Requires calibration data to precompute per-head budgets. Could improve quality significantly at extreme compression ratios.

**Option C: Qwen3.5-0.8B E2E validation (Phase 5)**
Run the full pipeline on a real model. Download Qwen3.5-0.8B-Q4_K_M.gguf, test at 4x/16x/50x compression. This validates everything built so far and surfaces integration issues early.

---

## Files to Read First

1. `include/kv-compact-math.h` — The core algorithm
2. `include/kv-compact-adapter.h` — Attention type abstraction
3. `plan.md` — 5-phase streaming roadmap
4. `docs/improvement-tracker.md` — What's done and what's next
5. `CLAUDE.md` — Qwen 3.5 architecture notes

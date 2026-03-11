# User Stories: KV Cache Compaction via Attention Matching

## Overview

These user stories describe the incremental path from the current POC to a
production-ready KV cache compaction feature in llama.cpp, based on the
"Fast KV Compaction via Attention Matching" paper (Zweiger et al., 2026).

### Status Legend
- **DONE** — Implemented and tested
- **PARTIAL** — Core functionality exists, needs integration or polish
- **TODO** — Not yet started

---

## Epic 1: Core Compaction Integration

### US-1: Inject attention biases (beta) into generation — DONE

**As a** developer integrating KV compaction into the inference pipeline,
**I want** the compacted cache's beta biases to be applied during attention
computation at generation time,
**so that** the compacted keys correctly approximate the original attention mass
distribution.

**Acceptance criteria:**
- Beta values are stored alongside compacted KV entries (per layer, per head)
- During `llama_decode`, attention scores for compacted positions have beta
  added before softmax: `score_ij = q_i @ k_j / sqrt(d) + beta_j`
- Generation output with beta injection matches or improves upon token-eviction
  baseline quality on a reference prompt
- No regression in inference speed for non-compacted contexts (beta = 0 path)

**Implementation:** `attn-bias.patch` adds `llama_memory_set_attn_bias()` API.
Beta injected via attention mask. See `docs/attn-bias-flow.md`.

---

### US-2: Write refitted values (C_v) back into the KV cache — DONE

**As a** developer integrating KV compaction into the inference pipeline,
**I want** the least-squares-optimized values (C_v) to replace the original
values in the KV cache after compaction,
**so that** the attention output with the compacted cache closely approximates
the original uncompressed output.

**Acceptance criteria:**
- After compaction, C_v values are written to the V tensor for selected
  positions via `llama_state_seq_set_data` or direct tensor writes
- Supports F32, F16, and quantized KV cache types
- Round-trip test: compact, write C_v, read back, verify values match
- Quality test: MSE of attention output with C_v < MSE with original V

**Implementation:** `build_compacted_state()` in `kv-compact-state.h`. All
GGML quant types supported (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1).

---

### US-3: Compact all layers and heads (not just layer 0) — DONE

**As a** user running compaction on a real model,
**I want** the algorithm to compact every layer and every KV head independently,
**so that** the full model benefits from cache reduction rather than just a
single demo layer.

**Acceptance criteria:**
- Compaction loop iterates over all `n_layer` layers and `n_head_kv` heads
- Each head gets its own selected indices, beta, and C_v
- Per-head compaction is independent (no cross-head data dependencies)
- Progress reporting shows layer/head progress
- Total wall-clock time is reported

**Implementation:** `compact_layer_all_heads()` in `kv-compact-math.h`.
CLI orchestration in `kv-compact.cpp`.

---

## Epic 2: Reference Query Generation

### US-4: Implement true repeat-prefill for reference query extraction — TODO

**As a** developer improving compaction quality,
**I want** to generate reference queries by running a "repeat-prefill" pass
(feeding the context twice as described in the paper),
**so that** the reference queries reflect actual model behavior rather than
using K vectors as a proxy.

**Acceptance criteria:**
- After initial prefill, a second pass processes the same context
- Query activations are captured from the second pass for each layer/head
- Configurable number of reference queries (`--n-ref-queries`)
- Quality comparison: repeat-prefill queries vs. K-vector proxy, measured by
  MSE of compacted attention output vs. original

---

## Epic 3: Advanced Key Selection

### US-5: Support per-head non-uniform compression budgets — DONE

**As a** user seeking optimal quality at a given compression ratio,
**I want** the compaction algorithm to allocate more budget (keep more keys)
for attention heads that are more sensitive to compression,
**so that** quality-critical heads retain more information while less important
heads are compressed more aggressively.

**Acceptance criteria:**
- Sensitivity metric computed per head
- Budget allocation redistributes the total `t` budget across heads
- Total tokens kept across all heads equals the global target
- Quality improves over uniform budget allocation on at least one benchmark

**Implementation:** `compute_head_sensitivity()` in `kv-compact-math.h`.
Caratheodory-informed budgets also implemented.

---

### US-6: Implement Orthogonal Matching Pursuit (OMP) key selection — TODO

**As a** researcher comparing compaction strategies,
**I want** an OMP-based key selection method as an alternative to "Highest
Attention Keys",
**so that** I can evaluate the quality/speed tradeoff described in the paper
(OMP is slower but can yield better key subsets).

**Acceptance criteria:**
- OMP iteratively selects keys that maximize residual reduction
- Selectable via `--key-selection omp` vs `--key-selection highest-attn`
- OMP produces equal or better quality than Highest Attention at same budget
- Wall-clock time reported for comparison

---

## Epic 4: Production Readiness

### US-7: Support quantized KV cache types — DONE

**As a** user running models with quantized KV caches (Q8_0, Q4_0, etc.),
**I want** compaction to work with quantized K and V tensors,
**so that** I can benefit from both quantization and compaction simultaneously.

**Acceptance criteria:**
- K/V data is dequantized to F32 for compaction math
- Compacted C_v is re-quantized to match the original KV type before writing
- Round-trip quantization error is measured and reported
- Quality degrades gracefully compared to F32/F16 compaction

**Implementation:** `kv-compact-state.h` handles all GGML quant types.
Round-trip tests in `test-kv-compact-math.cpp`.

---

### US-8: Expose compaction as a library API (not just a CLI tool) — TODO

**As a** developer building applications with llama.cpp,
**I want** a C API for KV cache compaction that I can call programmatically,
**so that** I can trigger compaction at runtime when context grows too large
without spawning a separate tool.

**Acceptance criteria:**
- New API functions in `llama.h`:
  - `llama_kv_compact(ctx, seq_id, target_ratio, params)` — compact a sequence
  - `llama_kv_compact_params_default()` — sensible defaults
- Thread-safe: compaction can run while other sequences are being decoded
- Returns compaction statistics (tokens before/after, quality metrics)
- Documented in header with usage example

---

### US-9: Iterative (multi-round) compaction support — TODO

**As a** user with very long conversations,
**I want** to apply compaction multiple times as the context grows,
**so that** the cache stays within budget over extended interactions without
catastrophic quality loss.

**Acceptance criteria:**
- Compaction can be applied to an already-compacted cache
- Quality after N rounds of compaction is measured and reported
- Paper claims 6 consecutive compressions on AIME maintain quality —
  verify this with at least 3 rounds on a reference task
- Beta values from previous compactions are preserved or re-optimized

---

## Epic 5: Benchmarking & Validation

### US-10: Automated quality benchmarks — TODO

**As a** developer validating compaction quality,
**I want** automated benchmark scripts that compare compacted vs. uncompressed
generation across standard tasks,
**so that** quality regressions are caught before merging changes.

**Acceptance criteria:**
- Benchmark script runs perplexity evaluation with and without compaction
- Reports: perplexity delta, token-level agreement rate, cosine similarity
- Tests at multiple compression ratios (20%, 50%, 80% retention)
- Runs on at least one small model (e.g., 1B parameter) in CI

---

### US-11: Memory and latency profiling — PARTIAL

**As a** user evaluating whether compaction is worthwhile for my use case,
**I want** the tool to report memory savings and compaction latency,
**so that** I can make informed decisions about the memory/quality/speed
tradeoff.

**Acceptance criteria:**
- Reports peak memory before and after compaction
- Reports wall-clock time for each compaction phase (key selection, NNLS, LS)
- Reports amortized cost: compaction time vs. time saved from smaller cache
  during subsequent generation
- Output format is machine-parseable (JSON option)

**Implementation:** Synthetic benchmarks exist (`bench-synthetic.cpp`) but
no JSON output or memory tracking yet.

---

## Epic 6: Streaming Compaction for Long Contexts

### US-12: Implement streaming compactor for incremental compaction — DONE

**As a** developer building agentic applications with 200K+ token contexts,
**I want** incremental chunk-based compaction that triggers automatically when
the cache exceeds a threshold,
**so that** the KV cache stays within memory budget throughout a long session
without requiring a single expensive compaction pass.

**Acceptance criteria:**
- `streaming_compactor` class manages compaction state across rounds ✅
- Compaction triggers when cache size exceeds configurable threshold ✅
- Cache is split into zones: pinned prefix, compactable middle, recent window ✅
- Compactable zone is compressed to target budget ✅
- Total overhead for a 200K-token session is <2.5s (25 rounds x 100ms) ⏳ (18.5ms per round measured)
- Quality after 25 rounds: MSE < 10x single-round MSE ⏳ (cumulative error test TODO)

**Implementation:** `streaming_config`, `streaming_head_state`, `streaming_compactor`
in `kv-compact-math.h`. CLI flags: `--pin-prefix`, `--recent-window`, `--trigger`,
`--budget`. 4 unit tests passing. Single-shot compaction: 5.2× in 18.5ms, cos_sim 0.9999+.

**Reference:** `plan.md` Phase 1 (Steps 1.1-1.3)

---

### US-13: Token pinning for system prompts and tool boundaries — PARTIAL

**As a** developer building tool-use agents,
**I want** certain tokens (system prompt, tool call boundaries, BOS) to be
protected from compaction,
**so that** critical context survives compression and the agent maintains
coherent tool-use behavior.

**Acceptance criteria:**
- Pin mask (`bool[]`) passed to compaction config ⏳
- Pinned tokens pass through unchanged — never selected for removal ✅ (zones work)
- Pinned tokens still contribute to attention computation ✅
- CLI flags: `--pin-prefix N`, `--pin-tokens <token_ids>` ✅ (`--pin-prefix` done)
- Verify: pinned tokens survive N rounds of streaming compaction unchanged ⏳

**Implementation:** Zone architecture (pin | compactable | recent) implemented.
Prefix and recent window protection via CLI flags work. Fine-grained token
pin mask API pending.

**Reference:** `plan.md` Phase 3

---

### US-14: Cumulative error monitoring and re-anchoring — TODO

**As a** developer ensuring multi-round compaction doesn't degrade catastrophically,
**I want** error metrics tracked across compaction rounds with an automatic
re-anchoring mechanism to correct accumulated drift,
**so that** quality remains bounded even after many compression rounds.

**Acceptance criteria:**
- After each round: MSE and beta mass error computed and accumulated
- If cumulative error exceeds threshold: increase budget or skip compaction
- Every K rounds: recompute beta from scratch (re-anchor) to correct drift
- Cumulative error benchmark: measure error after 25 rounds at 50x compression
- Adaptive trigger tested: verify budget increase when error spikes

**Reference:** `plan.md` Phase 4

---

## Epic 7: Hybrid Model Support (Qwen 3.5)

### US-15: Adapter abstraction for multiple attention types — DONE

**As a** developer supporting models with different attention mechanisms,
**I want** a polymorphic adapter layer that handles GQA, MLA, and hybrid
architectures transparently,
**so that** the compaction algorithm works across model families without
per-model special-casing.

**Acceptance criteria:**
- `kv_adapter` interface with `decode()`, `encode()`, `geometry()` methods
- `gqa_adapter` for standard GQA models (Llama, Mistral, Qwen 2.5)
- `mla_adapter` for MLA models (DeepSeek-V3) with latent projection + LSQ
- `noop_adapter` for incompatible layers (DeltaNet, Mamba)
- `hybrid_classifier` to detect layer types in mixed-architecture models
- Factory: `make_adapter()` dispatches by layer type
- 20+ adapter tests passing

**Implementation:** `kv-compact-adapter.h`, `test-kv-compact-adapter.cpp`

---

### US-16: Qwen3.5-0.8B end-to-end validation — TODO

**As a** developer validating compaction on the target model,
**I want** to run the full compaction pipeline on Qwen3.5-0.8B with real
prompts at multiple compression ratios,
**so that** I can measure actual quality and performance on the intended
deployment model.

**Acceptance criteria:**
- Download Qwen3.5-0.8B-Q4_K_M.gguf
- Hybrid layer detection: only compact 6/24 full-attention layers
- Run at 4x, 16x, and 50x compression ratios
- Report: perplexity delta, generation quality, wall-clock time
- Compare: full cache vs compacted vs naive token eviction
- GQA-aware sensitivity: multiply head sensitivity by GQA ratio (=4)

**Reference:** `plan.md` Phase 5

---

### US-17: Investigate DeltaNet state format in llama.cpp — TODO

**As a** developer ensuring correct handling of Qwen 3.5's hybrid layers,
**I want** to understand how llama.cpp stores Gated DeltaNet recurrent state
(vs standard KV cache),
**so that** the state parser correctly identifies and skips non-KV layers
during compaction.

**Acceptance criteria:**
- Document DeltaNet state layout in llama.cpp binary format
- Verify `parse_kv_state()` correctly handles hybrid state (KV + recurrent)
- `hybrid_classifier` correctly identifies DeltaNet vs full-attention layers
  based on model metadata
- No data corruption when writing back compacted state for hybrid models

---

## Epic 8: Greedy Budget Exchange

### US-18: Implement greedy budget exchange from paper §5 — TODO

**As a** researcher implementing the paper's most impactful optimization,
**I want** a calibration-based per-head budget allocation that greedily
exchanges budget between heads to minimize total reconstruction error,
**so that** each head gets the optimal number of retained tokens rather
than a uniform allocation.

**Acceptance criteria:**
- Calibration pass: run compaction on representative data, measure per-head
  reconstruction error at various budgets
- Greedy exchange: iteratively move budget from least-sensitive to
  most-sensitive heads until convergence
- Precomputed budgets can be saved/loaded per model
- Quality improvement over sensitivity weighting at 50x compression
- CLI: `--budget-exchange` flag to enable

---

## Summary Matrix

| # | Story | Epic | Status | Priority |
|---|-------|------|--------|----------|
| US-1 | Beta injection | Core | DONE | - |
| US-2 | C_v write-back | Core | DONE | - |
| US-3 | All layers/heads | Core | DONE | - |
| US-4 | Repeat-prefill Q_ref | Ref Queries | TODO | Medium |
| US-5 | Non-uniform budgets | Key Selection | DONE | - |
| US-6 | OMP key selection | Key Selection | TODO | Low |
| US-7 | Quantized KV | Production | DONE | - |
| US-8 | Library API | Production | TODO | Medium |
| US-9 | Multi-round compaction | Production | TODO | High |
| US-10 | Quality benchmarks | Benchmarking | TODO | Medium |
| US-11 | Memory/latency profiling | Benchmarking | PARTIAL | Medium |
| US-12 | Streaming compactor | Streaming | DONE | **Critical** |
| US-13 | Token pinning | Streaming | PARTIAL | **High** |
| US-14 | Error monitoring | Streaming | TODO | **High** |
| US-15 | Adapter abstraction | Hybrid Models | DONE | - |
| US-16 | Qwen3.5 E2E validation | Hybrid Models | TODO | **High** |
| US-17 | DeltaNet state format | Hybrid Models | TODO | Medium |
| US-18 | Greedy budget exchange | Budget Exchange | TODO | **High** |

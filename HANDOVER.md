# Handover: kv-compact

**Date:** 2026-03-19
**Branch:** `claude/arxiv-mcp-integration-t956A` (main worktree)
**Focus:** Qwen3.5-35B-A3B performance — 30 tok/s × 10 agents + Mistral Small 4 benchmarking

---

## What We're Trying to Achieve

Push **Qwen3.5-35B-A3B** on Strix Halo (128 GB LPDDR5X, 212 GB/s, Radeon 8060S)
from current 8.6 tok/s per slot at 10 agents to **30 tok/s per slot**.

Realistic ceiling based on memory bandwidth math: **~20 tok/s per slot at 10 agents**
(200 agg). For 30/slot, max ~7 concurrent agents. The 300 agg target is above the
hardware ceiling at 10 slots.

---

## Current Performance

| Agents | Per-slot tok/s | Agg tok/s | Build |
|--------|---------------|-----------|-------|
| 1 | 44.7 | 39.8 | Fork (Coder-Next) |
| 2 | 28.3 | 50.7 | Fork |
| 5 | 17.9 | 82.0 | Fork |
| 10 | **8.6** | **79.1** | Fork |
| **Target** | **30** | **300** | |

Single-slot generation: 60-68 tok/s (fork), 64 tok/s (stock).
Prefill: fork -17% vs stock (regression from flash attn row_split change).

---

## Per-Token Kernel Breakdown (single slot, Vulkan profiler)

| Category | % of Time | Multi-slot scaling |
|----------|-----------|-------------------|
| Shared expert FFN (q5_K) | **34.2%** | Batches well (GEMM n→10) |
| Attention QKV | 24.0% | Batches well |
| MoE experts (top-8, 256 pool) | **21.4%** | 10× unique reads (or 3× with caching) |
| SSM/DeltaNet state | ~10% | Does NOT batch — sequential per seq |
| Router + other | ~10% | Negligible |

Full profile: `docs/vulkan-perf-profile-35b.md`

### Key insight from profiling

The original plan assumed 4800 dispatches/step. Reality: **120 dispatches/step**
(llama.cpp batches all 8 experts per layer into one dispatch). Dispatch overhead
was NOT the bottleneck — the bottleneck at 10 slots is:

```
Single slot: ~16.5 ms/token
10 slots:    ~126 ms/step  (79 agg tok/s)
Overhead:     ~110 ms from multi-slot scaling
  SSM sequential:    10 × 1.7 ms = 17 ms  (must process each seq)
  MoE weight reads:  10 × 3.5 ms = 35 ms  (10× unique experts)
  Vulkan scheduling: ~58 ms               (sync, buffer mgmt, dispatch)
```

The **Vulkan scheduling overhead (58 ms)** is the largest single gap.

---

## What's Done

### Expert Caching (`include/kv-compact-moe-cache.h`)
EMA-based expert cache with dynamic routing bias. **Self-contained, zero deps.**

- `moe_expert_cache::init(nl, ne)` — initialize per-layer EMA
- `moe_expert_cache::update(layer, expert_ids, n_used)` — update EMA after decode
- `moe_expert_cache::get_bias(layer, expert_idx)` — routing bias for cached experts
- `moe_expert_cache::top_k_experts(layer, k)` — get hot expert indices
- Based on Cache-Aware Routing (arXiv:2412.00099) — 2x speedup, <0.1% quality loss

**Status: implemented but NOT wired into server decode loop.**

### Expert Cache Math Validation (`tests/test-moe-cache.cpp`)
Standalone benchmark validating all assumptions:
- Expert overlap formula vs Monte Carlo (5/5 PASS, error < 0.04)
- Bandwidth model for Qwen3.5-35B
- Cache-aware routing simulation (bias ≥ 0.1 fully steers routing)
- Full projection: **33.2 tok/s per-slot at 10 agents** achievable with caching

### Revised Plan (`docs/plan-moe-optimization-revised.md`)
Post-profiling updated strategy. Supersedes `docs/plan-expert-caching.md`.

---

## What's Left (Priority Order)

### P0: Vulkan Graph Caching (biggest gap — ~58 ms)

llama.cpp has `can_reuse()` for compute graph reuse. Unknown if it's working for
MoE multi-slot (the graph is identical across steps — only input data changes).

**Immediate action:**
```bash
GGML_VK_PERF_LOGGER=1 llama-server -m model.gguf -np 10 ...
```
Look for "graph reuse" messages. If absent, graph is being rebuilt every step.

- [ ] Verify `can_reuse()` is triggering for multi-slot MoE
- [ ] Fix buffer reallocation per slot if graph reuse is broken
- [ ] Vulkan command buffer batching (5-10 ms saved, low effort)

**Impact if fixed:** ~58 ms → ~15 ms → step ~83 ms → ~120 agg → 12 per-slot

### P1: SSM State Quantization (quick win — ~7 ms at 10 slots)

SSM/DeltaNet state is currently F32 (62 MB per sequence). Quantizing to F16
halves bandwidth. Try:
```bash
llama-server -m model.gguf -ctk f16 -ctv f16 ...
```
The flag may not apply to recurrent state (may be hardcoded F32 in llama.cpp).

- [ ] Test if `-ctk f16` reduces SSM state size (check VRAM or profiler output)
- [ ] If not, patch `llama_memory_recurrent` to use F16 for S/R tensors

**Impact:** 10 × 1.7 ms SSM → 10 × 0.85 ms → save ~8 ms/step

### P2: Wire Dynamic Cache Bias into Server

`kv-compact-moe-cache.h` is ready. Needs wiring into the server decode loop:

```cpp
// After each decode step:
expert_cache.update(layer, topk_expert_ids, n_used);

// Before next step:
float bias = expert_cache.get_bias(layer, expert_idx);
// inject as routing logit bias
```

Requires knowing where expert IDs are extracted in the Vulkan backend.
May need a callback or tensor hook in `llama_decode()`.

- [ ] Find where `ffn_moe_topk` tensor is read in llama.cpp Vulkan backend
- [ ] Add expert ID extraction callback
- [ ] Feed bias back into routing logits before each step

**Impact:** MoE unique experts: ~80 → ~20 per layer → 3× fewer weight reads →
save ~25 ms/step at 10 slots

### P3: Shared Expert Requant

Shared expert is **34% of single-slot time** (q5_K m=8192). No routing tricks help —
it runs every token. Re-quantizing to Q4_K reduces bandwidth ~30%.

- [ ] Re-quant shared expert weights: `llama-quantize --include "ffn_gate_shexp"` (check exact tensor name)
- [ ] Measure tg50 improvement

**Impact:** ~5.6 ms → ~3.9 ms per token (-30%), scales to all slot counts

### P4: Prefill Regression Fix

Fork is 17% slower at prefill (950 vs 1150 tok/s). Likely the flash attn
`row_split` UMA change in the fork.

- [ ] Profile prefill separately: `llama-bench -m model.gguf -p 512 -n 0`
- [ ] Compare fork vs stock for the specific UMA/flash-attn code path
- [ ] Revert `row_split` change or make it conditional on context size

---

## Worktrees

| Worktree | Path | Branch | Focus |
|----------|------|--------|-------|
| Main | `C:/Users/fabia/Projects/kv-compact` | `claude/arxiv-mcp-integration-t956A` | MoE caching, expert routing |
| MoE/ROCm throughput | `.claude/worktrees/teleport-session` | `claude/optimize-moe-rocm-throughput-mhaw9` | KV quant + B4/R-KV |
| Profiling infra | `.claude/worktrees/pr2-profiling-infrastructure` | `claude/list-branch-changes-qdPyO` | GPU profiling tools |

The MoE/ROCm worktree has:
- Quantized KV (B4): dequant→compact→requant pipeline
- R-KV reasoning token compression
- rocBLAS strided-batched GEMM (replaced custom HIP)
- Per-stage throughput benchmarks (NNLS is 94.5% of compaction time)

---

## Mistral Small 4 (New — 2026-03-19)

Mistral Small 4 (119B MoE, Apache 2.0, released 2026-03-18) is a strong new open model.
**No new kv-compact code needed** — it maps directly to the existing GQA adapter.

### Architecture

| Parameter | Value |
|-----------|-------|
| Adapter | `gqa` (MHA — 32Q/32KV heads, ratio=1) |
| `d_k` / `d_v` | 128 (head_dim) |
| `n_head_kv` | 32 |
| Layers | 36 (all full attention, uniform classifier) |
| MoE | 128 routed + 1 shared experts, 4 active/token |
| Context | 256K (YaRN RoPE) |

### Adapter config

```cpp
attention_arch arch{
    .type      = "gqa",
    .d_k       = 128,
    .d_v       = 128,
    .n_head_kv = 32,
};
auto adapter    = make_adapter(arch);         // gqa_adapter (identity)
auto classifier = make_classifier(arch, 36);  // all 36 layers compactible
```

### Model file

- **GGUF:** `D:\models\mistral-small-4\Mistral-Small-4-119B-2603-UD-Q4_K_XL.gguf` (73.8 GB)
- Unsloth UD-Q4_K_XL — best quality Q4 from Dynamic 2.0 quants
- **Download was in progress at session end** (background task) — verify file exists before benchmarking
- **Known issue:** llama.cpp #20703 — bad `--fit` offload; use `-ngl 99` explicitly, avoid `--fit`

### Next steps once downloaded

1. Verify: `ls -lh /d/models/mistral-small-4/*.gguf`
2. Smoke test: `llama-cli -m /d/models/mistral-small-4/Mistral-Small-4-119B-2603-UD-Q4_K_XL.gguf -ngl 99 -p "Hello" -n 20`
3. Perplexity sweep: reuse `bench_coding_agents.py` with `--ratios 4 16 50`
4. Compare quality vs Qwen3.5-35B-A3B and SmolLM3 baselines

---

## Key Files

| File | Purpose |
|------|---------|
| `include/kv-compact-moe-cache.h` | Expert cache implementation (EMA + routing bias) |
| `docs/plan-moe-optimization-revised.md` | Current strategy (post-profiling, 2026-03-15) |
| `docs/vulkan-perf-profile-35b.md` | Vulkan kernel breakdown (ground truth for bottleneck analysis) |
| `docs/plan-expert-caching.md` | Original plan (superseded but has bandwidth math) |
| `tests/test-moe-cache.cpp` | Expert cache validation + projection benchmark |
| `tests/moe_benchmark.json` | Benchmark results |

---

## Build Notes

```bash
# Main repo — test-only build
cmake -B build -DKV_COMPACT_BUILD_TOOL=OFF && cmake --build build

# Run expert cache tests
./build/test-moe-cache

# Benchmark
GGML_VK_PERF_LOGGER=1 ./llama-server -m /path/to/qwen35-35b-a3b-q4km.gguf -np 10 -ngl 99
```

MSVC requires explicit INCLUDE/LIB env vars. See `memory/feedback_msvc_env.md`.
Do NOT use the flash-attn fork for serving (5-11× server regression). Stock llama.cpp only.

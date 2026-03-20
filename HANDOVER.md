# Handover: kv-compact (MoE Optimization Branch)

**Date:** 2026-03-20
**Branch:** `claude/optimize-moe-rocm-throughput-mhaw9` (worktree: teleport-session)
**Focus:** MoE expert caching and multi-slot throughput optimization

---

## What Was Done This Session

### 1. Consolidated MoE-relevant code from main branch
Cherry-picked + copied expert caching infrastructure (commit `9bdbe09`):
- `include/kv-compact-moe-cache.h` — EMA expert cache + routing bias
- `docs/plan-expert-caching.md`, `vulkan-perf-profile-35b.md`, `plan-moe-optimization-revised.md`
- `tests/bench-expert-cache.cpp`, `test-moe-cache.cpp`, `test-budget-exchange.cpp`
- `patches/apply.sh` — llama.cpp attention bias + MoE routing patches
- `greedy_budget_exchange()` added to kv-compact-math.h
- All tests pass, all targets build

### 2. Investigation sweep (eliminated 3 of 4 priority items)

| Item | Result |
|------|--------|
| **P0: Vulkan graph reuse** | **Already exists** — backend-agnostic in `llama-graph.cpp`. `can_reuse()` is implemented. NOT the bottleneck. |
| **P1: SSM F16** | `-ctk`/`-ctv` flags do NOT affect recurrent state (hardcoded F32 in `llama_memory_recurrent`). Needs source patch. |
| **Shared expert requant** | **Wrong target** — profiler labels were incorrect. The 34.2% bottleneck is `attn_qkv` (Q5_K [2048,8192]), NOT shared expert (Q6_K [512,2048], 0.82 MiB, negligible). |
| **attn_qkv requant** | `--tensor-type` doesn't work with `--allow-requantize` (bug). Q4_K_S full requant shows no speed benefit on Vulkan (Q5_K/Q4_K dequant run at similar speed). |
| **Expert cache validation** | bench-expert-cache confirms: **2.25x speedup at 10 slots** with cache-aware routing (33.2 tok/s projected). |

### 3. Corrected Vulkan profiler labels (commit `6fd8d60`)
The handover's profiler categories were mislabeled:
- "Shared expert FFN (34.2%)" → actually **Attention QKV** (`blk.*.attn_qkv` Q5_K)
- "Attention QKV (24.0%)" → actually **SSM/DeltaNet gate** (`blk.*.attn_gate` Q4_K)
- Real shared expert is tiny (0.82 MiB/layer Q6_K)

### 4. P3: Cache-aware MoE routing — partial implementation
Patched llama.cpp source at `build-native/_deps/llama_cpp-src/`:
- Added `LLAMA_MOE_CACHE_AWARE=1` env var trigger
- Added `moe_cache` pointer to `llm_graph_params` and `llm_graph_context`
- Added `build_inp_moe_bias()` — creates per-layer bias tensors
- Wired into `qwen35moe.cpp` — passes `res->t_moe_bias[il]` to `build_moe_ffn`
- Built llama-server with Vulkan at `build-vk/bin/Release/llama-server.exe`
- **STATUS: CRASHES** during first inference (segfault). Without `LLAMA_MOE_CACHE_AWARE`, the patched build works fine (36 tok/s single-slot).

**Likely cause:** The `ggml_add(selection_probs, moe_cache_bias)` where `selection_probs` is [n_expert, n_tokens] (2D) and bias is [n_expert] (1D). Either broadcasting isn't supported on the Vulkan backend for this op, or the graph scheduler assigns the bias tensor to the wrong backend, causing a split.

**Next debug step:** Check if `ggml_add` broadcasting works for [n_expert, n_tokens] + [n_expert] on Vulkan. Try `ggml_repeat(bias, selection_probs)` before the add to explicitly match shapes. Or check graph splits — the extra split (3 vs 2) suggests a backend assignment issue.

---

## Revised Priority Stack

| # | Task | Status | Effort |
|---|------|--------|--------|
| 1 | **P3: Fix MoE cache bias crash** | Segfault in ggml_add broadcasting or graph scheduling | 2-4h debug |
| 2 | **P1: SSM F16 source patch** | Confirmed needs `llama_memory_recurrent` F16 support | 6h |
| 3 | **Investigate real scheduling overhead** | 8-15ms gap between kernel time and wall time at 2 slots | 2h |

---

## Key Files

| File | Purpose |
|------|---------|
| `include/kv-compact-moe-cache.h` | Expert cache (EMA + routing bias) — tested, works |
| `tests/bench-expert-cache.cpp` | Math validation — all projections confirmed |
| `docs/vulkan-perf-profile-35b.md` | **Corrected** Vulkan kernel breakdown |
| `docs/plan-moe-optimization-revised.md` | Strategy (post-profiling, P0-P4) |
| `patches/apply.sh` | llama.cpp patches (attention bias + MoE routing) |
| `build-native/_deps/llama_cpp-src/` | Patched llama.cpp source (P3 wiring in progress) |

## Build Notes

```bash
# kv-compact tests (this worktree)
cmake -B build -DKV_COMPACT_BUILD_TOOL=OFF && cmake --build build --config Release
./build/Release/test-moe-cache.exe
./build/Release/test-budget-exchange.exe
./build/Release/bench-expert-cache.exe

# Patched llama-server (in build-native/_deps/llama_cpp-src)
cd build-native/_deps/llama_cpp-src
cmake --build build-vk --config Release --target llama-server

# Test without MoE cache (works)
./build-vk/bin/Release/llama-server.exe -m model.gguf -ngl 99

# Test with MoE cache (CRASHES)
LLAMA_MOE_CACHE_AWARE=1 ./build-vk/bin/Release/llama-server.exe -m model.gguf -ngl 99
```

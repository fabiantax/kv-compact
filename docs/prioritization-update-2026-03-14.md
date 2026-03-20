# Prioritization Update — 2026-03-14 Session Findings

**Context:** Benchmarked Qwen3.5-4B/9B/35B-A3B and Qwen3-Coder-Next (80B.A3B).
Discovered hybrid model bottlenecks. Surveyed 16 arxiv papers. Key insight:
hybrid models benefit from compaction as **memory enablement**, not speed.

**Status update:** A1, B1, B2, B3, B4 are all DONE. Scoring only remaining TODO items.

---

## New Candidates (from this session)

| # | Feature | Description | Source |
|---|---------|-------------|--------|
| D1 | Integrate into main path | Make compaction callable from llama-server runtime | US-24 |
| D2 | Expert weight caching | Predict/prefetch MoE experts for consecutive tokens | MoE optimization |
| D3 | AIRE-Prune GDN state | Prune 60% of SSM state dims, 0.29% accuracy drop | arxiv 2602.00534 |
| D4 | Apt-Serve hybrid scheduling | KV + hidden-state dual-cache adaptive batching | arxiv 2504.07494 |
| D5 | Marconi prefix caching | Cache recurrent state for shared system prompts | arxiv 2411.19379 |
| D6 | LongFlow fused kernel | Fuse FlashAttn + importance + eviction in one kernel | arxiv 2603.11504 |
| D7 | Thin Keys SVD compression | Compress key dimension by 75% (orthogonal to token selection) | arxiv 2603.04427 |
| D8 | Fix multi-slot Vulkan pressure | Debug Coder-Next 2K+ KV failing at 5+ agents | Benchmark finding |
| D9 | Stock build benchmark suite | Re-run all benchmarks on stock llama.cpp b8334 | Benchmark finding |

---

## RICE Scores

| Feature | Reach | Impact | Confidence | Effort | RICE | Rank |
|---------|-------|--------|------------|--------|------|------|
| **D9** Stock benchmarks | 10 | 5 | 0.95 | 1 | **47.5** | 1 |
| **D1** Main path integration | 10 | 9 | 0.80 | 5 | **14.4** | 2 |
| **D3** AIRE-Prune GDN | 8 | 6 | 0.70 | 3 | **11.2** | 3 |
| **D5** Marconi prefix cache | 10 | 7 | 0.60 | 5 | **8.4** | 4 |
| **D8** Fix Vulkan pressure | 8 | 5 | 0.50 | 4 | **5.0** | 5 |
| **D2** Expert caching | 10 | 7 | 0.50 | 6 | **5.8** | 6 |
| **D4** Apt-Serve scheduling | 10 | 8 | 0.50 | 7 | **5.7** | 7 |
| **D7** Thin Keys SVD | 10 | 6 | 0.55 | 6 | **5.5** | 8 |
| **D6** LongFlow fused kernel | 10 | 8 | 0.45 | 8 | **4.5** | 9 |

---

## WSJF Scores

| Feature | UBV | TC | RR | CoD | Duration | WSJF | Rank |
|---------|-----|----|----|-----|----------|------|------|
| **D9** Stock benchmarks | 5 | 8 | 5 | **18** | 1 | **18.0** | 1 |
| **D1** Main path integration | 9 | 7 | 7 | **23** | 5 | **4.6** | 2 |
| **D3** AIRE-Prune GDN | 6 | 4 | 5 | **15** | 3 | **5.0** | 3 |
| **D5** Marconi prefix cache | 7 | 4 | 4 | **15** | 5 | **3.0** | 4 |
| **D2** Expert caching | 7 | 5 | 4 | **16** | 6 | **2.7** | 5 |
| **D8** Fix Vulkan pressure | 5 | 5 | 3 | **13** | 4 | **3.3** | 6 |
| **D4** Apt-Serve scheduling | 8 | 3 | 5 | **16** | 7 | **2.3** | 7 |
| **D7** Thin Keys SVD | 6 | 3 | 4 | **13** | 6 | **2.2** | 8 |
| **D6** LongFlow fused kernel | 8 | 3 | 4 | **15** | 8 | **1.9** | 9 |

---

## Kano Classification

| Feature | Kano | Rationale |
|---------|------|-----------|
| **D1** Main path integration | **Must-Be** | Without runtime integration, compaction is an offline curiosity |
| **D9** Stock benchmarks | **Must-Be** | Accurate baselines are required; fork numbers are 5-11x wrong |
| **D8** Fix Vulkan pressure | **Performance** | More agents at larger context = linear improvement |
| **D3** AIRE-Prune GDN | **Performance** | Less recurrent state = more agents fit linearly |
| **D2** Expert caching | **Performance** | Better expert reuse = higher tok/s linearly |
| **D5** Marconi prefix cache | **Attractive** | Shared prefill is an unexpected efficiency gain |
| **D4** Apt-Serve scheduling | **Attractive** | Smart scheduling is invisible to users; unexpected throughput |
| **D7** Thin Keys SVD | **Attractive** | Orthogonal compression users don't expect |
| **D6** LongFlow fused kernel | **Attractive** | Internal optimization; users see speed but don't expect kernel fusion |

---

## ROI Analysis

Assumptions: 128GB UMA, Qwen3-Coder-Next 80B.A3B (46GB), 10 agents target.

| Feature | Benefit | Cost (weeks) | ROI | Rank |
|---------|---------|--------------|-----|------|
| **D9** Stock benchmarks | Accurate data for all decisions. Currently 5-11x wrong. | 0.5 | **100x** | 1 |
| **D3** AIRE-Prune GDN | 60% less recurrent state: 750MB→300MB for 10 agents | 1.5 | **8x** | 2 |
| **D1** Main path integration | Gate for production use. 0→1 value unlock. | 2.5 | **7x** | 3 |
| **D5** Marconi prefix cache | Skip 10 redundant prefills of shared system prompt | 2.5 | **5x** | 4 |
| **D2** Expert caching | 20-40% less weight bandwidth from expert reuse | 3 | **4x** | 5 |
| **D8** Fix Vulkan pressure | Enable 10 agents at 2K+ KV (currently fails at 5+) | 2 | **3.5x** | 6 |
| **D4** Apt-Serve scheduling | 8.8x effective throughput improvement (paper claim) | 3.5 | **3x** | 7 |
| **D7** Thin Keys SVD | 75% key dimension reduction; stacks with token selection | 3 | **2.5x** | 8 |
| **D6** LongFlow fused kernel | Eliminate compaction overhead entirely (fuse into FlashAttn) | 4 | **2x** | 9 |

---

## Composite Score (all frameworks)

Weights: RICE 25% + WSJF 25% + Kano 20% + ROI 20% + Portability 10%

Kano: Must-Be=100, Performance=60, Attractive=30, Indifferent=0.
Portability: 10=universal, 5=AMD-specific, 3=model-specific.

| Rank | Feature | RICE_n | WSJF_n | Kano | ROI_n | Port | **Composite** |
|------|---------|--------|--------|------|-------|------|---------------|
| **1** | **D9** Stock benchmarks | 100 | 100 | 100 | 100 | 100 | **100** |
| **2** | **D1** Main path integration | 30 | 26 | 100 | 70 | 100 | **62** |
| **3** | **D3** AIRE-Prune GDN state | 24 | 28 | 60 | 80 | 30 | **42** |
| **4** | **D5** Marconi prefix cache | 18 | 17 | 30 | 50 | 50 | **30** |
| **5** | **D2** Expert weight caching | 12 | 15 | 60 | 40 | 30 | **30** |
| **6** | **D8** Fix Vulkan pressure | 11 | 18 | 60 | 35 | 30 | **29** |
| **7** | **D4** Apt-Serve scheduling | 12 | 13 | 30 | 30 | 50 | **24** |
| **8** | **D7** Thin Keys SVD | 12 | 12 | 30 | 25 | 80 | **27** |
| **9** | **D6** LongFlow fused kernel | 9 | 11 | 30 | 20 | 50 | **21** |

---

## Combined Backlog (TODO items, all sessions)

Merging new D-series with remaining TODO from previous sessions:

| Priority | Feature | Composite | Kano | Phase |
|----------|---------|-----------|------|-------|
| **P0** | **D9** Stock build benchmark suite | 100 | Must-Be | Now |
| **P0** | **D1** Integrate into main code path (US-24) | 62 | Must-Be | Now |
| **P1** | **B6** Multi-round compaction (US-9) | 37 | Performance | Next |
| **P1** | **D3** AIRE-Prune GDN state | 42 | Performance | Next |
| **P1** | **B5** C library API (US-8) | 37 | Performance | Next |
| **P2** | **D5** Marconi prefix caching | 30 | Attractive | Soon |
| **P2** | **D2** Expert weight caching (MoE) | 30 | Performance | Soon |
| **P2** | **D8** Fix Vulkan multi-slot pressure | 29 | Performance | Soon |
| **P2** | **US-4** Repeat-prefill Q_ref | 28 | Performance | Soon |
| **P3** | **D7** Thin Keys SVD compression | 27 | Attractive | Later |
| **P3** | **D4** Apt-Serve hybrid scheduling | 24 | Attractive | Later |
| **P3** | **D6** LongFlow fused kernel | 21 | Attractive | Later |
| **P3** | **B7** Iterative refinement | 21 | Attractive | Later |
| **P3** | **A2** Cross-layer KV sharing | 18 | Attractive | Later |
| **P4** | **C1** ROCm/HIP acceleration | 13 | Attractive | If needed |
| **--** | **A5** DeltaNet state quant | 6 | Indifferent | Skip |
| **--** | **A3** MoE batch routing | 4 | Indifferent | Skip |

---

## Recommended Next Actions

### Immediate (this week)
1. **D9: Stock build benchmarks** — Our SmolLM3 numbers are 5-11x underreported. Re-run the full suite on stock llama.cpp b8334. Effort: half a day. Impact: corrects ALL serving projections.

2. **D1: Integrate into main path (US-24)** — Currently compaction is offline. Making `llama-kv-compact` callable at runtime (or as a library) unlocks production use. This is the **Critical** blocker.

### Next sprint
3. **D3: AIRE-Prune** — 60% recurrent state reduction with 0.29% accuracy drop. For Coder-Next, cuts per-agent recurrent from 75MB to 30MB. Could fix the multi-slot scaling wall.

4. **B6: Multi-round compaction** — Coding agents have long sessions. Without this, agents die at context limits.

5. **D2: Expert caching** — Given the MoE focus: consecutive tokens often route to the same experts. Predicting and caching reduces the effective 46GB weight read to ~10GB.

### Decision point
After D9 (accurate benchmarks) and D1 (runtime integration), re-evaluate:
- If multi-slot works better on stock build → D8 may be moot
- If expert caching shows big wins → prioritize D2 over D3
- If AIRE-Prune is easy to prototype → validate D3 quickly

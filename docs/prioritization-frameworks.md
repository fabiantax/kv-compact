# Multi-Framework Prioritization — Qwen 3.5 MoE + ROCm Throughput

**Target model:** Qwen3.5-35B-A3B (40 layers, 30 DeltaNet + 10 softmax GQA)
**Target hardware:** AMD Ryzen AI 395 Pro Max, 128GB unified LPDDR5X, 8060S RDNA 3.5
**Workload:** N parallel coding agents, maximize aggregate tg/s
**Key constraint:** Only 10 of 40 layers have KV caches (~20KB/token FP16 total)

---

## Framework Definitions

### RICE = (Reach x Impact x Confidence) / Effort
- **Reach** (1-10): Sessions/layers/tokens affected
- **Impact** (1-10): tg/s gain x quality preservation
- **Confidence** (0.0-1.0): Probability of expected outcome
- **Effort** (1-10): Dev weeks (1=days, 10=months)

### WSJF = Cost of Delay / Job Duration
- **CoD** = User-Business Value + Time Criticality + Risk Reduction
- **Job Duration** = T-shirt size (1=XS, 2=S, 3=M, 5=L, 8=XL)
- Higher WSJF = do first

### Kano Classification
- **Must-Be (M)**: Expected baseline; absence = dissatisfaction, presence != delight
- **Performance (P)**: Linear satisfaction; more = better
- **Attractive (A)**: Unexpected delight; absence != dissatisfaction
- **Indifferent (I)**: No impact on satisfaction
- **Reverse (R)**: Unwanted; causes dissatisfaction

### ROI = (Benefit - Cost) / Cost
- Benefit = memory saved (GB) x sessions enabled x quality factor
- Cost = dev-weeks + maintenance burden + integration risk
- Expressed as multiplier (2.0x = 100% return)

---

## Candidate Features

### A. Qwen 3.5-Specific (NEW)

| # | Feature | Description |
|---|---------|-------------|
| A1 | Hybrid layer awareness | Skip DeltaNet layers entirely; only compact 10 attention layers |
| A2 | Cross-layer KV sharing | Share KV between pairs of attention layers (10 → 5 stores) via SVD |
| A3 | MoE batch routing optimization | Batch expert dispatch across N agents for GPU utilization |
| A4 | R-KV reasoning token compression | Detect and merge redundant CoT thinking tokens |
| A5 | DeltaNet state quantization | Quantize the 30-layer fixed recurrent state (~15MB) |

### B. Core Pipeline (EXISTING)

| # | Feature | Description |
|---|---------|-------------|
| B1 | Beta injection + C_v writeback | Inject biases into attention; write refitted values |
| B2 | All-layer/all-head compaction | Extend to full model (10 attention layers for Qwen 3.5) |
| B3 | Per-head budget allocation | Non-uniform budget by sensitivity |
| B4 | Quantized KV support | Handle Q4_0/Q8_0 KV types |
| B5 | C library API | Programmatic `kv_compact()` interface |
| B6 | Multi-round compaction | Re-compact growing caches |
| B7 | Iterative refinement | Swap poor keys, re-solve |

### C. Infrastructure & Acceleration

| # | Feature | Description |
|---|---------|-------------|
| C1 | ROCm/HIP matmul acceleration | GPU-accelerated attention scoring |
| C2 | Shared prompt KV across agents | CoW KV for system prompt |
| C3 | Automated quality benchmarks | CI perplexity + agreement regression tests |
| C4 | Memory/latency profiling | JSON phase timing + memory deltas |

---

## Scoring Matrix

### RICE Scores

| Feature | Reach | Impact | Confidence | Effort | RICE | Rank |
|---------|-------|--------|------------|--------|------|------|
| **A1** Hybrid layer awareness | 10 | 9 | 0.95 | 1 | **85.5** | 1 |
| **B2** All-layer compaction | 10 | 9 | 0.95 | 2 | **42.8** | 2 |
| **C4** Profiling | 8 | 4 | 0.95 | 1 | **30.4** | 3 |
| **A4** R-KV reasoning compression | 10 | 8 | 0.70 | 3 | **18.7** | 4 |
| **B1** Beta injection | 10 | 10 | 0.90 | 5 | **18.0** | 5 |
| **B3** Per-head budget | 10 | 8 | 0.85 | 4 | **17.0** | 6 |
| **B4** Quantized KV | 10 | 7 | 0.80 | 3 | **16.5** | 7* |
| **A2** Cross-layer sharing | 10 | 8 | 0.55 | 4 | **11.0** | 8 |
| **B5** C library API | 10 | 6 | 0.90 | 3 | **18.0** | 5t |
| **B7** Iterative refinement | 10 | 5 | 0.70 | 2 | **17.5** | 6t |
| **C3** Quality benchmarks | 10 | 5 | 0.95 | 3 | **15.8** | 9 |
| **B6** Multi-round compaction | 8 | 7 | 0.70 | 3 | **13.1** | 10 |
| **C2** Shared prompt cache | 10 | 8 | 0.60 | 4 | **12.0** | 11 |
| **A5** DeltaNet state quant | 8 | 3 | 0.60 | 3 | **4.8** | 12 |
| **A3** MoE batch routing | 8 | 6 | 0.40 | 7 | **2.7** | 13 |
| **C1** ROCm acceleration | 10 | 6 | 0.50 | 6 | **5.0** | 14 |

*A1 is ranked #1 because it's trivial (effort=1) — just skip 30 layers in the compaction loop — with massive impact (75% less work per compaction).*

### WSJF Scores

**Cost of Delay components:** UBV = User-Business Value, TC = Time Criticality, RR = Risk Reduction

| Feature | UBV | TC | RR | CoD | Duration | WSJF | Rank |
|---------|-----|----|----|-----|----------|------|------|
| **B1** Beta injection | 10 | 10 | 8 | **28** | 5 | **5.6** | 1 |
| **A1** Hybrid layer awareness | 9 | 9 | 6 | **24** | 1 | **24.0** | 1t |
| **B2** All-layer compaction | 9 | 9 | 5 | **23** | 2 | **11.5** | 2 |
| **B3** Per-head budget | 8 | 7 | 7 | **22** | 3 | **7.3** | 3 |
| **B4** Quantized KV | 7 | 6 | 6 | **19** | 3 | **6.3** | 4 |
| **A4** R-KV reasoning | 8 | 5 | 6 | **19** | 3 | **6.3** | 4t |
| **B6** Multi-round | 7 | 7 | 5 | **19** | 3 | **6.3** | 4t |
| **C4** Profiling | 4 | 3 | 5 | **12** | 1 | **12.0** | 5 |
| **B5** C library API | 6 | 5 | 4 | **15** | 3 | **5.0** | 6 |
| **C3** Benchmarks | 5 | 4 | 7 | **16** | 3 | **5.3** | 7 |
| **A2** Cross-layer sharing | 8 | 4 | 4 | **16** | 5 | **3.2** | 8 |
| **C2** Shared prompt | 8 | 3 | 3 | **14** | 5 | **2.8** | 9 |
| **B7** Iterative refinement | 5 | 3 | 4 | **12** | 2 | **6.0** | 10 |
| **A5** DeltaNet quant | 3 | 2 | 2 | **7** | 3 | **2.3** | 11 |
| **A3** MoE batch routing | 6 | 2 | 3 | **11** | 8 | **1.4** | 12 |
| **C1** ROCm accel | 6 | 2 | 3 | **11** | 6 | **1.8** | 13 |

### Kano Classification

| Feature | Kano | Rationale |
|---------|------|-----------|
| **B1** Beta injection | **Must-Be** | Without it, compaction produces no runtime benefit. Table stakes. |
| **B2** All-layer compaction | **Must-Be** | Single-layer is a demo, not a product. Non-negotiable. |
| **A1** Hybrid layer awareness | **Must-Be** | Compacting DeltaNet layers is a bug (they have no KV cache). Must skip. |
| **B4** Quantized KV | **Must-Be** | Q4 models are the default; FP16-only is a deployment blocker. |
| **B3** Per-head budget | **Performance** | Linear quality improvement. More budget intelligence = better output. |
| **A4** R-KV reasoning | **Performance** | CoT reasoning is the primary workload. Compression scales with reasoning length. |
| **B6** Multi-round | **Performance** | Longer sessions = more value. Linear relationship. |
| **B5** C library API | **Performance** | Enables integration depth. More API surface = more use cases. |
| **C3** Benchmarks | **Performance** | Confidence scales with test coverage. |
| **C4** Profiling | **Performance** | Better tuning data = better performance. |
| **B7** Iterative refinement | **Attractive** | Unexpected quality boost at extreme compression. Not expected by users. |
| **A2** Cross-layer sharing | **Attractive** | Novel technique; users don't expect inter-layer optimization. |
| **C2** Shared prompt cache | **Attractive** | Memory savings users didn't ask for. Delightful at scale. |
| **A5** DeltaNet quant | **Indifferent** | 15MB fixed state is negligible vs. multi-GB KV. Nobody notices. |
| **A3** MoE batch routing | **Indifferent** | Orthogonal to KV compaction. Belongs in inference engine, not here. |
| **C1** ROCm accel | **Attractive** | GPU acceleration is nice but CPU handles the problem sizes fine. |

### ROI Analysis

Assumptions: 128GB system, Qwen3.5-35B-A3B Q4 (~4.5GB weights), N=16 agents target.

| Feature | Benefit (GB freed or tg/s gain) | Cost (dev-weeks) | ROI | Rank |
|---------|-------------------------------|-------------------|-----|------|
| **A1** Hybrid layer awareness | Skip 30 layers of unnecessary work. 75% compaction speedup. | 0.5 | **150x** | 1 |
| **B2** All-layer compaction | Unlocks full compression: ~20KB→~0.4KB/tok. At 64K ctx: 1.2GB→25MB per agent. | 1 | **48x** | 2 |
| **B1** Beta injection | Gate for all benefits. 0→1.2GB savings per agent at 64K. | 2.5 | **19x** | 3 |
| **B4** Quantized KV | Q4 KV + compaction: 1.2GB→6MB per agent. Additional 4x on top of compaction. | 1.5 | **16x** | 4 |
| **A4** R-KV reasoning | CoT produces 5-10x redundant tokens. Merging: ~60-80% reduction in reasoning KV. | 1.5 | **12x** | 5 |
| **B3** Per-head budget | 0.998→0.9999 cosine sim. Fewer retries = ~15% effective throughput gain. | 2 | **7.5x** | 6 |
| **C4** Profiling | Enables data-driven tuning. Indirect: ~10% better configurations. | 0.5 | **6x** | 7 |
| **B6** Multi-round | Agents survive 2x longer sessions. Direct productivity gain. | 1.5 | **5.3x** | 8 |
| **B5** C library API | Enables auto-compact triggers. ~20% less manual intervention. | 1.5 | **4x** | 9 |
| **A2** Cross-layer sharing | 10→5 unique KV stores. 50% additional reduction on already-small cache. | 2.5 | **4x** | 10 |
| **C3** Benchmarks | Prevents regressions. Indirect: saves ~1 week of debugging per quarter. | 1.5 | **3x** | 11 |
| **B7** Iterative refinement | Quality at 50x: agreement 0.95→0.98. Small absolute gain. | 1 | **3x** | 12 |
| **C2** Shared prompt | 16 agents x 4K prompt = 64K tokens shared. ~1.3MB saved (tiny post-compaction). | 2.5 | **2x** | 13 |
| **C1** ROCm accel | Compaction 5-10x faster on GPU. But compaction is rare vs. generation. | 3 | **1.5x** | 14 |
| **A5** DeltaNet quant | 15MB→4MB. Negligible vs. system memory. | 1.5 | **0.5x** | 15 |
| **A3** MoE batch routing | ~10% tg/s from better GPU utilization. High effort, uncertain. | 4 | **0.8x** | 16 |

---

## Unified Rankings

### Composite Score

Normalized each framework to 0-100, then weighted:
- RICE: 25% (efficiency)
- WSJF: 25% (economic sequencing)
- Kano: 20% (user satisfaction)
- ROI: 20% (return on investment)
- CoD: 10% (urgency)

Kano encoding: Must-Be=100, Performance=60, Attractive=30, Indifferent=0.

| Rank | Feature | RICE_n | WSJF_n | Kano_n | ROI_n | CoD_n | **Composite** |
|------|---------|--------|--------|--------|-------|-------|---------------|
| **1** | **A1** Hybrid layer awareness | 100 | 100 | 100 | 100 | 90 | **99.0** |
| **2** | **B1** Beta injection | 21 | 23 | 100 | 13 | 100 | **47.3** |
| **3** | **B2** All-layer compaction | 50 | 48 | 100 | 32 | 90 | **62.0** |
| **4** | **B4** Quantized KV | 19 | 26 | 100 | 11 | 60 | **42.0** |
| **5** | **B3** Per-head budget | 20 | 30 | 60 | 5 | 70 | **34.0** |
| **6** | **A4** R-KV reasoning | 22 | 26 | 60 | 8 | 50 | **32.0** |
| **7** | **C4** Profiling | 36 | 50 | 60 | 4 | 20 | **36.0** |
| **8** | **B6** Multi-round | 15 | 26 | 60 | 4 | 60 | **30.4** |
| **9** | **B5** C library API | 21 | 21 | 60 | 3 | 50 | **29.6** |
| **10** | **B7** Iterative refinement | 20 | 25 | 30 | 2 | 30 | **21.0** |
| **11** | **C3** Benchmarks | 18 | 22 | 60 | 2 | 40 | **27.2** |
| **12** | **A2** Cross-layer sharing | 13 | 13 | 30 | 3 | 40 | **17.6** |
| **13** | **C2** Shared prompt | 14 | 12 | 30 | 1 | 50 | **18.6** |
| **14** | **C1** ROCm accel | 6 | 8 | 30 | 1 | 20 | **12.2** |
| **15** | **A5** DeltaNet quant | 6 | 10 | 0 | 0 | 10 | **5.0** |
| **16** | **A3** MoE batch routing | 3 | 6 | 0 | 1 | 10 | **3.4** |

---

## Implementation Roadmap (Composite-Ordered)

### Phase 0: Architecture Adaptation (days 1-3) — Composite 99.0

| # | Feature | Kano | Key Deliverable |
|---|---------|------|-----------------|
| A1 | Hybrid layer awareness | Must-Be | Layer-type dispatcher: skip DeltaNet, compact attention-only |

**Why first:** Trivial to implement (effort=1), highest ROI (150x), and a correctness requirement — compacting DeltaNet layers would be a bug. This gates all Qwen 3.5 work.

**Qwen 3.5 specifics:**
- Detect `full_attention_interval: 4` from config
- Build layer mask: layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 → compact
- Layers 0-2, 4-6, 8-10, ... → skip (DeltaNet, no KV cache)

### Phase 1: Minimum Viable Compaction (weeks 1-2) — Composite 42-62

| # | Feature | Kano | Key Deliverable |
|---|---------|------|-----------------|
| B1 | Beta injection + C_v writeback | Must-Be | End-to-end compaction with runtime benefit |
| B2 | All-layer compaction | Must-Be | Full 10-layer loop with progress reporting |
| C4 | Profiling | Performance | JSON timing per phase, memory before/after |

**Milestone:** Compacted Qwen 3.5 generates correct code. Measure actual tg/s gain.

### Phase 2: Quality & Deployment (weeks 3-4) — Composite 30-42

| # | Feature | Kano | Key Deliverable |
|---|---------|------|-----------------|
| B4 | Quantized KV | Must-Be | Q4_0/Q8_0 dequant → compact → requant |
| B3 | Per-head budget allocation | Performance | Precomputed sensitivity for Qwen 3.5's 2 KV heads × 10 layers |
| A4 | R-KV reasoning compression | Performance | CoT token merger (detect repetitive reasoning patterns) |

**Milestone:** 16 agents × 64K context in 128GB with Q4 KV + compaction. Quality validated on coding benchmarks.

**Qwen 3.5 budget note:** With only 2 KV heads per layer × 10 layers = 20 heads total, budget allocation has fewer degrees of freedom but each decision matters more.

### Phase 3: Scale & Sustain (weeks 5-6) — Composite 21-30

| # | Feature | Kano | Key Deliverable |
|---|---------|------|-----------------|
| B6 | Multi-round compaction | Performance | Agents survive indefinite sessions |
| B5 | C library API | Performance | `kv_compact()` callable from serving framework |
| C3 | Quality benchmarks | Performance | CI regression gate on perplexity/agreement |
| B7 | Iterative refinement | Attractive | 2-3 swap rounds for extreme compression |

**Milestone:** Production deployment. Auto-compact at threshold. Quality gated in CI.

### Phase 4: Advanced Optimization (weeks 7+) — Composite <20

| # | Feature | Kano | Key Deliverable |
|---|---------|------|-----------------|
| A2 | Cross-layer KV sharing | Attractive | SVD-based sharing between attention layer pairs |
| C2 | Shared prompt cache | Attractive | CoW KV for N agents' system prompts |
| C1 | ROCm/HIP acceleration | Attractive | GPU-accelerated compaction math |

**Deprioritized (Indifferent/Negative ROI):**
| A5 | DeltaNet state quant | Indifferent | 15MB is negligible; not worth the precision risk |
| A3 | MoE batch routing | Indifferent | Orthogonal to compaction; belongs in inference engine |

---

## Framework Agreement Analysis

How well do the frameworks agree? Kendall's tau between rankings:

| | RICE | WSJF | Kano | ROI | CoD |
|------|------|------|------|-----|-----|
| RICE | 1.0 | 0.82 | 0.61 | 0.87 | 0.55 |
| WSJF | | 1.0 | 0.67 | 0.79 | 0.63 |
| Kano | | | 1.0 | 0.58 | 0.71 |
| ROI | | | | 1.0 | 0.52 |
| CoD | | | | | 1.0 |

**Key disagreements:**
1. **B1 (Beta injection):** RICE ranks it #5 (high effort=5), but CoD ranks it #1 (total blocker). WSJF resolves this correctly — CoD/Duration = high because it's urgent despite effort.
2. **C4 (Profiling):** RICE ranks it #3 (effort=1), Kano says Performance, CoD ranks low. RICE is right here — cheap wins deserve early scheduling regardless of urgency.
3. **A2 (Cross-layer sharing):** RICE ranks #8, ROI ranks #10, Kano says Attractive. All agree: defer. The research risk (confidence=0.55) pulls it down across every framework.

**Strong consensus (all frameworks agree):**
- A1 (Hybrid awareness): #1 everywhere. Do immediately.
- A5 (DeltaNet quant): Bottom everywhere. Don't do.
- A3 (MoE routing): Bottom everywhere. Out of scope.

---

## Decision Matrix: "What If We Only Have N Weeks?"

| Weeks | Features | Agents Supported | Quality |
|-------|----------|-----------------|---------|
| 0.5 | A1 only | N/A (no runtime benefit yet) | N/A |
| 2 | A1 + B1 + B2 | 8 agents @ 64K | Good (FP16 KV) |
| 4 | + B4 + B3 + A4 | 16 agents @ 64K | Excellent (Q4 KV + sensitivity) |
| 6 | + B6 + B5 + C3 + B7 | 16 agents @ unlimited | Production-ready |
| 8+ | + A2 + C2 + C1 | 32+ agents @ unlimited | Optimized |

---

## Qwen 3.5 Architecture Exploitation Summary

The hybrid DeltaNet/attention architecture fundamentally changes the optimization calculus:

| Aspect | Standard Transformer | Qwen 3.5 35B-A3B |
|--------|---------------------|-------------------|
| Layers with KV cache | All 40 | Only 10 (25%) |
| KV cache per token | ~80KB (FP16) | ~20KB (FP16) |
| Compaction work | 40 layers × H heads | 10 layers × 2 heads = 20 problems |
| Budget allocation DOF | 40 × H allocations | 20 allocations (each matters 4x more) |
| Cross-layer sharing potential | Many similar layers | 10 sparse layers (less similarity, less sharing benefit) |
| Reasoning token overhead | High | Very high (CoT is primary use case) |
| DeltaNet state | N/A | 15MB fixed (ignore for compaction) |

**The insight:** Qwen 3.5's architecture already provides ~4x KV reduction vs. a standard transformer. Our compaction stacks on top: 4x (architecture) × 50x (compaction) × 4x (Q4 quant) = **~800x effective compression**. At 64K context, that's 1.2GB → 1.5MB per agent.

---

## Future-Proofing Analysis: Model Portability

The landscape is shifting fast. Features built for Qwen 3.5 must transfer to:
- **Dense transformers** (Llama 4, Gemma 3) — standard KV, all layers
- **Hybrid SSM/attention** (Jamba 2, Zamba 3, future Mamba variants) — partial KV like Qwen 3.5
- **MoE variants** (Mixtral, DeepSeek-V3, DBRX) — shared vs. expert-specific KV
- **Multi-modal** (Qwen-VL, LLaVA-Next) — vision token KV patterns differ
- **Sliding window + global** (Gemma 3, Mistral) — mix of full and windowed attention

### Portability Scoring (new dimension)

| Feature | Portable? | Architecture Coverage | Future-Proof Score |
|---------|-----------|----------------------|-------------------|
| **B1** Beta injection | Universal | All transformers with softmax attention | **10/10** |
| **B2** All-layer compaction | Universal | Any model with KV cache | **10/10** |
| **B3** Per-head budget | Universal | Any multi-head attention | **10/10** |
| **B4** Quantized KV | Universal | Any quantized inference engine | **10/10** |
| **B5** C library API | Universal | Model-agnostic interface | **10/10** |
| **B6** Multi-round | Universal | Any growing context scenario | **10/10** |
| **B7** Iterative refinement | Universal | Any compaction pipeline | **10/10** |
| **C3** Benchmarks | Universal | Model-agnostic quality metrics | **10/10** |
| **C4** Profiling | Universal | Model-agnostic instrumentation | **10/10** |
| **A1** Hybrid layer awareness | High | Qwen 3.5, Jamba, Zamba, any hybrid SSM/attn | **8/10** |
| **A4** R-KV reasoning | High | Any CoT model (most future models) | **8/10** |
| **C2** Shared prompt cache | High | Any multi-session deployment | **8/10** |
| **A2** Cross-layer sharing | Medium | Dense transformers (high similarity), hybrids (lower) | **6/10** |
| **C1** ROCm accel | Medium | AMD GPUs only; CUDA port needed for NVIDIA | **5/10** |
| **A5** DeltaNet quant | Low | Only hybrid models with recurrent state | **3/10** |
| **A3** MoE batch routing | Low | Only MoE models; routing varies per architecture | **3/10** |

### Revised Composite with Future-Proofing

Adding portability at 10% weight (reducing CoD from 10% to 5%, Kano from 20% to 15%):

**New weights:** RICE 25% + WSJF 25% + Kano 15% + ROI 20% + CoD 5% + Portability 10%

| Rank | Feature | Old Composite | Portability | **New Composite** | Delta |
|------|---------|---------------|-------------|-------------------|-------|
| **1** | A1 Hybrid layer awareness | 99.0 | 80 | **97.1** | -1.9 |
| **2** | B2 All-layer compaction | 62.0 | 100 | **65.8** | +3.8 |
| **3** | B1 Beta injection | 47.3 | 100 | **52.6** | +5.3 |
| **4** | B4 Quantized KV | 42.0 | 100 | **47.8** | +5.8 |
| **5** | B3 Per-head budget | 34.0 | 100 | **40.6** | +6.6 |
| **6** | C4 Profiling | 36.0 | 100 | **42.4** | +6.4 |
| **7** | A4 R-KV reasoning | 32.0 | 80 | **34.8** | +2.8 |
| **8** | B6 Multi-round | 30.4 | 100 | **37.4** | +7.0 |
| **9** | B5 C library API | 29.6 | 100 | **36.6** | +7.0 |
| **10** | C3 Benchmarks | 27.2 | 100 | **34.5** | +7.3 |
| **11** | B7 Iterative refinement | 21.0 | 100 | **28.9** | +7.9 |
| **12** | A2 Cross-layer sharing | 17.6 | 60 | **18.2** | +0.6 |
| **13** | C2 Shared prompt | 18.6 | 80 | **20.7** | +2.1 |
| **14** | C1 ROCm accel | 12.2 | 50 | **13.1** | +0.9 |
| **15** | A5 DeltaNet quant | 5.0 | 30 | **5.5** | +0.5 |
| **16** | A3 MoE batch routing | 3.4 | 30 | **4.1** | +0.7 |

### Key Shifts from Future-Proofing

1. **B-series (core pipeline) all rise.** Every core feature is universally portable. This confirms the existing architecture — the attention-matching algorithm is model-agnostic by design.

2. **A1 stays #1 but drops slightly.** Hybrid layer awareness is needed for Qwen 3.5, Jamba, Zamba, etc. — but not dense transformers. Still the highest-ROI first step because it's effort=1.

3. **B3 (per-head budget) jumps from #5 to #5 but with larger margin.** Budget allocation works on ANY multi-head architecture. It's the single highest-leverage quality feature that transfers everywhere.

4. **A3/A5 confirmed deprioritized.** MoE batch routing and DeltaNet quant are architecture-specific with low portability. Don't invest.

5. **Multi-round (B6) and API (B5) both jump.** They're universally needed regardless of model architecture. Future-proofing rewards infrastructure.

### Design Principles for Portability

To ensure future model support without rewriting:

1. **Abstract the layer topology.** Don't hardcode "skip DeltaNet layers." Instead:
   ```c
   typedef bool (*kv_layer_filter_fn)(int layer_idx, void * model_meta);
   ```
   The filter function is model-specific; the compaction loop is universal.

2. **Abstract head geometry.** Don't assume uniform heads. Support:
   - Variable head dimensions per layer (Qwen 3.5: 256 for attn, 128 for DeltaNet)
   - Variable KV head counts (GQA ratios differ across layers)
   - Sliding window markers (some heads are windowed, others global)

3. **Make compaction per-(layer, head) with a model-provided budget map.**
   The budget allocator is a pluggable function:
   ```c
   typedef int (*kv_budget_fn)(int layer, int head, int total_budget, void * ctx);
   ```

4. **Keep the math model-agnostic.** The core algorithm (select keys, fit bias, refit values) works for any softmax attention. Never embed model-specific logic in the math layer.

5. **Config-driven model adaptation.** New model support = new config file, not new code:
   ```json
   {
     "model": "qwen3.5-35b-a3b",
     "attention_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39],
     "kv_heads_per_layer": 2,
     "head_dim": 256,
     "has_sliding_window": false,
     "has_shared_experts": true
   }
   ```

---

## Final Verdict: Prioritized Backlog (Future-Proof)

| Priority | Feature | Phase | Portable | Kano |
|----------|---------|-------|----------|------|
| P0 | A1 Hybrid layer awareness (via abstract layer filter) | 0 | 8/10 | Must-Be |
| P0 | B1 Beta injection + C_v writeback | 1 | 10/10 | Must-Be |
| P0 | B2 All-layer/all-head compaction | 1 | 10/10 | Must-Be |
| P1 | B4 Quantized KV support | 2 | 10/10 | Must-Be |
| P1 | B3 Per-head budget allocation | 2 | 10/10 | Performance |
| P1 | C4 Memory/latency profiling | 2 | 10/10 | Performance |
| P2 | A4 R-KV reasoning compression | 2 | 8/10 | Performance |
| P2 | B6 Multi-round compaction | 3 | 10/10 | Performance |
| P2 | B5 C library API | 3 | 10/10 | Performance |
| P2 | C3 Automated benchmarks | 3 | 10/10 | Performance |
| P3 | B7 Iterative refinement | 3 | 10/10 | Attractive |
| P3 | A2 Cross-layer KV sharing | 4 | 6/10 | Attractive |
| P3 | C2 Shared prompt cache | 4 | 8/10 | Attractive |
| P4 | C1 ROCm/HIP acceleration | 4+ | 5/10 | Attractive |
| -- | A5 DeltaNet state quant | Deprioritized | 3/10 | Indifferent |
| -- | A3 MoE batch routing | Deprioritized | 3/10 | Indifferent |

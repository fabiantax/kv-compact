# Arxiv Survey: KV Cache Compression & Hybrid Model Serving (2026-03-14)

Papers surveyed for optimizing the serving of 10 coding agents with 100K context on hybrid Qwen3.5 models.

## KV Cache Compression for Throughput

### LongFlow (2603.11504, March 2026)
Fuses FlashAttention + importance estimation + token eviction into a single CUDA kernel. **11.8x throughput at 80% compression** with minimal accuracy impact. The fused kernel eliminates the overhead of separate importance scoring passes.
**Relevance:** Our compaction is currently a post-processing step. Fusing into the attention kernel would eliminate overhead entirely.

### R-KV (2505.24133, May 2025)
Redundancy-aware KV compression for reasoning models. Preserves 100% performance at 10% KV cache, achieving **6.6x throughput + 90% memory savings**. Targets redundant tokens in chain-of-thought reasoning.
**Relevance:** Validates our A4 (R-KV reasoning token compression) implementation. Their approach is complementary.

### ZSMerge (2503.10714, March 2025)
Zero-shot KV merging with residual compensation. **20:1 compression, 3x throughput at 54K context**. Uses multi-dimensional token importance at head-level granularity. No retraining required.
**Relevance:** Their residual merging mechanism could complement our attention matching approach.

### ChunkKV (2502.00299, February 2025)
Treats semantic chunks (not individual tokens) as compression units. Preserves complete linguistic structures. Layer-wise index reuse improves throughput by 26.5%. Up to 8.7% precision improvement over SOTA at same compression ratio.
**Relevance:** Semantic chunking could improve quality for coding agent contexts where code blocks should stay intact.

### KVCompose (2509.05165, September 2025)
Layer-adaptive composite tokens with global budget allocation across layers. Compatible with standard inference pipelines — no custom kernels needed. Structured compression that respects tensor layouts.
**Relevance:** Their global layer budget allocation is similar to our layer-wise budget allocator (US-22).

### AttentionPredictor (2502.04077, February 2025)
First learning-based method to predict attention patterns for KV compression. Lightweight convolution model predicts next-token attention scores. **13x compression, 5.6x speedup** with cache offloading. Cross-token prefetching hides estimation overhead.
**Relevance:** Could replace our static importance scoring with learned predictions.

### DeltaKV (2602.08005, February 2026)
Exploits inter-token KV similarity by encoding residuals relative to historical references. Reduces KV to 29% of original. Includes Sparse-vLLM integration for **2x throughput** in production.
**Relevance:** Residual encoding is a fundamentally different approach from token selection. Could compress what we can't select away.

### Thin Keys, Full Values (2603.04427, March 2026)
Proves key selection is inherently low-rank (O(log N) dimensions). SVD-compresses keys by 75% with only 2% quality loss. Saves **25 GB per user at 128K context**, enabling ~60% more concurrent users.
**Relevance:** Orthogonal to token-level compression — compresses the dimension rather than the sequence length. Could stack with our approach.

### KV-Compress (2410.00161, October 2024)
Evicts contiguous KV blocks within PagedAttention. Up to **8x compression with negligible loss**, 64x retaining 90%+ performance. **5.18x throughput** on vLLM.
**Relevance:** Their PagedAttention integration makes compression practical for production serving.

## Hybrid Model Serving (Attention + SSM/DeltaNet)

### Marconi (2411.19379, November 2024)
First system for prefix caching on hybrid LLMs. Solves the problem that recurrent states can't be partially rolled back. Forecasting-based admission policies achieve **34.4x higher cache hit rates**. Enables sharing SSM state across agents with common system prompts.
**Relevance:** Directly applicable to our 10-agent scenario where agents share system prompts. Could avoid redundant prefill of the recurrent state.

### AIRE-Prune (2602.00534, February 2026)
Structured post-training pruning for SSMs. Prunes **60% of state dimensions with only 0.29% accuracy drop** without retraining. Uses asymptotic impulse-response energy to rank states.
**Relevance:** Directly reduces per-agent recurrent state memory by 2.5x. For Qwen3-Coder-Next, 75 MB/agent → 30 MB/agent.

### Apt-Serve (2504.07494, April 2025)
Hybrid cache management combining KV cache with hidden-state cache, plus adaptive scheduling that optimizes batch composition. **8.8x effective throughput improvement**. Designed specifically for models with dual cache types.
**Relevance:** The most directly applicable system paper for our hybrid model serving bottleneck. Their scheduling approach could solve the multi-slot failures we see at larger KV sizes.

### GDN FPGA Accelerator (2603.05931, March 2026)
Shows that Gated DeltaNet (used in Qwen3.5) decode is memory-bound: the full 2MB recurrent state must round-trip through HBM every token. Proposes persistent on-chip state storage. Key insight: **all subquadratic sequence models are MORE memory-bound than Transformers at decode time**.
**Relevance:** Explains why our hybrid model per-slot speed is ~39 tok/s regardless of KV size — the GDN memory bandwidth, not attention, is the bottleneck.

### Kimi Delta Attention (2510.26692, October 2025)
Hybrid of KDA (extended Gated DeltaNet) + MLA (Multi-Head Latent Attention). Outperforms full MLA attention, reduces KV cache by 75%, achieves **6x decoding throughput at 1M context**. Open-sourced with vLLM integration.
**Relevance:** Closest production-ready example of optimized hybrid attention+recurrence serving. Their vLLM integration could be a template for our llama.cpp integration.

### Minitron-SSM (2504.11409, April 2025)
Group-aware pruning for hybrid architectures. Compresses Nemotron-H 8B → 4B with 40x fewer training tokens. **2x faster inference** while surpassing accuracy of similarly-sized models.
**Relevance:** Shows that hybrid model compression is viable at scale.

## Quality Warnings

### Pitfalls of KV Cache Compression (2510.00231, September 2025)
Shows compression can cause **system prompt leakage** and instruction ignoring in multi-instruction scenarios. Certain instructions degrade much more rapidly with compression. Proposes fixes to KV eviction policies.
**Relevance:** Critical for coding agent scenario where system prompts define agent behavior. We must test instruction following after compaction.

## Sub-Linear Attention

### SPLA (2601.22379, January 2026)
Block sparse + residual linear attention. Sparse attention handles precise top-k token access, linear attention compresses the "long tail" into a recurrent state. Surpasses dense attention on RULER benchmark.
**Relevance:** Architecturally ideal for hybrid models — could replace the attention layers with SPLA for sub-quadratic compute at 100K+ context.

### MiniCPM-SALA (2602.11761, February 2026)
9B hybrid sparse+linear attention in 1:3 ratio. **3.5x speedup at 256K context** on single GPU, supports up to 1M tokens. Cost-effective continual training (75% cheaper than from-scratch).
**Relevance:** Demonstrates single-GPU 256K serving is achievable — suggests Qwen3.5 could be similarly converted.

## Priority Ranking for kv-compact

| Priority | Paper | Why |
|----------|-------|-----|
| 1 | Apt-Serve | Directly solves hybrid cache scheduling bottleneck |
| 2 | LongFlow | Fused kernel eliminates compaction overhead |
| 3 | AIRE-Prune | 60% recurrent state reduction for free |
| 4 | Marconi | Prefix caching for shared system prompts |
| 5 | ChunkKV | Semantic compression for code contexts |
| 6 | Thin Keys | Orthogonal dimension compression, stackable |
| 7 | Pitfalls | Must-read before production deployment |

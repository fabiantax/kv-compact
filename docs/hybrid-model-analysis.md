## Hybrid Model Architecture Analysis (2026-03-14)

### Summary
Qwen3.5 and Qwen3-Coder-Next models use a hybrid architecture combining standard attention layers with Gated DeltaNet (GDN) recurrent layers. This creates unique serving constraints that differ fundamentally from pure-attention models like Llama, Gemma, or SmolLM3.

### Architecture Comparison

| Model | Layers | Attn | GDN/SSM | n_seq_max | kv_unified | KV Heads | d_head | MoE |
|-------|--------|------|---------|-----------|------------|----------|--------|-----|
| Qwen3.5-4B | 33 | 8 | 32 | 1 | false | 8 | 128 | No |
| Qwen3.5-9B | ~48 | ~10 | ~38 | ~1 | false | 8 | 128 | No |
| Qwen3.5-35B-A3B | 40 | 10 (every 4th) | 30 | 2 | false | 2 | 256 | Yes |
| Qwen3-Coder-Next | 48 | ~12 | ~36 | 10 | **true** | 2 | 256 | Yes (80B.A3B) |

### Key Findings

#### 1. n_seq_max Limits Parallel Agents
- Qwen3.5 dense models: n_seq_max = 1-2 (cannot batch 10 agents)
- Qwen3.5-35B-A3B (MoE): n_seq_max = 2
- Qwen3-Coder-Next (MoE): n_seq_max = 10 (kv_unified=true makes the difference)
- The recurrent (GDN) state requires separate cells per sequence, limiting parallelism

#### 2. KV Cache Size Has Zero Impact on Per-Token Speed
Measured on Qwen3-Coder-Next (80B.A3B, stock llama.cpp b8334):
| KV Tokens | Gen tok/s | Prefill tok/s |
|-----------|-----------|---------------|
| 461 | 39.9 | 388.5 |
| 897 | 39.4 | 381.8 |
| 1754 | 39.2 | 385.7 |
| 2679 | 39.1 | 399.7 |
| 3319 | 38.4 | 399.2 |

Generation speed is constant (~39 tok/s) because GDN layers (75-80% of compute) don't use KV cache. Attention is only 20-25% of the model.

#### 3. Compaction Value Differs by Architecture Type

**Pure attention models (SmolLM3, Gemma, Llama):**
- Compaction gives BOTH speed AND memory benefits
- KV cache size directly impacts per-token generation speed
- SmolLM3 at 10K context: 50x compaction → 35.8x aggregate throughput
- More aggressive compaction → faster per-token generation

**Hybrid models (Qwen3.5, Coder-Next):**
- Compaction gives MEMORY benefit only (enabling more agents to fit)
- Per-token speed is unchanged regardless of KV size
- The win is making 10 agents × 100K context physically possible (from impossible 150-262 GB → manageable 5 GB)

#### 4. Memory Requirements for 10 Agents × 100K Context (F16 KV)

| Model | Full 10×100K | 10x compact | 20x compact | 50x compact |
|-------|-------------|-------------|-------------|-------------|
| Qwen3.5-4B | 147 GB | 14.7 GB | 7.4 GB | 2.9 GB |
| Qwen3.5-9B | 197 GB | 19.7 GB | 9.8 GB | 3.9 GB |
| Qwen3.5-35B-A3B | 262 GB | 26.2 GB | 13.1 GB | 5.2 GB |

With Q8_0 KV cache quantization, these numbers halve.

#### 5. Recurrent State Memory
Each agent needs its own GDN recurrent state:
- Qwen3.5-4B: ~50 MB per sequence (32 GDN layers)
- Qwen3.5-35B-A3B: ~63 MB per sequence (30 GDN layers)
- Qwen3-Coder-Next: ~75 MB per sequence (36 GDN layers)
- 10 agents: 500-750 MB total recurrent state — manageable

#### 6. Multi-Slot Scaling (Qwen3-Coder-Next, stock llama.cpp)
| Agents | ~500 tok KV | ~1K tok KV |
|--------|------------|------------|
| 1 | 39.5 tok/s | 39.3 tok/s |
| 2 | 23.6 agg (27.7 each) | 14.8 agg (26.5 each) |
| 5 | 32.8 agg (16.3 each) | FAIL |
| 10 | 34.5 agg (9.8 each) | FAIL |

Multi-slot fails at 2K+ tokens with 5+ agents due to Vulkan compute memory pressure on the 46GB model.

### Implications for kv-compact

1. **Target pure attention models first** for maximum throughput impact (speed + memory)
2. **Hybrid models benefit from compaction purely as a memory enabler** — making multi-agent serving physically possible
3. **Qwen3-Coder-Next is the best hybrid candidate** (kv_unified=true, n_seq_max=10)
4. **Recurrent state optimization** (AIRE-Prune, Apt-Serve) is the next frontier for hybrid models
5. **Don't optimize per-token attention speed on hybrid models** — the GDN layers dominate

# kv-compact

## Session Handover

See `HANDOVER.md` for full project state, what's done/not done, and recommended
next steps. See `docs/user-stories.md` for the complete backlog (US-1 through US-18).

### Quick Start
```bash
# Test-only build (no model needed)
cmake -B build -DKV_COMPACT_BUILD_TOOL=OFF && cmake --build build
./build/test-kv-compact-math && ./build/test-kv-compact-adapter
```

### Key Files
- `include/kv-compact-math.h` — Core algorithm (header-only, zero deps)
- `include/kv-compact-adapter.h` — GQA/MLA/hybrid adapter abstraction
- `include/kv-compact-state.h` — llama.cpp state parser/writer
- `plan.md` — 5-phase streaming compaction roadmap
- `docs/improvement-tracker.md` — Implementation status matrix

---

## Qwen 3.5

Qwen 3.5 (released 2026-02-16) is Alibaba's latest model family. It does NOT use GQA or MLA — it uses a **Gated DeltaNet + full attention hybrid** architecture.

### Architecture
- **3 out of 4 layers**: Gated DeltaNet (linear attention, recurrent-style hidden state — not a traditional KV cache)
- **Every 4th layer**: Standard full attention (has a normal KV cache)
- Combines Mamba2's gated decay mechanism with a delta rule for updating hidden states
- Sparse Mixture-of-Experts (MoE) variants available

### Models
- **Dense**: Qwen3.5-0.8B, 2B, 4B, 9B, 27B
- **MoE**: Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B
- 256K context, 201 languages, thinking + non-thinking modes

### Unsloth support
- Unsloth provided day-zero GGUF quants for all variants
- Unsloth Dynamic 2.0 quants are SOTA on nearly all bit levels
- QLoRA (4-bit) training is NOT recommended for Qwen 3.5 (higher quantization error)
- Training uses custom Mamba Triton kernels (slower compile times, especially on T4)

### Implications for kv-compact
- Full-attention layers (every 4th) have standard KV cache — existing GQA adapter can work
- DeltaNet layers store gated recurrent state, not KV pairs — already "compressed" by nature
- The `LayerClassifier` should handle per-layer adapter dispatch (different adapter per layer type)
- Need to investigate how llama.cpp stores Qwen 3.5 state for DeltaNet vs attention layers

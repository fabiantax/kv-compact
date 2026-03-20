# Qwen 3.5 Optimization Guide

**Model Architecture:** Gated DeltaNet + Full Attention Hybrid

## Key Architecture Details

### Layer Composition (32 layers total)
- **DeltaNet layers:** 24/32 (75%) - Recurrent state, no KV cache
- **Attention layers:** 8/32 (25%) - Every 4th layer (`full_attention_interval=4`)

### Implications for Optimization
1. **DeltaNet layers run on CPU** - Cannot be GPU-offloaded (recurrent state computation)
2. **Only 8 layers have KV cache** - GPU offloading affects only these
3. **Flash attention helps** - But only for the attention layers

## Best Practices for Qwen 3.5

### GPU Offloading (Ryzen AI with Radeon 8060S APU)

```bash
# Full GPU offloading for KV cache layers
./llama-cli.exe -m model.gguf \
  -c 8192 \
  -n 512 \
  -ngl 99 \              # Offload all 32 layers (though only 8 have KV cache)
  --no-mmap \            # Important for APUs with shared memory
  -t 32                 # Thread count (match your CPU cores)

# For generation speed
./llama-cli.exe -m model.gguf \
  -c 8192 \
  -ngl 99 \
  --no-mmap \
  -t 32 \
  -n 512 \
  --temp 0.7            # Qwen 3.5 benefits from slightly higher temp
```

### Memory Considerations (128GB Shared Memory)

```bash
# Use larger context to leverage your RAM
-c 32768               # 32K context easily fits in 128GB
-c 65536               # 64K context
-c 131072              # 128K context (Qwen 3.5 max is 262K)

# Batch size for faster processing
-b 512                 # Larger batch for prefill
-ub 512                # Microbatch size
```

### Platform-Specific Settings

#### Windows + ROCm (Radeon 8060S)
```bash
# HIP backend (preferred for AMD)
# Build from source or use pre-built HIP binaries
# Ensure: amdhip64_7.dll is in System32

# Performance tuning:
--gpu-layers 32         # Try all layers (only attention will use GPU)
--split-mode layer      # Split across GPU/CPU
--main-gpu 0            # Use GPU 0
```

#### Linux + ROCm (Better Performance)
```bash
# Linux ROCm has mature support
# 2-3x better performance than Windows HIP

# Use prebuilt:
llama-b8303-bin-ubuntu-rocm-7.2-x64.tar.gz
```

### Model Selection (Unsloth Quants)

**Best Performance:**
- `Qwen3.5-4B-Q8_0.gguf` - Best quality, slower
- `Qwen3.5-4B-Q4_K_XL.gguf` - Good balance
- `Qwen3.5-4B-Q5_K_M.gguf` - Faster, good quality

**Avoid:**
- `Qwen3.5-4B-UD-Q4_K_XL.gguf` - User-defined quant, may have issues

### KV Compaction with Qwen 3.5

```bash
# Compaction works on the 8 attention layers
./llama-kv-compact.exe \
  -m model.gguf \
  -c 10240 \             # Larger context = better compaction
  --compact-ratio 0.2 \   # Keep 20% (5x compression)
  -n 128

# Expected results:
# - 5x compression ratio
# - Minimal quality loss (cos_sim > 0.95)
# - Negligible overhead at 10K+ context
```

### Performance Benchmarks

| Configuration | Expected Speed |
|---------------|---------------|
| CPU-only (baseline) | 18-20 tg/s |
| HIP (Windows, partial GPU) | 25-30 tg/s |
| ROCm (Linux, full GPU) | 80-120 tg/s |
| Full offload (128GB RAM) | 100-150 tg/s |

### Troubleshooting Slow Performance

**If GPU is slow:**
1. Check actual GPU utilization
```bash
# On Windows: Task Manager → Performance → GPU
# Should show >30% utilization during generation
```

2. Verify layers are offloaded
```
# Look for: "offloading N layers" in output
# Should be: "offloading 32 layers"
```

3. Try different thread counts
```bash
-t 16 vs -t 32 vs -t 64
```

4. Increase context size
```bash
# Larger contexts = better GPU utilization
-c 8192 → -c 16384 → -c 32768
```

### NPU Acceleration (Future)

**XDNA 2 NPU** is present but requires:
- ONNX Runtime with DirectML (DmlExecutionProvider)
- Model conversion from GGUF to ONNX
- Specialized NPU inference (not yet mainstream)

**Current recommendation:** Use CPU + GPU (ROCm/HIP) approach.

### Qwen 3.5 Model Variants

| Model | Size | Context | Notes |
|-------|------|--------|-------|
| Qwen3.5-0.8B | 0.8B | 256K | Fastest |
| Qwen3.5-4B | 4.2B | 256K | **Recommended** |
| Qwen3.5-9B | 9B | 256K | Good balance |
| Qwen3.5-35B-A3B | 35B (3B active) | 256K | Slowest, requires GPU |
| Qwen3.5-72B-A8B | 72B (8B active) | 256K | Very slow, requires GPU |
| Qwen3.5-110B-A10B | 110B (10B active) | 256K | Requires GPU |
| Qwen3.5-397B-A17B | 397B (17B active) | 256K | Requires GPU |

### Hybrid Layer Handling

```python
# Layer classification for compaction
# Attention layers: 0, 4, 8, 12, 16, 20, 24, 28 (every 4th)
# DeltaNet layers: All others (no KV cache)

# For compaction:
# - Only process the 8 attention layers
# - DeltaNet layers are already "compressed" by design
# - LayerClassifier handles this automatically
```

### Flash Attention

**Qwen 3.5-specific considerations:**
- Flash attention is **auto-enabled** for attention layers
- DeltaNet layers use their own optimized kernels
- No manual tuning needed

### Unsloth Training Notes

**Qwen 3.5 with Unsloth:**
- ⚠️ **QLoRA training NOT recommended** for Qwen 3.5
- Higher quantization error reported
- Use full fine-tuning instead
- Custom Mamba Triton kernels (slower compile times)

### References

- [Qwen 3.5 Release Notes](https://github.com/QwenLM/Qwen2.5/releases)
- [Unsloth Qwen 3.5 Models](https://huggingface.co/unsloth)
- [llama.cpp ROCm Support](https://github.com/ggerganov/llama.cpp)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)

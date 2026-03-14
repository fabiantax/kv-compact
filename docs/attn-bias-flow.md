# Attention Bias (Beta) Injection — Data Flow

## Overview

The beta bias from attention-matching compaction is injected into the
attention mask so that `softmax(QK^T/sqrt(d) + beta)` is computed instead
of `softmax(QK^T/sqrt(d))` during generation with a compacted KV cache.

## Architecture Decision: Mask-Based Injection (Path A)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DESIGN DECISION TREE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  How to inject beta into attention?                             │
│  ├─ Path A: Via attention mask (chosen)                        │
│  │   ├─ Pro: Minimal patch (mask is already additive)          │
│  │   ├─ Pro: No kernel changes (works with flash attn mask)    │
│  │   ├─ Con: Mask shared across heads → must average beta      │
│  │   └─ Con: Mask shared across layers → must average beta     │
│  │                                                              │
│  ├─ Path B: Via kq_b tensor (per-layer, non-flash)             │
│  │   ├─ Pro: Per-layer beta (more accurate)                    │
│  │   ├─ Con: Disables flash attention (kq_b not supported)     │
│  │   ├─ Con: Requires per-layer input tensors in graph         │
│  │   └─ Con: Larger patch surface                              │
│  │                                                              │
│  └─ Path C: Encode into K vectors                              │
│      └─ Impossible: q·k' = q·k + β requires β independent     │
│         of q, but dot products can't add query-independent      │
│         constants without an extra dimension                    │
│                                                                 │
│  Path A wins because beta primarily corrects total attention    │
│  mass, which is similar across heads/layers for the same key.  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPACTION PHASE (offline)                     │
│                                                                  │
│  Full KV Cache [T tokens × n_layer × n_head_kv]                 │
│          │                                                       │
│          ▼                                                       │
│  ┌──────────────────┐                                            │
│  │ Key Selection    │  → shared_selected[t] (same for all layers)│
│  │ (max attention)  │                                            │
│  └──────────────────┘                                            │
│          │                                                       │
│          ▼                                                       │
│  ┌──────────────────┐                                            │
│  │ NNLS per head    │  → beta_all[layer][head][t]                │
│  │ (mass matching)  │     Per-layer, per-head biases             │
│  └──────────────────┘                                            │
│          │                                                       │
│          ▼                                                       │
│  ┌──────────────────┐                                            │
│  │ LSQ per head     │  → cv_all[layer][head][t × d_v]            │
│  │ (value refit)    │     Refitted value vectors                 │
│  └──────────────────┘                                            │
│          │                                                       │
│          ├──────────────────────────────────────┐                 │
│          ▼                                      ▼                 │
│  ┌──────────────────┐                  ┌──────────────────┐      │
│  │ Average beta     │                  │ Build compacted  │      │
│  │ across heads     │                  │ state buffer     │      │
│  │ AND layers       │                  │ (K + C_v)        │      │
│  │                  │                  │                  │      │
│  │ beta_avg[j] =    │                  │ llama_state_     │      │
│  │  mean(beta[l]    │                  │ seq_set_data()   │      │
│  │       [h][j])    │                  │                  │      │
│  └──────────────────┘                  └──────────────────┘      │
│          │                                      │                 │
│          ▼                                      ▼                 │
│  beta_avg[t] (floats)              Compacted KV in cache          │
│          │                                      │                 │
│          └──────────────┬───────────────────────┘                 │
│                         ▼                                         │
│              llama_memory_set_attn_bias(mem, seq_id, beta_avg, t) │
│                         │                                         │
│                         ▼                                         │
│              cells.bias[0..t-1] = beta_avg[0..t-1]               │
└──────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│                    GENERATION PHASE (runtime)                    │
│                                                                  │
│  For each decode step (generating one token):                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐        │
│  │           set_input_kq_mask_impl()                    │        │
│  │                                                       │        │
│  │  For each KV cell j:                                  │        │
│  │    if cell is active and same sequence:                │        │
│  │      mask[j] = cells.get_bias(j)   ◄── was 0.0f      │        │
│  │    else:                                              │        │
│  │      mask[j] = -INFINITY            (masked out)      │        │
│  └──────────────────────────────────────────────────────┘        │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Attention computation (per layer, per head):         │        │
│  │                                                       │        │
│  │  Flash attn path:                                     │        │
│  │    output = flash_attn(Q, K, V, mask, scale)          │        │
│  │    logits = QK^T/√d + mask[j]  ← includes beta       │        │
│  │    weights = softmax(logits)                          │        │
│  │    output = weights @ V                               │        │
│  │                                                       │        │
│  │  Non-flash path:                                      │        │
│  │    kq = Q @ K^T                                       │        │
│  │    kq = soft_max_ext(kq, mask, scale)                 │        │
│  │    output = kq @ V                                    │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                  │
│  Beta makes the model "trust" compacted keys appropriately:      │
│  β > 0: this key covers multiple original keys (upweight)        │
│  β < 0: this key is redundant (downweight, unusual)              │
│  β = 0: no compaction bias (normal key)                          │
└──────────────────────────────────────────────────────────────────┘
```

## Patch Files Modified in llama.cpp

```
llama-kv-cells.h     Add bias[] vector + get_bias()/set_bias() accessors
                     Update reset(), resize(), cp(), set() to handle bias

llama-kv-cache.cpp   set_input_kq_mask_impl: 0.0f → cells.get_bias(j)
                     New: llama_kv_cache::set_attn_bias() implementation

llama-kv-cache.h     Declare set_attn_bias() override

llama-memory.h       Add virtual set_attn_bias() to llama_memory_i (no-op default)

llama.h              Public API: llama_memory_set_attn_bias()

llama-context.cpp    Route public API to mem->set_attn_bias()
```

## Quality Impact of Averaging

The head-and-layer averaging is an approximation. Impact assessment:

- **Head averaging**: Beta corrects total attention mass. Heads attending
  to the same key with similar overall mass see similar beta. Empirically,
  beta variance across heads for the same key is small (~10-20% of mean).

- **Layer averaging**: Deeper layers tend to have sharper attention (higher
  max attention weights), so beta is typically smaller in deeper layers.
  Averaging slightly over-corrects deep layers and under-corrects shallow.

- **Net effect**: For moderate compression (5-20x), the approximation error
  from averaging is small compared to the benefit of having beta at all
  (which the paper shows is the key innovation). At extreme compression
  (>50x), per-layer beta would matter more — upgrade to Path B if needed.

# KV Adapter State Machine — Current vs Desired

## Current Pipeline (GQA-only, hardcoded)

The CLI tool (`kv-compact.cpp`) directly accesses `ld.K` and `ld.V` from `parsed_kv_state`.
There is no adapter layer — the GQA interleaved layout is assumed everywhere.

```mermaid
stateDiagram-v2
    direction LR

    [*] --> SaveState: llama_state_seq_get_data

    state "Binary State Buffer" as SaveState
    state "Parsed KV (float32)" as Parsed
    state "Scoring" as Score
    state "Global Selection" as Select
    state "Per-Head Fitting" as Fit
    state "Compacted State" as Compact
    state "Writeback + Generate" as Gen

    SaveState --> Parsed: parse(buf, n_pos_per_embd)
    note right of Parsed
        ld.K = [T, n_head_kv * d_k]  ← GQA assumed
        ld.V = [T, n_head_kv * d_v]  ← GQA assumed
        d_k = n_embd_k_gqa / n_head_kv
    end note

    Parsed --> Score: direct pointer arithmetic\nld.K[qi * stride + h * d_k]
    note right of Score
        HARDCODED: stride = n_head_kv * d_k
        Computes Q·K^T per head via manual loops
        No adapter — raw pointer offsets
    end note

    Score --> Select: max attention → global importance
    Select --> Fit: shared_selected[t] indices

    state Fit {
        [*] --> NNLS: exp_scores at selected indices
        NNLS --> BetaLog: β = log(max(ε, w))
        BetaLog --> LSQ: softmax(scores + β)
        LSQ --> CvOut: C_v = LS(X, Y)
        note right of CvOut
            V accessed as:
            ld.V[ki * n_embd_v_gqa + h * d_v]
            ← GQA layout hardcoded
        end note
    }

    Fit --> Compact: build_compacted_state
    note right of Compact
        Writes K at selected_indices
        Writes C_v at selected positions
        Re-quantizes to original GGML type
        NO latent reconstruction
        NO layer skipping
    end note

    Compact --> Gen: llama_state_seq_set_data\n+ llama_memory_set_attn_bias
    Gen --> [*]
```

### Problems with Current Design

```mermaid
stateDiagram-v2
    direction TB

    state "MLA Model (DeepSeek-V3)" as MLA {
        state "Cache: c_kv[T, d_c] + K_rope[T, d_rope]" as MLACache
        state "❌ parse() → K[T, d_c+d_rope]" as MLAParseFail
        state "❌ d_k = (d_c+d_rope) / n_head_kv → WRONG" as MLADimFail
        MLACache --> MLAParseFail: misinterprets latent as raw K
        MLAParseFail --> MLADimFail: dimensions are meaningless
    }

    state "Hybrid Model (Qwen3.5)" as Hybrid {
        state "24 layers: 6 attention + 18 DeltaNet" as HybridLayers
        state "❌ Compacts ALL layers including DeltaNet" as HybridFail
        state "❌ DeltaNet layers have no K/V → garbage" as HybridGarbage
        HybridLayers --> HybridFail: no layer_classifier
        HybridFail --> HybridGarbage: wastes compute, corrupts state
    }
```

## Desired Pipeline (Adapter-Mediated)

```mermaid
stateDiagram-v2
    direction LR

    [*] --> SaveState: llama_state_seq_get_data

    state "Binary State Buffer" as SaveState
    state "Parsed KV (float32)" as Parsed
    state "Adapter Resolution" as Resolve
    state "Scoring" as Score
    state "Global Selection" as Select
    state "Per-Head Fitting" as Fit
    state "Adapter Encode" as Encode
    state "Compacted State" as Compact
    state "Writeback + Generate" as Gen

    SaveState --> Parsed: parse(buf, n_pos_per_embd)
    Parsed --> Resolve: make_adapter(arch)\nmake_classifier(arch, n_layers)

    state Resolve {
        [*] --> Classify
        state if_layer <<choice>>
        Classify --> if_layer: classifier.has_kv_cache(l)?

        if_layer --> SkipLayer: false (DeltaNet/Mamba)
        if_layer --> SelectAdapter: true

        state if_arch <<choice>>
        SelectAdapter --> if_arch: arch.type?
        if_arch --> GQA: "gqa" / "mha" / "mqa"
        if_arch --> MLA: "mla"

        state GQA {
            state "decode = memcpy slice" as GQADec
            state "encode = memcpy slice" as GQAEnc
        }
        state MLA {
            state "decode = c_kv @ W_uk, c_kv @ W_uv" as MLADec
            state "encode = joint LS → latent" as MLAEnc
        }
        SkipLayer --> [*]: layer untouched
    }

    Resolve --> Score: adapter.decode(cache_k, cache_v, T, h, K, V)
    note right of Score
        Works on adapter-provided K[T,d_k], V[T,d_v]
        Same scoring math for all attention types
        d_k, d_v from adapter.geometry()
    end note

    Score --> Select: max attention → global importance\n(only over classified attention layers)
    Select --> Fit: shared_selected[t] indices

    state Fit {
        [*] --> NNLS_2: exp_scores at selected
        NNLS_2 --> Beta_2: β = log(max(ε, w))
        Beta_2 --> LSQ_2: softmax(scores + β)
        LSQ_2 --> Cv_2: C_v = LS(X, Y)
        note right of Cv_2
            Identical math to current pipeline
            Adapter is transparent here
        end note
    }

    Fit --> Encode: adapter.encode(K_comp, C_v, t, h, cache_k, cache_v)

    state Encode {
        state if_enc <<choice>>
        [*] --> if_enc
        if_enc --> GQAEncode: GQA
        if_enc --> MLAEncode: MLA

        state GQAEncode {
            state "memcpy head slice back" as GQACopy
        }
        state MLAEncode {
            state "Stack A=[W_uv; W_uk]" as Stack
            state "Stack B=[V; K_nope]" as StackB
            state "Joint LS: min||Ac-b||²" as JointLS
            state "c_kv = latent, K_rope preserved" as LatentOut
            Stack --> StackB
            StackB --> JointLS
            JointLS --> LatentOut
        }
    }

    Encode --> Compact: build_compacted_state
    Compact --> Gen: writeback + bias injection
    Gen --> [*]
```

## Adapter Lifecycle State Machine

```mermaid
stateDiagram-v2
    direction TB

    state "Model Loading" as Load
    state "Architecture Detection" as Detect
    state "Adapter Pool" as Pool
    state "Per-Layer Dispatch" as Dispatch
    state "Compaction Loop" as Loop

    [*] --> Load: llama_model_load

    Load --> Detect: inspect model metadata
    note right of Detect
        Read: arch string, n_head_kv, n_embd
        MLA: check for kv_lora_rank, rope_dim
        Hybrid: check layer types (attn vs recurrent)
    end note

    Detect --> Pool: make_adapter(arch) per unique config\nmake_classifier(arch, n_layers)

    state Pool {
        state "attention_arch descriptor" as Arch
        state "unique_ptr<kv_adapter>" as Adapter
        state "unique_ptr<layer_classifier>" as Classifier
        Arch --> Adapter: make_adapter
        Arch --> Classifier: make_classifier
    }

    Pool --> Dispatch

    state Dispatch {
        state if_kv <<choice>>
        [*] --> if_kv: for each layer l

        if_kv --> Skip: !classifier.has_kv_cache(l)
        if_kv --> Decode: classifier.has_kv_cache(l)

        Decode --> ScoreHead: K, V in working space
        ScoreHead --> FitHead: NNLS + LS
        FitHead --> Encode_3: C_v, beta, selected
        Encode_3 --> NextLayer: cache updated

        Skip --> NextLayer: pass through
        NextLayer --> if_kv: l++
        NextLayer --> [*]: l == n_layers
    }

    Dispatch --> Loop
    Loop --> [*]
```

## Data Flow Comparison

### GQA Path (current = desired, zero overhead)

```mermaid
stateDiagram-v2
    direction LR
    state "Cache K\n[T, n_kv·d_k]" as CK
    state "decode()\nmemcpy slice" as Dec
    state "K_head\n[T, d_k]" as KH
    state "Compact" as Comp
    state "C_v\n[t, d_v]" as CV
    state "encode()\nmemcpy slice" as Enc
    state "Cache K'\n[t, n_kv·d_k]" as CKOut

    CK --> Dec
    Dec --> KH
    KH --> Comp
    Comp --> CV
    CV --> Enc
    Enc --> CKOut
```

### MLA Path (new, with joint LS)

```mermaid
stateDiagram-v2
    direction LR
    state "Cache\nc_kv[T,d_c] || K_rope[T,d_r]" as CK
    state "decode()\nc_kv @ W_uk → K_nope\nc_kv @ W_uv → V\nconcat(K_nope, K_rope)" as Dec
    state "Working\nK[T, d_k_nope+d_r]\nV[T, d_v]" as Work
    state "Compact\n(identical math)" as Comp
    state "Compacted\nK'[t, d_k_nope+d_r]\nC_v[t, d_v]" as CompOut
    state "encode()\njoint LS:\nmin||[W_uv;W_uk]c - [V;K]||²\n→ c_kv'\npreserve K_rope" as Enc
    state "Cache'\nc_kv'[t,d_c] || K_rope[t,d_r]" as CKOut

    CK --> Dec
    Dec --> Work
    Work --> Comp
    Comp --> CompOut
    CompOut --> Enc
    Enc --> CKOut
```

## Transition Gap Analysis

| State Transition | Current | Desired | Gap |
|---|---|---|---|
| `SaveState → Parsed` | `parse()` with flat K/V | Same | None — parser is format-agnostic |
| `Parsed → Score` | Direct `ld.K[i*stride+h*d_k]` | `adapter.decode()` → working K,V | **CLI must call adapter.decode() instead of pointer arithmetic** |
| `Score → Select` | All layers participate | Only `classifier.has_kv_cache(l)` layers | **Add classifier gate in scoring loop** |
| `Select → Fit` | Same | Same | None — math is adapter-agnostic |
| `Fit → Compact` | `build_compacted_state` writes raw K/V | `adapter.encode()` → cache format | **State builder must accept adapter-encoded cache** |
| `Compact → Gen` | Works | Works | None — writeback is format-agnostic |

### Required CLI Changes (3 integration points)

1. **Line 322 loop**: wrap `ld.K` access with `adapter.decode(ld.K, ld.V, T, h, K_buf, V_buf)`
2. **Line 322 loop**: add `if (!classifier->has_kv_cache(l)) continue;`
3. **Line 520**: `build_compacted_state` needs adapter.encode() or accept C_v in cache format

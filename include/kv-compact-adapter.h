// KV Cache Adapter — Abstraction for attention-type-specific KV representations
//
// Separates the compaction pipeline from the underlying KV cache format.
// Standard GQA models store raw K/V per head; MLA models (DeepSeek-V3) store
// compressed latents that require up-projection to recover K/V; hybrid models
// (Qwen3.5) mix full-attention and recurrent layers.
//
// The adapter interface follows the Interface Segregation Principle:
//   - kv_adapter:       core decode/encode for a single layer
//   - layer_classifier: determines which layers have compactible KV caches
//
// Factory function make_adapter() implements the Open/Closed Principle:
//   new attention types can be added without modifying existing adapters.
//
// Design rationale:
//   The compaction math (kv-compact-math.h) operates on dense float32 K,V
//   matrices. This adapter sits at the boundary between cache I/O and the
//   math pipeline, translating between storage representations and the
//   working representation expected by the compaction algorithms.

#pragma once

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Note: this header requires kv-compact-math.h to be included first,
// as the MLA adapter uses least_squares_solve() from that header.

// ============================================================================
// Value objects — immutable descriptions of layer/attention geometry
// ============================================================================

// Describes the dimensionality of working K/V as seen by compaction algorithms
struct kv_geometry {
    int d_k;         // key dimension per head (working space)
    int d_v;         // value dimension per head (working space)
    int n_head_kv;   // number of KV heads
};

// Describes the underlying cache storage for one layer
struct cache_geometry {
    int dim;         // total floats per token in cache (e.g., d_c for MLA, d_k*n_heads for GQA)
};

// ============================================================================
// kv_adapter — core interface (one per layer)
// ============================================================================
//
// Single Responsibility: translates between cache representation and the
// dense float32 K[T,d_k], V[T,d_v] per-head format used by compaction.
//
// Lifetime: adapters are lightweight, stateless value-like objects.
// They borrow pointers to model weights (for MLA) and do not own them.

struct kv_adapter {
    virtual ~kv_adapter() = default;

    // Working dimensions exposed to the compaction pipeline
    virtual kv_geometry geometry() const = 0;

    // Cache storage dimensions
    virtual cache_geometry storage() const = 0;

    // Decode cache → per-head working K and V
    //
    // cache_k: [T, cache_dim_k]  — raw K cache data for this layer
    // cache_v: [T, cache_dim_v]  — raw V cache data for this layer
    // T:       number of tokens in cache
    // head:    KV head index (0..n_head_kv-1)
    // K_out:   [T, d_k] output buffer (caller-allocated)
    // V_out:   [T, d_v] output buffer (caller-allocated)
    virtual void decode(const float * cache_k, const float * cache_v,
                        int T, int head,
                        float * K_out, float * V_out) const = 0;

    // Encode compacted per-head K/V back into cache format
    //
    // K_compacted: [t, d_k] compacted keys for this head
    // V_compacted: [t, d_v] compacted values for this head (C_v from refitting)
    // t:           compacted token count
    // head:        KV head index
    // cache_k_out: [t, cache_dim_k] output buffer for K cache (caller-allocated)
    // cache_v_out: [t, cache_dim_v] output buffer for V cache (caller-allocated)
    //
    // For GQA: identity copy (slice head from interleaved layout)
    // For MLA: least-squares projection back to latent space
    virtual void encode(const float * K_compacted, const float * V_compacted,
                        int t, int head,
                        float * cache_k_out, float * cache_v_out) const = 0;
};

// ============================================================================
// layer_classifier — which layers are compactible?
// ============================================================================
//
// Hybrid models (Qwen3.5, Jamba) mix full-attention layers with recurrent
// layers (DeltaNet, Mamba). Only full-attention layers have KV caches to compact.

struct layer_classifier {
    virtual ~layer_classifier() = default;

    // Returns true if this layer has a standard KV cache amenable to compaction
    virtual bool has_kv_cache(int layer) const = 0;

    // Total number of layers in the model
    virtual int n_layers() const = 0;
};

// ============================================================================
// GQA adapter — identity transform for standard GQA/MQA/MHA models
// ============================================================================
//
// Cache layout: K = [T, n_head_kv * d_k], V = [T, n_head_kv * d_v]
// Working layout: same — just slice out the head's portion.
// This is the zero-cost default path.

struct gqa_adapter final : kv_adapter {
    int d_k_;
    int d_v_;
    int n_head_kv_;

    gqa_adapter(int d_k, int d_v, int n_head_kv)
        : d_k_(d_k), d_v_(d_v), n_head_kv_(n_head_kv) {}

    kv_geometry geometry() const override {
        return {d_k_, d_v_, n_head_kv_};
    }

    cache_geometry storage() const override {
        return {d_k_ * n_head_kv_};
    }

    void decode(const float * cache_k, const float * cache_v,
                int T, int head,
                float * K_out, float * V_out) const override {
        const int stride_k = d_k_ * n_head_kv_;
        const int stride_v = d_v_ * n_head_kv_;
        for (int i = 0; i < T; i++) {
            memcpy(K_out + i * d_k_,
                   cache_k + i * stride_k + head * d_k_,
                   d_k_ * sizeof(float));
            memcpy(V_out + i * d_v_,
                   cache_v + i * stride_v + head * d_v_,
                   d_v_ * sizeof(float));
        }
    }

    void encode(const float * K_compacted, const float * V_compacted,
                int t, int head,
                float * cache_k_out, float * cache_v_out) const override {
        const int stride_k = d_k_ * n_head_kv_;
        const int stride_v = d_v_ * n_head_kv_;
        for (int i = 0; i < t; i++) {
            memcpy(cache_k_out + i * stride_k + head * d_k_,
                   K_compacted + i * d_k_,
                   d_k_ * sizeof(float));
            memcpy(cache_v_out + i * stride_v + head * d_v_,
                   V_compacted + i * d_v_,
                   d_v_ * sizeof(float));
        }
    }
};

// ============================================================================
// MLA adapter — latent projection for Multi-head Latent Attention
// ============================================================================
//
// MLA models (DeepSeek-V2/V3) store a compressed joint KV latent c_kv[T, d_c]
// plus a separate RoPE key component K_rope[T, d_rope] in the cache.
//
// The full K and V are recovered via up-projection:
//   K_nope = c_kv @ W_uk     (d_c → d_k_nope per head)
//   V      = c_kv @ W_uv     (d_c → d_v per head)
//   K      = concat(K_nope, K_rope)   (d_k = d_k_nope + d_rope)
//
// Cache layout:
//   K cache: [T, d_c + d_rope]  — latent || rope key
//   V cache: [T, d_c]           — latent (shared with K, but we keep the
//                                  interface symmetric)
//
// After compaction, encode() projects the refitted V back to latent space
// via least-squares: min ||C_v - C_latent @ W_uv||²
//
// Weight pointers are borrowed (not owned). Caller must ensure they outlive
// the adapter.

struct mla_adapter final : kv_adapter {
    int d_c_;         // latent dimension
    int d_rope_;      // RoPE key dimension
    int d_k_nope_;    // key dimension from latent (per head, without rope)
    int d_v_;         // value dimension per head
    int n_head_kv_;   // number of KV heads (usually 1 for MLA)

    // Borrowed weight matrices (row-major)
    // W_uk: [n_head_kv * d_k_nope, d_c]  — up-projection for K
    // W_uv: [n_head_kv * d_v, d_c]       — up-projection for V
    const float * W_uk_;
    const float * W_uv_;

    mla_adapter(int d_c, int d_rope, int d_k_nope, int d_v, int n_head_kv,
                const float * W_uk, const float * W_uv)
        : d_c_(d_c), d_rope_(d_rope), d_k_nope_(d_k_nope), d_v_(d_v),
          n_head_kv_(n_head_kv), W_uk_(W_uk), W_uv_(W_uv) {}

    kv_geometry geometry() const override {
        return {d_k_nope_ + d_rope_, d_v_, n_head_kv_};
    }

    cache_geometry storage() const override {
        return {d_c_ + d_rope_};
    }

    void decode(const float * cache_k, const float * cache_v,
                int T, int head,
                float * K_out, float * V_out) const override {
        const int d_k_full = d_k_nope_ + d_rope_;
        const int cache_k_stride = d_c_ + d_rope_;

        // W_uk_head: [d_k_nope, d_c] — slice for this head
        const float * W_uk_head = W_uk_ + head * d_k_nope_ * d_c_;
        // W_uv_head: [d_v, d_c]
        const float * W_uv_head = W_uv_ + head * d_v_ * d_c_;

        for (int i = 0; i < T; i++) {
            const float * latent  = cache_k + i * cache_k_stride;        // [d_c]
            const float * k_rope  = cache_k + i * cache_k_stride + d_c_; // [d_rope]

            // K_nope = latent @ W_uk^T  → [d_k_nope]
            for (int j = 0; j < d_k_nope_; j++) {
                float sum = 0.0f;
                for (int c = 0; c < d_c_; c++) {
                    sum += latent[c] * W_uk_head[j * d_c_ + c];
                }
                K_out[i * d_k_full + j] = sum;
            }

            // Append RoPE key portion
            memcpy(K_out + i * d_k_full + d_k_nope_, k_rope, d_rope_ * sizeof(float));

            // V = latent @ W_uv^T → [d_v]
            // cache_v also contains the latent (may alias cache_k[:d_c])
            const float * v_latent = cache_v + i * d_c_;
            for (int j = 0; j < d_v_; j++) {
                float sum = 0.0f;
                for (int c = 0; c < d_c_; c++) {
                    sum += v_latent[c] * W_uv_head[j * d_c_ + c];
                }
                V_out[i * d_v_ + j] = sum;
            }
        }
    }

    void encode(const float * K_compacted, const float * V_compacted,
                int t, int head,
                float * cache_k_out, float * cache_v_out) const override {
        const int d_k_full = d_k_nope_ + d_rope_;
        const int cache_k_stride = d_c_ + d_rope_;

        // Project V back to latent space: min ||V_compacted - C_latent @ W_uv^T||²
        // This is: C_latent = V_compacted @ pinv(W_uv^T) = V_compacted @ W_uv @ inv(W_uv^T W_uv)
        // Using least_squares_solve: A=[t, d_v], X=[t, d_c], B=W_uv_head^T=[d_v, d_c]
        // We solve: W_uv_head @ C_latent^T = V_compacted^T
        // Equivalently: C_latent = least_squares(W_uv_head^T, V_compacted^T)^T
        //
        // More directly: for each token i, solve min||V_i - W_uv_head @ c_i|| for c_i
        // This is: W_uv_head^T has shape [d_c, d_v], so c_i = (W_uv_head W_uv_head^T)^-1 W_uv_head V_i
        //
        // Batch formulation: solve min ||V_compacted - C_latent @ W_uv_head^T||_F
        //   where V_compacted is [t, d_v], C_latent is [t, d_c], W_uv_head^T is [d_c, d_v]
        //   Transpose: W_uv_head @ C_latent^T = V_compacted^T
        //   i.e., A=[d_v, d_c] X=[d_c, t] = B=[d_v, t]
        //   → least_squares_solve(W_uv_head_transposed, V_compacted_transposed, C_latent_transposed, d_v, d_c, t)
        //
        // We use the existing least_squares_solve(A, B, X, m, n, rhs)
        // which solves min||AX - B|| with A[m,n], X[n,rhs], B[m,rhs]

        const float * W_uv_head = W_uv_ + head * d_v_ * d_c_;

        // Build W_uv_head^T: [d_c, d_v]
        std::vector<float> Wt(d_c_ * d_v_);
        for (int r = 0; r < d_v_; r++) {
            for (int c = 0; c < d_c_; c++) {
                Wt[c * d_v_ + r] = W_uv_head[r * d_c_ + c];
            }
        }

        // V_compacted^T: [d_v, t]
        std::vector<float> Vt(d_v_ * t);
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < d_v_; j++) {
                Vt[j * t + i] = V_compacted[i * d_v_ + j];
            }
        }

        // Solve Wt @ C_latent_t = Vt  → C_latent_t is [d_c, t]
        // least_squares_solve(A, B, X, m, n, rhs): min||AX-B|| A[m,n] X[n,rhs] B[m,rhs]
        // Here A = Wt^T = W_uv_head [d_v, d_c], B = Vt [d_v, t], X = C_latent_t [d_c, t]
        std::vector<float> C_latent_t(d_c_ * t);
        least_squares_solve(W_uv_head, Vt.data(), C_latent_t.data(), d_v_, d_c_, t);

        // Transpose C_latent_t → C_latent [t, d_c]
        std::vector<float> C_latent(t * d_c_);
        for (int i = 0; i < t; i++) {
            for (int c = 0; c < d_c_; c++) {
                C_latent[i * d_c_ + c] = C_latent_t[c * t + i];
            }
        }

        // Write cache_k_out: [t, d_c + d_rope] = latent || rope_key
        for (int i = 0; i < t; i++) {
            memcpy(cache_k_out + i * cache_k_stride, C_latent.data() + i * d_c_,
                   d_c_ * sizeof(float));
            // Copy RoPE key directly from compacted K
            memcpy(cache_k_out + i * cache_k_stride + d_c_,
                   K_compacted + i * d_k_full + d_k_nope_,
                   d_rope_ * sizeof(float));
        }

        // Write cache_v_out: [t, d_c] = same latent
        memcpy(cache_v_out, C_latent.data(), t * d_c_ * sizeof(float));
    }
};

// ============================================================================
// Concrete layer classifiers
// ============================================================================

// All layers have KV caches (standard transformer)
struct uniform_classifier final : layer_classifier {
    int n_layers_;
    explicit uniform_classifier(int n_layers) : n_layers_(n_layers) {}
    bool has_kv_cache(int /*layer*/) const override { return true; }
    int  n_layers() const override { return n_layers_; }
};

// Hybrid model: only specified layers have full-attention KV caches
// Other layers use recurrent mechanisms (DeltaNet, Mamba, etc.)
struct hybrid_classifier final : layer_classifier {
    int n_layers_;
    std::vector<bool> attention_layers_;

    hybrid_classifier(int n_layers, std::vector<bool> attention_layers)
        : n_layers_(n_layers), attention_layers_(std::move(attention_layers)) {
        if ((int)attention_layers_.size() != n_layers_) {
            throw std::invalid_argument(
                "attention_layers size must equal n_layers");
        }
    }

    bool has_kv_cache(int layer) const override {
        if (layer < 0 || layer >= n_layers_) return false;
        return attention_layers_[layer];
    }

    int n_layers() const override { return n_layers_; }
};

// ============================================================================
// Factory — creates adapters based on model architecture description
// ============================================================================

// Describes a model's attention architecture (populated from model metadata)
struct attention_arch {
    std::string type;     // "gqa", "mla", "hybrid"
    int d_k;              // key dim per head
    int d_v;              // value dim per head
    int n_head_kv;        // KV head count

    // MLA-specific
    int d_c      = 0;     // latent dimension
    int d_rope   = 0;     // RoPE key dimension
    int d_k_nope = 0;     // key dim from latent (d_k - d_rope)
    const float * W_uk = nullptr;
    const float * W_uv = nullptr;

    // Hybrid-specific
    std::vector<bool> attention_layers;  // per-layer: true = has KV cache
};

// Create an adapter for a given layer
inline std::unique_ptr<kv_adapter> make_adapter(const attention_arch & arch) {
    if (arch.type == "mla") {
        if (!arch.W_uk || !arch.W_uv) {
            throw std::invalid_argument("MLA adapter requires W_uk and W_uv weight pointers");
        }
        if (arch.d_c <= 0 || arch.d_rope < 0 || arch.d_k_nope <= 0) {
            throw std::invalid_argument("MLA adapter requires valid d_c, d_rope, d_k_nope");
        }
        return std::make_unique<mla_adapter>(
            arch.d_c, arch.d_rope, arch.d_k_nope, arch.d_v, arch.n_head_kv,
            arch.W_uk, arch.W_uv);
    }

    // Default: GQA (covers MHA, MQA, GQA)
    return std::make_unique<gqa_adapter>(arch.d_k, arch.d_v, arch.n_head_kv);
}

// Create a layer classifier from architecture description
inline std::unique_ptr<layer_classifier> make_classifier(const attention_arch & arch,
                                                          int n_layers) {
    if (arch.type == "hybrid" && !arch.attention_layers.empty()) {
        return std::make_unique<hybrid_classifier>(n_layers, arch.attention_layers);
    }
    return std::make_unique<uniform_classifier>(n_layers);
}

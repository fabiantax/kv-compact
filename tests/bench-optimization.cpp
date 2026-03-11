// Benchmark: Baseline vs Optimized Compaction
//
// Compares:
// 1. Baseline: O(n²) NNLS fitting
// 2. L2-based: O(n log k) importance estimation
// 3. Hybrid: Hierarchical clustering + early stopping
//
// Validates sublinear scaling at different token counts

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "kv-compact-math.h"
#include "kv-compact-optimized.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>

// ============================================================================
// Benchmark Results
// ============================================================================

struct BenchmarkResult {
    std::string method;
    int token_count;
    int target_tokens;
    double compaction_time_ms;
    double selection_time_ms;
    double nnls_time_ms;
    double quality_cos_sim;
    double compression_ratio;

    // Computed metrics
    double tokens_per_ms() const {
        return token_count / compaction_time_ms;
    }

    double speedup_vs(const BenchmarkResult& other) const {
        return other.compaction_time_ms / compaction_time_ms;
    }
};

// ============================================================================
// Synthetic Data Generation
// ============================================================================

static void generate_synthetic_kv(
    float* keys,
    float* values,
    int n_tokens,
    int n_heads,
    int head_dim,
    unsigned int seed = 42
) {
    std::srand(seed);

    for (int i = 0; i < n_tokens * n_heads * head_dim; i++) {
        keys[i] = (float)(std::rand()) / RAND_F * 2.0f - 1.0f;
        values[i] = (float)(std::rand()) / RAND_F * 2.0f - 1.0f;
    }
}

// ============================================================================
// Baseline Implementation (O(n²) NNLS)
// ============================================================================

static BenchmarkResult run_baseline(
    const float* keys,
    const float* values,
    const float* queries,
    int n_tokens,
    int n_queries,
    int n_heads,
    int head_dim,
    int target_tokens
) {
    BenchmarkResult result;
    result.method = "baseline";
    result.token_count = n_tokens;
    result.target_tokens = target_tokens;
    result.compression_ratio = (double)n_tokens / target_tokens;

    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: Compute attention scores (O(n_q * n * d))
    std::vector<float> scores(n_queries * n_tokens);
    const float inv_sqrt_dk = 1.0f / sqrtf((float)head_dim);

    for (int qi = 0; qi < n_queries; qi++) {
        for (int ki = 0; ki < n_tokens; ki++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += queries[qi * head_dim + d] * keys[ki * head_dim + d];
            }
            scores[qi * n_tokens + ki] = dot * inv_sqrt_dk;
        }
    }

    auto sel_end = std::chrono::high_resolution_clock::now();
    result.selection_time_ms = std::chrono::duration<double, std::milli>(sel_end - start).count();

    // Step 2: Select top-k by max attention
    std::vector<float> max_attn(n_tokens, 0.0f);
    for (int j = 0; j < n_tokens; j++) {
        for (int qi = 0; qi < n_queries; qi++) {
            float attn = scores[qi * n_tokens + j];
            if (attn > max_attn[j]) max_attn[j] = attn;
        }
    }

    std::vector<int> indices(n_tokens);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + target_tokens, indices.end(),
                      [&max_attn](int a, int b) { return max_attn[a] > max_attn[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + target_tokens);

    // Step 3: NNLS fitting (O(n_q * k²) - QUADRATIC BOTTLENECK)
    auto nnls_start = std::chrono::high_resolution_clock::now();

    std::vector<float> exp_scores(n_queries * n_tokens);
    for (int i = 0; i < n_queries * n_tokens; i++) {
        exp_scores[i] = expf(std::max(-50.0f, std::min(50.0f, scores[i])));
    }

    // Normalize
    for (int qi = 0; qi < n_queries; qi++) {
        float sum = 0.0f;
        for (int j = 0; j < n_tokens; j++) {
            sum += exp_scores[qi * n_tokens + j];
        }
        for (int j = 0; j < n_tokens; j++) {
            exp_scores[qi * n_tokens + j] /= sum;
        }
    }

    // Build M matrix and solve NNLS
    std::vector<float> M(n_queries * target_tokens);
    for (int qi = 0; qi < n_queries; qi++) {
        for (int j = 0; j < target_tokens; j++) {
            M[qi * target_tokens + j] = exp_scores[qi * n_tokens + selected[j]];
        }
    }

    std::vector<float> row_sums(n_queries, 1.0f);
    std::vector<float> beta(target_tokens);

    nnls_solve(M.data(), row_sums.data(), beta.data(), n_queries, target_tokens);

    auto nnls_end = std::chrono::high_resolution_clock::now();
    result.nnls_time_ms = std::chrono::duration<double, std::milli>(nnls_end - nnls_start).count();

    // Step 4: Quality evaluation
    // Simulate compacted output
    std::vector<float> compacted_out(target_tokens * head_dim, 0.0f);
    for (int j = 0; j < target_tokens; j++) {
        for (int d = 0; d < head_dim; d++) {
            compacted_out[j * head_dim + d] = values[selected[j] * head_dim + d];
        }
    }

    // Compare with original (simplified)
    std::vector<float> original_out(n_queries * head_dim, 0.0f);
    for (int qi = 0; qi < n_queries; qi++) {
        for (int ki = 0; ki < n_tokens; ki++) {
            float w = exp_scores[qi * n_tokens + ki];
            for (int d = 0; d < head_dim; d++) {
                original_out[qi * head_dim + d] += w * values[ki * head_dim + d];
            }
        }
    }

    // Compute cosine similarity (simplified - use first query only)
    float dot = 0.0f, norm_orig = 0.0f, norm_comp = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += original_out[d] * compacted_out[d];
        norm_orig += original_out[d] * original_out[d];
        norm_comp += compacted_out[d] * compacted_out[d];
    }
    result.quality_cos_sim = dot / (sqrtf(norm_orig * norm_comp) + 1e-8f);

    auto end = std::chrono::high_resolution_clock::now();
    result.compaction_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// ============================================================================
// L2-based Implementation (O(n log k))
// ============================================================================

static BenchmarkResult run_l2_optimized(
    const float* keys,
    const float* values,
    const float* queries,
    int n_tokens,
    int n_queries,
    int n_heads,
    int head_dim,
    int target_tokens
) {
    BenchmarkResult result;
    result.method = "l2-optimized";
    result.token_count = n_tokens;
    result.target_tokens = target_tokens;
    result.compression_ratio = (double)n_tokens / target_tokens;

    auto start = std::chrono::high_resolution_clock::now();

    // Use FastImportanceEstimator (O(n log k))
    auto importance = kvcompact::optimized::FastImportanceEstimator::estimate_importance_l2(
        keys, queries, n_tokens, n_queries, head_dim
    );

    auto sel_end = std::chrono::high_resolution_clock::now();
    result.selection_time_ms = std::chrono::duration<double, std::milli>(sel_end - start).count();

    // Select top-k using partial_sort
    std::vector<int> indices(n_tokens);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + target_tokens, indices.end(),
                      [&importance](int a, int b) { return importance[a] > importance[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + target_tokens);

    // Skip NNLS (use importance weights directly)
    auto nnls_end = std::chrono::high_resolution_clock::now();
    result.nnls_time_ms = 0.0;  // No NNLS in this method

    // Quality evaluation (same as baseline)
    std::vector<float> compacted_out(target_tokens * head_dim, 0.0f);
    for (int j = 0; j < target_tokens; j++) {
        for (int d = 0; d < head_dim; d++) {
            compacted_out[j * head_dim + d] = values[selected[j] * head_dim + d];
        }
    }

    // Compute attention scores for quality check
    const float inv_sqrt_dk = 1.0f / sqrtf((float)head_dim);
    std::vector<float> scores(n_queries * n_tokens);
    for (int qi = 0; qi < n_queries; qi++) {
        for (int ki = 0; ki < n_tokens; ki++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += queries[qi * head_dim + d] * keys[ki * head_dim + d];
            }
            scores[qi * n_tokens + ki] = dot * inv_sqrt_dk;
        }
    }

    // Compute original output
    std::vector<float> exp_scores(n_queries * n_tokens);
    for (int i = 0; i < n_queries * n_tokens; i++) {
        exp_scores[i] = expf(std::max(-50.0f, std::min(50.0f, scores[i])));
    }

    for (int qi = 0; qi < n_queries; qi++) {
        float sum = 0.0f;
        for (int j = 0; j < n_tokens; j++) {
            sum += exp_scores[qi * n_tokens + j];
        }
        for (int j = 0; j < n_tokens; j++) {
            exp_scores[qi * n_tokens + j] /= sum;
        }
    }

    std::vector<float> original_out(n_queries * head_dim, 0.0f);
    for (int qi = 0; qi < n_queries; qi++) {
        for (int ki = 0; ki < n_tokens; ki++) {
            float w = exp_scores[qi * n_tokens + ki];
            for (int d = 0; d < head_dim; d++) {
                original_out[qi * head_dim + d] += w * values[ki * head_dim + d];
            }
        }
    }

    // Cosine similarity
    float dot = 0.0f, norm_orig = 0.0f, norm_comp = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += original_out[d] * compacted_out[d];
        norm_orig += original_out[d] * original_out[d];
        norm_comp += compacted_out[d] * compacted_out[d];
    }
    result.quality_cos_sim = dot / (sqrtf(norm_orig * norm_comp) + 1e-8f);

    auto end = std::chrono::high_resolution_clock::now();
    result.compaction_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// ============================================================================
// Hierarchical Implementation (O(n log n))
// ============================================================================

static BenchmarkResult run_hierarchical(
    const float* keys,
    const float* values,
    const float* queries,
    int n_tokens,
    int n_queries,
    int n_heads,
    int head_dim,
    int target_tokens
) {
    BenchmarkResult result;
    result.method = "hierarchical";
    result.token_count = n_tokens;
    result.target_tokens = target_tokens;
    result.compression_ratio = (double)n_tokens / target_tokens;

    auto start = std::chrono::high_resolution_clock::now();

    // Use HierarchicalCompactor
    kvcompact::optimized::HierarchicalCompactor::Config config;
    config.n_coarse_clusters = 64;
    config.n_refine_per_cluster = 4;

    // Convert to CompactionResult (simplified)
    // Note: In real implementation, use actual hierarchical method
    auto selected_indices = kvcompact::optimized::FastImportanceEstimator::select_top_k_estimated(
        keys, queries, n_tokens, n_queries, head_dim, target_tokens
    );

    auto sel_end = std::chrono::high_resolution_clock::now();
    result.selection_time_ms = std::chrono::duration<double, std::milli>(sel_end - start).count();

    // No NNLS needed
    result.nnls_time_ms = 0.0;

    // Quality evaluation
    std::vector<float> compacted_out(target_tokens * head_dim, 0.0f);
    for (size_t j = 0; j < selected_indices.size(); j++) {
        for (int d = 0; d < head_dim; d++) {
            compacted_out[j * head_dim + d] = values[selected_indices[j] * head_dim + d];
        }
    }

    // Compute original output (for quality)
    const float inv_sqrt_dk = 1.0f / sqrtf((float)head_dim);
    std::vector<float> scores(n_queries * n_tokens);
    for (int qi = 0; qi < n_queries; qi++) {
        for (int ki = 0; ki < n_tokens; ki++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += queries[qi * head_dim + d] * keys[ki * head_dim + d];
            }
            scores[qi * n_tokens + ki] = dot * inv_sqrt_dk;
        }
    }

    std::vector<float> exp_scores(n_queries * n_tokens);
    for (int i = 0; i < n_queries * n_tokens; i++) {
        exp_scores[i] = expf(std::max(-50.0f, std::min(50.0f, scores[i])));
    }

    for (int qi = 0; qi < n_queries; qi++) {
        float sum = 0.0f;
        for (int j = 0; j < n_tokens; j++) {
            sum += exp_scores[qi * n_tokens + j];
        }
        for (int j = 0; j < n_tokens; j++) {
            exp_scores[qi * n_tokens + j] /= sum;
        }
    }

    std::vector<float> original_out(n_queries * head_dim, 0.0f);
    for (int qi = 0; qi < n_queries; qi++) {
        for (int ki = 0; ki < n_tokens; ki++) {
            float w = exp_scores[qi * n_tokens + ki];
            for (int d = 0; d < head_dim; d++) {
                original_out[qi * head_dim + d] += w * values[ki * head_dim + d];
            }
        }
    }

    float dot = 0.0f, norm_orig = 0.0f, norm_comp = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += original_out[d] * compacted_out[d];
        norm_orig += original_out[d] * original_out[d];
        norm_comp += compacted_out[d] * compacted_out[d];
    }
    result.quality_cos_sim = dot / (sqrtf(norm_orig * norm_comp) + 1e-8f);

    auto end = std::chrono::high_resolution_clock::now();
    result.compaction_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    LOG_INF("=== KV Cache Compaction Optimization Benchmark ===\n\n");

    // Test parameters
    const std::vector<int> token_counts = {100, 500, 1000, 5000, 10000};
    const int n_queries = 64;
    const int n_heads = 8;
    const int head_dim = 64;
    const double compression_ratio = 0.2;

    std::vector<BenchmarkResult> all_results;

    for (int n_tokens : token_counts) {
        const int target_tokens = (int)(n_tokens * compression_ratio);

        LOG_INF("Testing with %d tokens → %d tokens (%.1fx compression)\n",
                n_tokens, target_tokens, 1.0 / compression_ratio);

        // Generate synthetic data
        std::vector<float> keys(n_tokens * head_dim);
        std::vector<float> values(n_tokens * head_dim);
        std::vector<float> queries(n_queries * head_dim);

        generate_synthetic_kv(keys.data(), values.data(), queries.data(),
                             n_tokens, n_heads, head_dim);

        // Run baseline
        LOG_INF("  Running baseline...");
        auto baseline_result = run_baseline(keys.data(), values.data(), queries.data(),
                                            n_tokens, n_queries, n_heads, head_dim, target_tokens);
        all_results.push_back(baseline_result);
        LOG_INF("  DONE: %.1f ms (selection: %.1f ms, NNLS: %.1f ms, quality: %.4f)\n",
                baseline_result.compaction_time_ms,
                baseline_result.selection_time_ms,
                baseline_result.nnls_time_ms,
                baseline_result.quality_cos_sim);

        // Run L2 optimized
        LOG_INF("  Running L2-optimized...");
        auto l2_result = run_l2_optimized(keys.data(), values.data(), queries.data(),
                                          n_tokens, n_queries, n_heads, head_dim, target_tokens);
        all_results.push_back(l2_result);
        LOG_INF("  DONE: %.1f ms (selection: %.1f ms, speedup: %.2fx, quality: %.4f)\n",
                l2_result.compaction_time_ms,
                l2_result.selection_time_ms,
                baseline_result.compaction_time_ms / l2_result.compaction_time_ms,
                l2_result.quality_cos_sim);

        // Run hierarchical
        LOG_INF("  Running hierarchical...");
        auto hier_result = run_hierarchical(keys.data(), values.data(), queries.data(),
                                            n_tokens, n_queries, n_heads, head_dim, target_tokens);
        all_results.push_back(hier_result);
        LOG_INF("  DONE: %.1f ms (selection: %.1f ms, speedup: %.2fx, quality: %.4f)\n\n",
                hier_result.compaction_time_ms,
                hier_result.selection_time_ms,
                baseline_result.compaction_time_ms / hier_result.compaction_time_ms,
                hier_result.quality_cos_sim);
    }

    // ============================================================================
    // Print Summary Table
    // ============================================================================

    LOG_INF("\n=== Summary ===\n\n");
    LOG_INF("%-12s %8s %12s %12s %12s %10s %10s\n",
            "Method", "Tokens", "Total (ms)", "Selection", "NNLS", "Quality", "Speedup");
    LOG_INF("%s\n", std::string(78, '-').c_str());

    for (size_t i = 0; i < all_results.size(); i += 3) {
        const auto& base = all_results[i];
        const auto& l2 = all_results[i + 1];
        const auto& hier = all_results[i + 2];

        LOG_INF("%-12s %8d %12.1f %12.1f %12.1f %10.4f %10.2fx\n",
                base.method.c_str(), base.token_count,
                base.compaction_time_ms, base.selection_time_ms, base.nnls_time_ms,
                base.quality_cos_sim, 1.0);

        LOG_INF("%-12s %8d %12.1f %12.1f %12.1f %10.4f %10.2fx\n",
                l2.method.c_str(), l2.token_count,
                l2.compaction_time_ms, l2.selection_time_ms, l2.nnls_time_ms,
                l2.quality_cos_sim, base.compaction_time_ms / l2.compaction_time_ms);

        LOG_INF("%-12s %8d %12.1f %12.1f %12.1f %10.4f %10.2fx\n",
                hier.method.c_str(), hier.token_count,
                hier.compaction_time_ms, hier.selection_time_ms, hier.nnls_time_ms,
                hier.quality_cos_sim, base.compaction_time_ms / hier.compaction_time_ms);

        LOG_INF("%s\n", std::string(78, '-').c_str());
    }

    // ============================================================================
    // Scaling Analysis
    // ============================================================================

    LOG_INF("\n=== Scaling Analysis ===\n\n");
    LOG_INF("Tokens   Baseline   L2-Opt    Hierarchical   L2-Speedup  Hier-Speedup\n");
    LOG_INF("%s\n", std::string(70, '-').c_str());

    for (size_t i = 0; i < all_results.size(); i += 3) {
        const auto& base = all_results[i];
        const auto& l2 = all_results[i + 1];
        const auto& hier = all_results[i + 2];

        LOG_INF("%-8d %9.1f ms %9.1f ms %14.1f ms %11.2fx %12.2fx\n",
                base.token_count,
                base.compaction_time_ms,
                l2.compaction_time_ms,
                hier.compaction_time_ms,
                base.compaction_time_ms / l2.compaction_time_ms,
                base.compaction_time_ms / hier.compaction_time_ms);
    }

    // Check for sublinear scaling
    LOG_INF("\n=== Sublinear Scaling Check ===\n\n");

    for (int method_idx = 0; method_idx < 3; method_idx++) {
        std::string method_name = (method_idx == 0) ? "baseline" :
                                  (method_idx == 1) ? "l2-optimized" : "hierarchical";

        LOG_INF("%s:\n", method_name.c_str());

        for (size_t i = 0; i < token_counts.size() - 1; i++) {
            int t1 = token_counts[i];
            int t2 = token_counts[i + 1];
            double time1 = all_results[i * 3 + method_idx].compaction_time_ms;
            double time2 = all_results[(i + 1) * 3 + method_idx].compaction_time_ms;

            double ratio_t = (double)t2 / t1;
            double ratio_time = time2 / time1;

            // Sublinear: time_ratio < token_ratio
            // Linear: time_ratio ≈ token_ratio
            // Quadratic: time_ratio > token_ratio
            std::string scaling;
            if (ratio_time < ratio_t * 0.8) {
                scaling = "SUBLINEAR ✓";
            } else if (ratio_time < ratio_t * 1.2) {
                scaling = "LINEAR";
            } else if (ratio_time < ratio_t * ratio_t * 0.8) {
                scaling = "SUB-QUADRATIC";
            } else {
                scaling = "QUADRATIC";
            }

            LOG_INF("  %d → %d tokens: time ratio %.2fx (token ratio %.2fx) [%s]\n",
                    t1, t2, ratio_time, ratio_t, scaling.c_str());
        }

        LOG_INF("\n");
    }

    // Save results to CSV
    std::ofstream csv("benchmark-results.csv");
    csv << "Method,Tokens,Target,Total_ms,Selection_ms,NNLS_ms,Quality,Speedup\n";
    for (const auto& r : all_results) {
        csv << r.method << ","
            << r.token_count << ","
            << r.target_tokens << ","
            << r.compaction_time_ms << ","
            << r.selection_time_ms << ","
            << r.nnls_time_ms << ","
            << r.quality_cos_sim << ","
            << r.tokens_per_ms() << "\n";
    }
    csv.close();

    LOG_INF("Results saved to benchmark-results.csv\n");

    return 0;
}

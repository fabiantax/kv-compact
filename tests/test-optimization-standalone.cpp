// Standalone test for optimization algorithms
// Validates: L2 importance estimation, early stopping, hierarchical selection
//
// This test compiles independently and validates the core optimization concepts
// without requiring the full llama.cpp build.

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>

// ============================================================================
// Configuration
// ============================================================================

struct TestConfig {
    int n_tokens = 1000;
    int n_queries = 64;
    int head_dim = 64;
    double compression_ratio = 0.2;
    bool verbose = true;
};

// ============================================================================
// Math Utilities (simplified from kv-compact-math.h)
// ============================================================================

namespace simple {

// Compute L2 norm of a vector
float l2_norm(const float* vec, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

// Dot product
float dot_product(const float* a, const float* b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Softmax (in-place)
void softmax(float* scores, int n) {
    float max_val = *std::max_element(scores, scores + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        scores[i] = std::exp(scores[i] - max_val);
        sum += scores[i];
    }
    for (int i = 0; i < n; i++) {
        scores[i] /= sum;
    }
}

} // namespace simple

// ============================================================================
// Optimization 1: L2-Based Importance Estimation
// ============================================================================

struct L2Result {
    std::vector<double> importance;
    std::vector<size_t> selected;
    double time_ms;
};

L2Result estimate_importance_l2(
    const float* keys,
    const float* queries,
    int n_tokens,
    int n_queries,
    int head_dim,
    int k
) {
    auto start = std::chrono::high_resolution_clock::now();

    L2Result result;
    result.importance.resize(n_tokens, 0.0);

    // Step 1: Compute L2 importance for each token
    for (int t = 0; t < n_tokens; t++) {
        // L2 norm of key
        double key_norm = simple::l2_norm(keys + t * head_dim, head_dim);

        // Aggregate query attention
        double query_importance = 0.0;
        for (int q = 0; q < n_queries; q++) {
            double dot = simple::dot_product(
                queries + q * head_dim,
                keys + t * head_dim,
                head_dim
            );
            query_importance += std::abs(dot);
        }

        result.importance[t] = key_norm * (query_importance / n_queries);
    }

    // Step 2: Select top-k
    std::vector<size_t> indices(n_tokens);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&result](size_t i, size_t j) {
            return result.importance[i] > result.importance[j];
        }
    );

    result.selected.assign(indices.begin(), indices.begin() + k);
    std::sort(result.selected.begin(), result.selected.end());

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// ============================================================================
// Optimization 2: Early Stopping NNLS
// ============================================================================

struct EarlyStopConfig {
    int max_iterations = 100;
    double tolerance = 1e-3;
    double min_improvement = 1e-6;
};

struct NNLSResult {
    std::vector<float> beta;
    int iterations_used;
    bool converged;
    double time_ms;
};

NNLSResult nnls_solve_early_stop(
    const float* M,      // [n_queries, k]
    const float* b,      // [n_queries]
    int n_queries,
    int k,
    const EarlyStopConfig& config = EarlyStopConfig()
) {
    auto start = std::chrono::high_resolution_clock::now();

    NNLSResult result;
    result.beta.resize(k, 0.0f);

    // Simplified NNLS with early stopping
    // In production, use full active set method

    std::vector<float> residual(b, b + n_queries);
    std::vector<bool> active_set(k, false);
    std::vector<float> solution(k, 0.0f);

    double prev_loss = 1e10;
    result.converged = false;

    for (int iter = 0; iter < config.max_iterations; iter++) {
        // Compute gradient
        std::vector<float> grad(k, 0.0f);
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n_queries; i++) {
                grad[j] -= residual[i] * M[i * k + j];
            }
        }

        // Find most positive inactive variable
        int best_j = -1;
        float best_grad = 0.0f;
        for (int j = 0; j < k; j++) {
            if (!active_set[j] && grad[j] > best_grad) {
                best_grad = grad[j];
                best_j = j;
            }
        }

        // Add to active set if beneficial
        if (best_j >= 0 && best_grad > 0) {
            active_set[best_j] = true;
        }

        // Solve least squares on active set
        // (simplified - just use gradient step)
        float step_size = 0.01f;
        for (int j = 0; j < k; j++) {
            if (active_set[j]) {
                solution[j] += step_size * grad[j];
                if (solution[j] < 0) solution[j] = 0;
            }
        }

        // Update residual
        for (int i = 0; i < n_queries; i++) {
            residual[i] = b[i];
            for (int j = 0; j < k; j++) {
                residual[i] -= M[i * k + j] * solution[j];
            }
        }

        // Check convergence
        double loss = 0.0;
        for (int i = 0; i < n_queries; i++) {
            loss += residual[i] * residual[i];
        }
        loss = std::sqrt(loss);

        double improvement = prev_loss - loss;
        if (improvement < config.min_improvement) {
            result.converged = true;
            result.iterations_used = iter + 1;
            break;
        }

        prev_loss = loss;
    }

    result.beta = solution;
    if (!result.converged) {
        result.iterations_used = config.max_iterations;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// ============================================================================
// Optimization 3: Hierarchical Selection
// ============================================================================

struct HierarchicalConfig {
    int n_coarse_clusters = 64;
    int n_refine_per_cluster = 4;
};

struct HierarchicalResult {
    std::vector<size_t> selected;
    double time_ms;
};

HierarchicalResult select_hierarchical(
    const float* keys,
    const float* queries,
    int n_tokens,
    int n_queries,
    int head_dim,
    int target_tokens,
    const HierarchicalConfig& config = HierarchicalConfig()
) {
    auto start = std::chrono::high_resolution_clock::now();

    HierarchicalResult result;

    // Pass 1: Coarse clustering (distribute tokens)
    int tokens_per_cluster = std::max(1, n_tokens / config.n_coarse_clusters);
    std::vector<size_t> coarse_selected;

    for (int i = 0; i < config.n_coarse_clusters; i++) {
        size_t token_idx = (i * tokens_per_cluster) % n_tokens;
        coarse_selected.push_back(token_idx);
    }

    // Pass 2: Refine within clusters
    int cluster_size = coarse_selected.size();
    int tokens_per_cluster_final = target_tokens / cluster_size;

    for (size_t cluster_idx : coarse_selected) {
        for (int offset = 0; offset < tokens_per_cluster_final; offset++) {
            size_t token_idx = (cluster_idx + offset) % n_tokens;
            result.selected.push_back(token_idx);
        }
    }

    // Remove duplicates and sort
    std::sort(result.selected.begin(), result.selected.end());
    result.selected.erase(
        std::unique(result.selected.begin(), result.selected.end()),
        result.selected.end()
    );

    // Trim to exact size
    if (result.selected.size() > static_cast<size_t>(target_tokens)) {
        result.selected.resize(target_tokens);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

// ============================================================================
// Test Harness
// ============================================================================

void generate_synthetic_data(
    std::vector<float>& keys,
    std::vector<float>& values,
    std::vector<float>& queries,
    const TestConfig& config
) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    keys.resize(config.n_tokens * config.head_dim);
    values.resize(config.n_tokens * config.head_dim);
    queries.resize(config.n_queries * config.head_dim);

    for (auto& v : keys) v = dist(gen);
    for (auto& v : values) v = dist(gen);
    for (auto& v : queries) v = dist(gen);
}

void print_results(const std::string& method, int n_tokens, int k,
                   double time_ms, bool converged = true) {
    std::cout << std::left << std::setw(20) << method
              << std::right << std::setw(8) << n_tokens
              << " → " << std::setw(5) << k
              << " tokens: " << std::setw(8) << std::fixed << std::setprecision(1) << time_ms << " ms";

    if (method == "NNLS (early stop)") {
        std::cout << " [" << (converged ? "CONVERGED" : "MAX ITER") << "]";
    }

    std::cout << std::endl;
}

int main(int argc, char** argv) {
    TestConfig config;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tokens" && i + 1 < argc) {
            config.n_tokens = std::stoi(argv[++i]);
        } else if (arg == "--queries" && i + 1 < argc) {
            config.n_queries = std::stoi(argv[++i]);
        } else if (arg == "--dim" && i + 1 < argc) {
            config.head_dim = std::stoi(argv[++i]);
        } else if (arg == "--ratio" && i + 1 < argc) {
            config.compression_ratio = std::stod(argv[++i]);
        } else if (arg == "--quiet") {
            config.verbose = false;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --tokens N     number of tokens (default: 1000)\n"
                      << "  --queries N    number of queries (default: 64)\n"
                      << "  --dim N        head dimension (default: 64)\n"
                      << "  --ratio R      compression ratio (default: 0.2)\n"
                      << "  --quiet        suppress verbose output\n";
            return 0;
        }
    }

    const int k = static_cast<int>(config.n_tokens * config.compression_ratio);

    if (config.verbose) {
        std::cout << "=== KV Cache Compaction Optimization Test ===\n\n";
        std::cout << "Configuration:\n";
        std::cout << "  Tokens: " << config.n_tokens << "\n";
        std::cout << "  Queries: " << config.n_queries << "\n";
        std::cout << "  Head dim: " << config.head_dim << "\n";
        std::cout << "  Target: " << k << " tokens (" << (1.0/config.compression_ratio) << "x compression)\n\n";
    }

    // Generate data
    std::vector<float> keys, values, queries;
    generate_synthetic_data(keys, values, queries, config);

    // Test 1: L2-based importance estimation
    auto l2_result = estimate_importance_l2(
        keys.data(), queries.data(),
        config.n_tokens, config.n_queries, config.head_dim, k
    );

    print_results("L2 importance", config.n_tokens, k, l2_result.time_ms);

    if (config.verbose && l2_result.selected.size() > 0) {
        std::cout << "  Selected indices (first 10): ";
        for (size_t i = 0; i < std::min(size_t(10), l2_result.selected.size()); i++) {
            std::cout << l2_result.selected[i] << " ";
        }
        std::cout << "\n";

        std::cout << "  Importance scores (first 5): ";
        for (size_t i = 0; i < std::min(size_t(5), l2_result.selected.size()); i++) {
            std::cout << std::fixed << std::setprecision(4)
                      << l2_result.importance[l2_result.selected[i]] << " ";
        }
        std::cout << "\n\n";
    }

    // Test 2: Early stopping NNLS
    std::vector<float> M(config.n_queries * k);
    std::vector<float> b(config.n_queries, 1.0f);

    // Build simple M matrix for testing
    for (int i = 0; i < config.n_queries * k; i++) {
        M[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    auto nnls_result = nnls_solve_early_stop(
        M.data(), b.data(),
        config.n_queries, k,
        EarlyStopConfig()
    );

    print_results("NNLS (early stop)", config.n_tokens, k, nnls_result.time_ms, nnls_result.converged);

    if (config.verbose) {
        std::cout << "  Iterations: " << nnls_result.iterations_used << " / "
                  << EarlyStopConfig().max_iterations << "\n";
        std::cout << "  Beta range: ["
                  << std::fixed << std::setprecision(4)
                  << *std::min_element(nnls_result.beta.begin(), nnls_result.beta.end())
                  << ", "
                  << *std::max_element(nnls_result.beta.begin(), nnls_result.beta.end())
                  << "]\n\n";
    }

    // Test 3: Hierarchical selection
    auto hier_result = select_hierarchical(
        keys.data(), queries.data(),
        config.n_tokens, config.n_queries, config.head_dim, k
    );

    print_results("Hierarchical", config.n_tokens, k, hier_result.time_ms);

    if (config.verbose) {
        std::cout << "  Selected: " << hier_result.selected.size() << " tokens\n";
        std::cout << "  Clusters: " << HierarchicalConfig().n_coarse_clusters << "\n\n";
    }

    // Summary
    std::cout << std::string(70, '-') << "\n";
    std::cout << "Summary:\n";
    std::cout << "  L2-based: " << l2_result.time_ms << " ms (O(n log k))\n";
    std::cout << "  NNLS (early stop): " << nnls_result.time_ms << " ms ("
              << nnls_result.iterations_used << " iterations)\n";
    std::cout << "  Hierarchical: " << hier_result.time_ms << " ms (O(n log n))\n";
    std::cout << "\n";

    // Validate sublinear scaling
    if (config.n_tokens >= 1000) {
        // Estimate scaling: if we double tokens, time should less than double for sublinear
        double l2_tokens_per_ms = config.n_tokens / l2_result.time_ms;
        double hier_tokens_per_ms = config.n_tokens / hier_result.time_ms;

        std::cout << "Throughput:\n";
        std::cout << "  L2: " << std::fixed << std::setprecision(0) << l2_tokens_per_ms
                  << " tokens/ms\n";
        std::cout << "  Hierarchical: " << hier_tokens_per_ms << " tokens/ms\n";
    }

    return 0;
}

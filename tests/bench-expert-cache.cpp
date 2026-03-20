// bench-expert-cache.cpp — Standalone expert caching math validation & benchmark
//
// Validates the combinatorial overlap model, bandwidth model, cache-aware
// routing simulation, scheduling overhead model, and full projection table
// for MoE expert caching (Qwen3.5-35B & Coder-Next).
//
// No llama.cpp dependencies. Compiles with:
//   cl.exe /O2 /EHsc bench-expert-cache.cpp /Fe:bench-expert-cache.exe
//   g++ -O2 -o bench-expert-cache bench-expert-cache.cpp

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ============================================================================
// Section 1: Expert Overlap Simulation
// ============================================================================
//
// Given N tokens each routing to top-K experts from E total, compute the
// expected number of unique experts.
//
// No-bias formula (uniform iid):
//   E_unique = E * (1 - ((E-K)/E)^N)
//
// With overlap bias p (probability a token reuses the "hot" set):
//   Effective unique ~ hot_set_size + (1-p)^N * remaining_unique
//   Approximated by simulation.

// Analytical: expected unique experts under uniform iid routing
static double expected_unique_uniform(int E, int K, int N) {
    // Each token independently picks K of E experts uniformly.
    // P(expert i not chosen by one token) = C(E-1,K)/C(E,K) = (E-K)/E
    // P(expert i not chosen by any of N tokens) = ((E-K)/E)^N
    // E[unique] = E * (1 - ((E-K)/E)^N)
    double p_miss_one = (double)(E - K) / (double)E;
    double p_miss_all = pow(p_miss_one, (double)N);
    return (double)E * (1.0 - p_miss_all);
}

// Monte Carlo simulation: uniform iid routing
static double simulate_unique_uniform(int E, int K, int N, int n_trials, std::mt19937 & rng) {
    double total = 0.0;
    for (int trial = 0; trial < n_trials; trial++) {
        std::vector<bool> seen(E, false);
        for (int tok = 0; tok < N; tok++) {
            // Pick K distinct experts uniformly at random
            std::vector<int> pool(E);
            for (int i = 0; i < E; i++) pool[i] = i;
            for (int i = 0; i < K; i++) {
                std::uniform_int_distribution<int> dist(i, E - 1);
                int j = dist(rng);
                std::swap(pool[i], pool[j]);
                seen[pool[i]] = true;
            }
        }
        int count = 0;
        for (int i = 0; i < E; i++) count += seen[i] ? 1 : 0;
        total += count;
    }
    return total / n_trials;
}

// Monte Carlo simulation: biased routing with overlap probability p
// Each token has probability p of reusing the same "hot set" as the previous
// token, and (1-p) of picking a fresh uniform set.
static double simulate_unique_biased(int E, int K, int N, double overlap_p,
                                     int n_trials, std::mt19937 & rng) {
    double total = 0.0;
    for (int trial = 0; trial < n_trials; trial++) {
        std::vector<bool> seen(E, false);
        std::vector<int> hot_set(K);  // current hot set

        // First token: pick uniformly
        {
            std::vector<int> pool(E);
            for (int i = 0; i < E; i++) pool[i] = i;
            for (int i = 0; i < K; i++) {
                std::uniform_int_distribution<int> dist(i, E - 1);
                int j = dist(rng);
                std::swap(pool[i], pool[j]);
                hot_set[i] = pool[i];
                seen[pool[i]] = true;
            }
        }

        for (int tok = 1; tok < N; tok++) {
            std::uniform_real_distribution<double> coin(0.0, 1.0);
            if (coin(rng) < overlap_p) {
                // Reuse hot set — mark them seen (already are)
                for (int i = 0; i < K; i++) seen[hot_set[i]] = true;
            } else {
                // Fresh uniform pick
                std::vector<int> pool(E);
                for (int i = 0; i < E; i++) pool[i] = i;
                for (int i = 0; i < K; i++) {
                    std::uniform_int_distribution<int> dist(i, E - 1);
                    int j = dist(rng);
                    std::swap(pool[i], pool[j]);
                    hot_set[i] = pool[i];
                    seen[pool[i]] = true;
                }
            }
        }
        int count = 0;
        for (int i = 0; i < E; i++) count += seen[i] ? 1 : 0;
        total += count;
    }
    return total / n_trials;
}

static void test_expert_overlap() {
    printf("===========================================================================\n");
    printf("  Section 1: Expert Overlap Simulation\n");
    printf("===========================================================================\n\n");

    std::mt19937 rng(42);
    const int n_trials = 10000;

    struct overlap_case {
        int         N;
        int         K;
        int         E;
        double      overlap_p;    // 0 = uniform, >0 = biased
        double      expected;     // analytical or approximate expected unique
        double      tolerance;    // absolute tolerance for PASS
        const char *label;
    };

    std::vector<overlap_case> cases = {
        //  N   K    E    overlap   expected  tol     label
        {   1,  8, 256,  0.0,        8.0,    0.5,   "N=1, K=8, E=256 (trivial)"              },
        {  10,  8, 256,  0.0,       68.0,    5.0,   "N=10, K=8, E=256 (no bias)"             },
        {  10,  8, 256,  0.7,       20.0,   10.0,   "N=10, K=8, E=256, 70% overlap"          },
        // Note: E*(1-((E-K)/E)^N) = 512*(1-(502/512)^10) = 91.7, not 180
        // (180 would require ~N=20 tokens with K=10 from E=512)
        {  10, 10, 512,  0.0,       92.0,    5.0,   "N=10, K=10, E=512 (Coder-Next, no bias)"},
        {  10, 10, 512,  0.7,       30.0,   15.0,   "N=10, K=10, E=512, 70% overlap"         },
    };

    printf("%-46s  %8s  %8s  %8s  %8s  %s\n",
           "Case", "Formula", "Sim", "Expect", "Delta", "Result");
    printf("%-46s  %8s  %8s  %8s  %8s  %s\n",
           "----------------------------------------------",
           "--------", "--------", "--------", "--------", "------");

    int pass_count = 0;
    for (auto & c : cases) {
        double formula = expected_unique_uniform(c.E, c.K, c.N);
        double sim;

        if (c.overlap_p > 0.0) {
            sim = simulate_unique_biased(c.E, c.K, c.N, c.overlap_p, n_trials, rng);
        } else {
            sim = simulate_unique_uniform(c.E, c.K, c.N, n_trials, rng);
        }

        double delta = fabs(sim - c.expected);
        bool pass = delta < c.tolerance;
        if (pass) pass_count++;

        printf("%-46s  %8.1f  %8.1f  %8.1f  %8.1f  %s\n",
               c.label, formula, sim, c.expected, delta,
               pass ? "PASS" : "FAIL");
    }

    // Extra: verify formula vs simulation for uniform case
    printf("\n  Formula vs Simulation validation (uniform, 10000 trials):\n");
    printf("  %-20s  %8s  %8s  %8s  %s\n", "Config", "Formula", "Sim", "Diff", "Match");
    struct validate_case { int N, K, E; };
    std::vector<validate_case> vcases = {
        {1, 8, 256}, {5, 8, 256}, {10, 8, 256}, {20, 8, 256},
        {1, 10, 512}, {5, 10, 512}, {10, 10, 512}, {20, 10, 512},
    };
    for (auto & v : vcases) {
        double f = expected_unique_uniform(v.E, v.K, v.N);
        double s = simulate_unique_uniform(v.E, v.K, v.N, n_trials, rng);
        double diff = fabs(f - s);
        char label[64];
        snprintf(label, sizeof(label), "N=%d K=%d E=%d", v.N, v.K, v.E);
        printf("  %-20s  %8.1f  %8.1f  %8.2f  %s\n",
               label, f, s, diff, diff < 3.0 ? "OK" : "DRIFT");
    }

    printf("\n  Overlap tests: %d/%d passed\n\n", pass_count, (int)cases.size());
}

// ============================================================================
// Section 2: Bandwidth Model
// ============================================================================
//
// Per-step cost:
//   weight_read_GB = (unique_experts * expert_size_MB * n_layers + shared_weight_MB) / 1024
//   step_time_ms   = weight_read_GB / bandwidth_GBps * 1000
//   agg_tok_per_s  = 1000 / step_time_ms
//   slot_tok_per_s  = agg_tok_per_s / n_slots

struct model_spec {
    const char * name;
    int    n_experts;
    int    top_k;
    int    n_layers;
    double expert_size_mb;
    double shared_weight_mb;
    double bandwidth_gbps;
};

struct bandwidth_result {
    double unique_experts;
    double weight_read_gb;
    double step_time_ms;
    double agg_tok_s;
};

static bandwidth_result compute_bandwidth(const model_spec & m, double unique_experts) {
    bandwidth_result r;
    r.unique_experts = unique_experts;
    r.weight_read_gb = (unique_experts * m.expert_size_mb * m.n_layers
                        + m.shared_weight_mb) / 1024.0;
    r.step_time_ms   = r.weight_read_gb / m.bandwidth_gbps * 1000.0;
    r.agg_tok_s      = 1000.0 / r.step_time_ms;
    return r;
}

static void test_bandwidth_model() {
    printf("===========================================================================\n");
    printf("  Section 2: Bandwidth Model\n");
    printf("===========================================================================\n\n");

    std::vector<model_spec> models = {
        // Qwen3.5-35B MoE: 256 experts, top-8, 40 MoE layers, ~5MB/expert, 0.8GB shared
        {"Qwen3.5-35B", 256, 8, 40, 5.0, 800.0, 212.0},
        // Coder-Next: 512 experts, top-10, 48 MoE layers, ~3MB/expert, 1.2GB shared
        {"Coder-Next",  512, 10, 48, 3.0, 1200.0, 212.0},
    };

    for (auto & m : models) {
        printf("  Model: %s\n", m.name);
        printf("    %d experts, top-%d, %d layers, %.1f MB/expert, %.0f MB shared, %.0f GB/s\n",
               m.n_experts, m.top_k, m.n_layers, m.expert_size_mb,
               m.shared_weight_mb, m.bandwidth_gbps);
        printf("\n");

        printf("    %-12s  %12s  %12s  %12s\n",
               "Uniq Experts", "Weight (GB)", "Step (ms)", "Agg tok/s");
        printf("    %-12s  %12s  %12s  %12s\n",
               "------------", "------------", "------------", "------------");

        // Test with different unique expert counts
        std::vector<double> unique_counts;
        // top_k (single token), 2*top_k, formula for 5 tokens, 10 tokens, all experts
        for (int n_slots : {1, 2, 5, 10}) {
            double u = expected_unique_uniform(m.n_experts, m.top_k, n_slots);
            unique_counts.push_back(u);
        }
        unique_counts.push_back((double)m.n_experts);  // worst case: all experts

        const char * labels[] = {"1 slot", "2 slots", "5 slots", "10 slots", "all experts"};
        for (size_t i = 0; i < unique_counts.size(); i++) {
            auto r = compute_bandwidth(m, unique_counts[i]);
            printf("    %-12s  %12.2f  %12.2f  %12.1f\n",
                   labels[i], r.weight_read_gb, r.step_time_ms, r.agg_tok_s);
        }

        // Show per-slot tok/s for N=1..10
        printf("\n    Per-slot throughput (no scheduling overhead):\n");
        printf("    %-8s  %10s  %10s  %14s  %s\n",
               "N slots", "Uniq exp", "Step (ms)", "Per-slot t/s", ">=30 t/s?");
        printf("    %-8s  %10s  %10s  %14s  %s\n",
               "--------", "----------", "----------", "--------------", "---------");
        for (int n = 1; n <= 10; n++) {
            double u = expected_unique_uniform(m.n_experts, m.top_k, n);
            auto r = compute_bandwidth(m, u);
            double per_slot = r.agg_tok_s / n;
            printf("    %-8d  %10.1f  %10.2f  %14.1f  %s\n",
                   n, u, r.step_time_ms, per_slot,
                   per_slot >= 30.0 ? "YES" : "no");
        }
        printf("\n");
    }
}

// ============================================================================
// Section 3: Cache-Aware Routing Simulation
// ============================================================================
//
// Simulates the routing process:
//   1. Initialize router logits randomly for N tokens x E experts
//   2. Apply softmax
//   3. Apply cache bonus to a subset of experts
//   4. Select top-K
//   5. Measure actual overlap vs predicted overlap
//   6. Sweep cache_bonus from 0.0 to 2.0 and measure overlap %

static void softmax_row(float * logits, int n) {
    float mx = *std::max_element(logits, logits + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = expf(logits[i] - mx);
        sum += logits[i];
    }
    for (int i = 0; i < n; i++) {
        logits[i] /= sum;
    }
}

static void test_cache_aware_routing() {
    printf("===========================================================================\n");
    printf("  Section 3: Cache-Aware Routing Simulation\n");
    printf("===========================================================================\n\n");

    std::mt19937 rng(123);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    struct routing_config {
        const char * name;
        int E;
        int K;
        int N_tokens;
        int cache_size;   // how many experts are "cached" (get the bonus)
    };

    std::vector<routing_config> configs = {
        {"Qwen3.5-35B", 256,  8, 10, 32},
        {"Coder-Next",  512, 10, 10, 40},
    };

    for (auto & cfg : configs) {
        printf("  Config: %s (E=%d, K=%d, N=%d, cache=%d)\n\n",
               cfg.name, cfg.E, cfg.K, cfg.N_tokens, cfg.cache_size);

        // Generate random logits for N tokens x E experts
        std::vector<std::vector<float>> token_logits(cfg.N_tokens,
                                                     std::vector<float>(cfg.E));
        for (int t = 0; t < cfg.N_tokens; t++) {
            for (int e = 0; e < cfg.E; e++) {
                token_logits[t][e] = normal(rng);
            }
        }

        // Define the "cached" set: first cache_size experts (arbitrary)
        std::vector<bool> is_cached(cfg.E, false);
        for (int e = 0; e < cfg.cache_size; e++) {
            is_cached[e] = true;
        }

        // Sweep cache_bonus
        printf("    %-12s  %10s  %10s  %10s  %10s  %10s\n",
               "Cache Bonus", "Overlap %", "Uniq Exp", "Cached Hit", "Pred Uniq", "Reduction");
        printf("    %-12s  %10s  %10s  %10s  %10s  %10s\n",
               "------------", "----------", "----------", "----------", "----------", "----------");

        double no_bias_unique = 0.0;

        for (double bonus = 0.0; bonus <= 2.05; bonus += 0.1) {
            // For each token, apply softmax + bonus, select top-K
            std::vector<std::vector<int>> selections(cfg.N_tokens);

            for (int t = 0; t < cfg.N_tokens; t++) {
                std::vector<float> probs(cfg.E);
                std::copy(token_logits[t].begin(), token_logits[t].end(), probs.begin());
                softmax_row(probs.data(), cfg.E);

                // Apply cache bonus
                for (int e = 0; e < cfg.E; e++) {
                    if (is_cached[e]) {
                        probs[e] += (float)bonus;
                    }
                }

                // Top-K selection
                std::vector<int> indices(cfg.E);
                for (int i = 0; i < cfg.E; i++) indices[i] = i;
                std::partial_sort(indices.begin(), indices.begin() + cfg.K, indices.end(),
                    [&probs](int a, int b) { return probs[a] > probs[b]; });
                selections[t].assign(indices.begin(), indices.begin() + cfg.K);
            }

            // Measure overlap: how many experts selected by token t are also
            // selected by token t-1?
            double total_overlap = 0.0;
            for (int t = 1; t < cfg.N_tokens; t++) {
                int overlap = 0;
                for (int a : selections[t]) {
                    for (int b : selections[t - 1]) {
                        if (a == b) { overlap++; break; }
                    }
                }
                total_overlap += (double)overlap / cfg.K;
            }
            double avg_overlap = total_overlap / (cfg.N_tokens - 1) * 100.0;

            // Count unique experts across all tokens
            std::vector<bool> seen(cfg.E, false);
            for (int t = 0; t < cfg.N_tokens; t++) {
                for (int e : selections[t]) seen[e] = true;
            }
            int unique = 0;
            int cached_hit = 0;
            for (int e = 0; e < cfg.E; e++) {
                if (seen[e]) {
                    unique++;
                    if (is_cached[e]) cached_hit++;
                }
            }

            double predicted_unique = expected_unique_uniform(cfg.E, cfg.K, cfg.N_tokens);

            if (bonus < 0.05) no_bias_unique = unique;

            double reduction = (no_bias_unique > 0)
                ? (1.0 - (double)unique / no_bias_unique) * 100.0
                : 0.0;

            printf("    %12.1f  %10.1f  %10d  %10d  %10.1f  %9.1f%%\n",
                   bonus, avg_overlap, unique, cached_hit, predicted_unique, reduction);
        }
        printf("\n");
    }
}

// ============================================================================
// Section 4: Scheduling Overhead Model
// ============================================================================
//
// Given step_time_ms, n_slots, scheduling_overhead_ms:
//   effective_step_ms = step_time_ms + scheduling_overhead_ms
//   agg_tok_s = n_slots * 1000 / effective_step_ms
//   per_slot_tok_s = 1000 / effective_step_ms

static void test_scheduling_overhead() {
    printf("===========================================================================\n");
    printf("  Section 4: Scheduling Overhead Model\n");
    printf("===========================================================================\n\n");

    // Use Qwen3.5-35B as reference
    model_spec qwen = {"Qwen3.5-35B", 256, 8, 40, 5.0, 800.0, 212.0};

    struct overhead_scenario {
        const char * label;
        double scheduling_overhead_ms;
    };

    std::vector<overhead_scenario> scenarios = {
        {"ideal (0 ms)",   0.0},
        {"light (0.5 ms)", 0.5},
        {"medium (1 ms)",  1.0},
        {"heavy (2 ms)",   2.0},
        {"worst (5 ms)",   5.0},
    };

    for (auto & sc : scenarios) {
        printf("  Scheduling overhead: %s\n\n", sc.label);
        printf("    %-8s  %10s  %10s  %10s  %10s  %12s  %s\n",
               "N slots", "Uniq exp", "BW (ms)", "Sched (ms)", "Total (ms)",
               "Per-slot t/s", ">=30?");
        printf("    %-8s  %10s  %10s  %10s  %10s  %12s  %s\n",
               "--------", "----------", "----------", "----------", "----------",
               "------------", "-----");

        for (int n = 1; n <= 20; n++) {
            double u = expected_unique_uniform(qwen.n_experts, qwen.top_k, n);
            auto bw = compute_bandwidth(qwen, u);
            double total_ms = bw.step_time_ms + sc.scheduling_overhead_ms;
            double per_slot = 1000.0 / total_ms;

            printf("    %-8d  %10.1f  %10.2f  %10.2f  %10.2f  %12.1f  %s\n",
                   n, u, bw.step_time_ms, sc.scheduling_overhead_ms, total_ms,
                   per_slot, per_slot >= 30.0 ? "YES" : "no");
        }
        printf("\n");
    }

    // Find sweet spot for 30 tok/s target
    printf("  Sweet-spot analysis (target: 30 tok/s per slot):\n\n");
    printf("    %-16s  %12s  %12s\n",
           "Overhead (ms)", "Max slots", "Agg tok/s");
    printf("    %-16s  %12s  %12s\n",
           "----------------", "------------", "------------");

    for (auto & sc : scenarios) {
        int max_slots = 0;
        double best_agg = 0.0;
        for (int n = 1; n <= 50; n++) {
            double u = expected_unique_uniform(qwen.n_experts, qwen.top_k, n);
            auto bw = compute_bandwidth(qwen, u);
            double total_ms = bw.step_time_ms + sc.scheduling_overhead_ms;
            double per_slot = 1000.0 / total_ms;
            if (per_slot >= 30.0) {
                max_slots = n;
                best_agg = n * per_slot;
            }
        }
        printf("    %-16s  %12d  %12.1f\n",
               sc.label, max_slots, best_agg);
    }
    printf("\n");
}

// ============================================================================
// Section 5: Full Projection Table
// ============================================================================
//
// Combines all models to produce a table showing:
//   For N=1,2,5,7,10 slots with and without cache-aware routing:
//   expected agg tok/s and per-slot tok/s.
//   Marks which configs achieve 30 tok/s per-slot.

static void test_full_projection() {
    printf("===========================================================================\n");
    printf("  Section 5: Full Projection Table\n");
    printf("===========================================================================\n\n");

    std::vector<model_spec> models = {
        {"Qwen3.5-35B", 256, 8, 40, 5.0, 800.0, 212.0},
        {"Coder-Next",  512, 10, 48, 3.0, 1200.0, 212.0},
    };

    // Cache-aware routing parameters
    const double cache_overlap_p = 0.70;  // 70% overlap with cache-aware routing
    const double scheduling_ms   = 0.5;   // realistic scheduling overhead

    std::mt19937 rng(999);
    const int sim_trials = 5000;

    int slot_counts[] = {1, 2, 5, 7, 10};

    for (auto & m : models) {
        printf("  Model: %s\n", m.name);
        printf("    Config: %d experts, top-%d, %d layers, %.1f MB/expert\n",
               m.n_experts, m.top_k, m.n_layers, m.expert_size_mb);
        printf("    BW: %.0f GB/s, shared: %.0f MB, sched overhead: %.1f ms\n\n",
               m.bandwidth_gbps, m.shared_weight_mb, scheduling_ms);

        printf("    %-6s  |  %-10s  %10s  %10s  %8s  |  %-10s  %10s  %10s  %8s\n",
               "",
               "NO CACHE", "Step(ms)", "Agg t/s", "Slot t/s",
               "CACHED", "Step(ms)", "Agg t/s", "Slot t/s");
        printf("    %-6s  |  %-10s  %10s  %10s  %8s  |  %-10s  %10s  %10s  %8s\n",
               "Slots",
               "Uniq exp", "", "", "",
               "Uniq exp", "", "", "");
        printf("    %s\n",
               "------  |  ----------  ----------  ----------  --------"
               "  |  ----------  ----------  ----------  --------");

        for (int n : slot_counts) {
            // No cache: uniform routing
            double u_no = expected_unique_uniform(m.n_experts, m.top_k, n);
            auto bw_no = compute_bandwidth(m, u_no);
            double total_no = bw_no.step_time_ms + scheduling_ms;
            double agg_no = (double)n * 1000.0 / total_no;
            double slot_no = 1000.0 / total_no;

            // With cache: biased routing reduces unique experts
            double u_cache = simulate_unique_biased(
                m.n_experts, m.top_k, n, cache_overlap_p, sim_trials, rng);
            auto bw_cache = compute_bandwidth(m, u_cache);
            double total_cache = bw_cache.step_time_ms + scheduling_ms;
            double agg_cache = (double)n * 1000.0 / total_cache;
            double slot_cache = 1000.0 / total_cache;

            char mark_no[4]    = "  ";
            char mark_cache[4] = "  ";
            if (slot_no >= 30.0)    snprintf(mark_no, sizeof(mark_no), "**");
            if (slot_cache >= 30.0) snprintf(mark_cache, sizeof(mark_cache), "**");

            printf("    %-6d  |  %10.1f  %10.2f  %10.1f  %6.1f%s  |"
                   "  %10.1f  %10.2f  %10.1f  %6.1f%s\n",
                   n,
                   u_no, total_no, agg_no, slot_no, mark_no,
                   u_cache, total_cache, agg_cache, slot_cache, mark_cache);
        }

        printf("\n    ** = achieves >= 30 tok/s per slot\n\n");

        // Speedup summary
        printf("    Cache-aware routing speedup summary:\n");
        printf("    %-6s  %14s  %14s  %10s\n",
               "Slots", "No-cache t/s", "Cached t/s", "Speedup");
        printf("    %-6s  %14s  %14s  %10s\n",
               "------", "--------------", "--------------", "----------");
        for (int n : slot_counts) {
            double u_no = expected_unique_uniform(m.n_experts, m.top_k, n);
            auto bw_no = compute_bandwidth(m, u_no);
            double slot_no = 1000.0 / (bw_no.step_time_ms + scheduling_ms);

            double u_cache = simulate_unique_biased(
                m.n_experts, m.top_k, n, cache_overlap_p, sim_trials, rng);
            auto bw_cache = compute_bandwidth(m, u_cache);
            double slot_cache = 1000.0 / (bw_cache.step_time_ms + scheduling_ms);

            printf("    %-6d  %14.1f  %14.1f  %9.2fx\n",
                   n, slot_no, slot_cache,
                   slot_cache / slot_no);
        }
        printf("\n");
    }

    // Final comparison: single-token decode (N=1) sanity check
    printf("  ============================================================\n");
    printf("  Single-token decode sanity check (N=1, no scheduling):\n");
    printf("  ============================================================\n\n");
    printf("    %-14s  %10s  %10s  %10s  %10s\n",
           "Model", "Experts", "Read (GB)", "Time (ms)", "tok/s");
    printf("    %-14s  %10s  %10s  %10s  %10s\n",
           "--------------", "----------", "----------", "----------", "----------");
    for (auto & m : models) {
        double u = (double)m.top_k;  // N=1 always uses exactly top_k experts
        auto bw = compute_bandwidth(m, u);
        printf("    %-14s  %10.0f  %10.3f  %10.2f  %10.1f\n",
               m.name, u, bw.weight_read_gb, bw.step_time_ms, bw.agg_tok_s);
    }
    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("\n");
    printf("###########################################################################\n");
    printf("#                                                                         #\n");
    printf("#  Expert Caching System: Math Validation & Bandwidth Benchmark           #\n");
    printf("#                                                                         #\n");
    printf("#  Models: Qwen3.5-35B (256e, top-8) and Coder-Next (512e, top-10)        #\n");
    printf("#  Hardware: 212 GB/s memory bandwidth (MI300X single-die assumption)      #\n");
    printf("#                                                                         #\n");
    printf("###########################################################################\n\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    test_expert_overlap();
    test_bandwidth_model();
    test_cache_aware_routing();
    test_scheduling_overhead();
    test_full_projection();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("===========================================================================\n");
    printf("  Benchmark complete. Total wall time: %.1f ms\n", elapsed_ms);
    printf("===========================================================================\n");

    return 0;
}

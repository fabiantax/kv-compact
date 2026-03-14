// Simple profiling instrumentation for kv-compact
// Add this to kv-compact.cpp for detailed timing analysis

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

class SimpleProfiler {
private:
    std::unordered_map<std::string, std::chrono::microseconds> timings;
    std::unordered_map<std::string, size_t> call_counts;
    std::string current_scope;
    std::chrono::steady_clock::time_point scope_start;

public:
    void enter_scope(const std::string& name) {
        current_scope = name;
        scope_start = std::chrono::steady_clock::now();
        call_counts[name]++;
    }

    void exit_scope() {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - scope_start);
        timings[current_scope] += duration;
    }

    void print_summary() {
        std::cout << "\n=== Profiling Summary ===\n";
        std::cout << "Function                  Calls      Total (ms)   Avg (us)\n";
        std::cout << "----------------------------------------------------------------\n";

        // Sort by total time
        std::vector<std::pair<std::string, std::chrono::microseconds>> sorted;
        for (auto& [name, time] : timings) {
            sorted.push_back({name, time});
        }
        std::sort(sorted.begin(), sorted.end(),
            [](auto& a, auto& b) { return a.second > b.second; });

        for (auto& [name, time] : sorted) {
            double total_ms = time.count() / 1000.0;
            double avg_us = time.count() / (double)call_counts[name];
            printf("%-25s %-10zu %10.2f %8.2f\n",
                name.c_str(), call_counts[name], total_ms, avg_us);
        }
    }

    static SimpleProfiler& instance() {
        static SimpleProfiler prof;
        return prof;
    }
};

// RAII scoped profiler
class ScopedProfiler {
public:
    ScopedProfiler(const std::string& name) {
        SimpleProfiler::instance().enter_scope(name);
    }

    ~ScopedProfiler() {
        SimpleProfiler::instance().exit_scope();
    }
};

#define PROFILE_SCOPE(name) ScopedProfiler _prof(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

// Example usage in kv-compact.cpp:
/*
void compact_layer(...) {
    PROFILE_FUNCTION();

    {
        PROFILE_SCOPE("key_selection");
        // key selection code
    }

    {
        PROFILE_SCOPE("nnls_fitting");
        // NNLS code
    }

    {
        PROFILE_SCOPE("value_refitting");
        // value refitting code
    }
}

int main(...) {
    // ... existing code ...

    // At the end:
    SimpleProfiler::instance().print_summary();

    return 0;
}
*/

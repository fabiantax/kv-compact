Feature: L2-Based Importance Estimation
  As a developer optimizing KV cache compaction
  I want to estimate token importance using L2 norm correlation
  So that key selection runs in O(n log k) instead of O(n²)

  Background:
    Given a head dimension of 64
    And 64 reference queries
    And synthetic keys and values generated with seed 42

  Scenario: Estimate importance for small token count
    Given 1000 tokens in the cache
    And a target compression ratio of 0.2 (200 tokens)
    When I compute L2-based importance scores
    Then I should select exactly 200 tokens
    And the selection should complete in less than 5 ms
    And the selected tokens should be unique
    And the selected tokens should be sorted in ascending order

  Scenario: Estimate importance for medium token count
    Given 10000 tokens in the cache
    And a target compression ratio of 0.2 (2000 tokens)
    When I compute L2-based importance scores
    Then I should select exactly 2000 tokens
    And the selection should complete in less than 20 ms
    And the algorithm should scale near-linearly with token count

  Scenario: Estimate importance for large token count
    Given 100000 tokens in the cache
    And a target compression ratio of 0.2 (20000 tokens)
    When I compute L2-based importance scores
    Then I should select exactly 20000 tokens
    And the selection should complete in less than 150 ms
    And the scaling should be sub-quadratic (time < 10 × baseline)

  Scenario: Importance scores correlate with attention
    Given 1000 tokens in the cache
    And queries generated from the last 256 tokens
    When I compute L2-based importance scores
    Then tokens with higher L2 norm should have higher importance
    And importance scores should be non-negative
    And the maximum importance should be significantly higher than average

  Scenario: Handle edge cases
    Given 100 tokens in the cache
    And a target compression ratio of 0.9 (90 tokens)
    When I compute L2-based importance scores
    Then I should select exactly 90 tokens
    And the algorithm should handle small token counts correctly
    And no division by zero should occur

Feature: Early-Stopping NNLS Solver
  As a developer reducing NNLS iterations
  I want the NNLS solver to stop early when convergence is detected
  So that compaction is faster without quality loss

  Background:
    Given a tolerance of 1e-3
    And a minimum improvement threshold of 1e-6
    And a maximum of 100 iterations

  Scenario: Converge on well-conditioned problem
    Given a well-conditioned M matrix (100 queries × 200 selected)
    And a target vector b with unit sum
    When I solve NNLS with early stopping
    Then the solver should converge in less than 10 iterations
    And the final solution should be non-negative
    And the residual norm should be less than 0.1

  Scenario: Stop at maximum iterations for hard problem
    Given an ill-conditioned M matrix
    And a target vector b with conflicting constraints
    When I solve NNLS with early stopping
    Then the solver should reach maximum iterations
    And the solution should be non-negative
    And the solver should not hang indefinitely

  Scenario: Warm start from previous solution
    Given a previous NNLS solution for 100 tokens
    And a new problem with 120 tokens
    When I solve NNLS with warm start enabled
    Then convergence should be faster than cold start
    And the iteration count should be less than cold start
    And the final quality should be equivalent

  Scenario: Adaptive iteration scaling
    Given 1000 selected tokens (large problem)
    When I solve NNLS with adaptive iterations
    Then the max iterations should scale with token count
    And the iterations should be approximately 5 × k
    And the solver should still converge quickly

Feature: Hierarchical Compaction
  As a developer optimizing for large contexts
  I want a two-pass hierarchical clustering approach
  So that compaction scales as O(n log n)

  Background:
    Given 64 coarse clusters
    And 4 refinements per cluster
    And a head dimension of 64

  Scenario: Compact small context
    Given 1000 tokens in the cache
    And a target of 200 tokens
    When I perform hierarchical compaction
    Then I should select approximately 200 tokens
    And compaction should complete in less than 1 ms
    And selected tokens should be unique

  Scenario: Compact medium context
    Given 10000 tokens in the cache
    And a target of 2000 tokens
    When I perform hierarchical compaction
    Then I should select approximately 2000 tokens
    And compaction should complete in less than 5 ms
    And the selection should distribute across clusters

  Scenario: Compact large context
    Given 100000 tokens in the cache
    And a target of 20000 tokens
    When I perform hierarchical compaction
    Then I should select approximately 20000 tokens
    And compaction should complete in less than 1 ms
    And the scaling should be sublinear (O(n log n))

  Scenario: Cluster distribution is balanced
    Given 10000 tokens in the cache
    And 64 coarse clusters
    When I perform hierarchical compaction
    Then each cluster should contain approximately 156 tokens
    And no cluster should be empty
    And cluster sizes should have low variance

Feature: Layer-Wise Budget Allocation
  As a researcher optimizing per-layer sensitivity
  I want different token budgets per layer
  So that quality is maintained while reducing total tokens

  Background:
    Given 36 layers in the model
    And a total budget of 4096 tokens
    And default sensitivities (first/last: 0.9, middle: 0.5)

  Scenario: Allocate budgets proportionally
    When I compute layer-wise budgets
    Then first and last quarters should receive higher budgets
    And middle layers should receive lower budgets
    And all budgets should be at least 16 tokens
    And the sum should equal the total budget

  Scenario: Handle minimum threshold
    Given a total budget of 100 tokens
    And 36 layers in the model
    When I compute layer-wise budgets
    Then each layer should receive at least 16 tokens
    And the allocation should be capped if sum exceeds total
    And the allocation should be scaled proportionally

  Scenario: Use custom sensitivities
    Given custom per-layer sensitivities
    And a total budget of 4096 tokens
    When I compute layer-wise budgets
    Then budgets should match the custom sensitivities
    And higher sensitivity should get more tokens
    And the allocation should be optimal for quality

Feature: Sublinear Streaming Compaction
  As a developer processing 200K-1M token contexts
  I want fixed-size windowing for O(1) amortized compaction
  So that total complexity is O(n)

  Background:
    Given a window size of 1024 tokens
    And a target budget of 4096 tokens

  Scenario: Compact single window
    Given 1024 new tokens arrive
    When I add tokens to the stream
    Then the window should be compacted
    And compaction should complete in O(window log window)
    And the token count should reset after compaction

  Scenario: Accumulate tokens across windows
    Given 500 new tokens arrive
    When I add tokens to the stream
    Then the window should NOT be compacted yet
    And the token count should be 500
    And no compaction should occur

  Scenario: Handle multiple windows
    Given 3000 new tokens arrive
    When I add tokens to the stream
    Then multiple compactions should occur
    And each compaction should be independent
    And total time should scale linearly with total tokens

  Scenario: Scale to 1M tokens
    Given 1000000 tokens arrive over time
    When I process all tokens with windowing
    Then compaction should complete in reasonable time (< 10s)
    And the per-window cost should be constant
    And the scaling should be O(n) not O(n²)

Feature: Integration with Main Compaction Pipeline
  As a user wanting faster compaction
  I want a single flag to enable all optimizations
  So that I don't need to configure each optimization individually

  Scenario: Enable optimizations with default method
    Given a prompt with 10000 tokens
    And the --optimized flag is set
    When I run compaction
    Then L2-based importance should be used
    And early stopping should be enabled
    And compaction should be faster than baseline
    And quality should be preserved (cos_sim > 0.95)

  Scenario: Select specific optimization method
    Given a prompt with 10000 tokens
    And the --method l2 flag is set
    When I run compaction
    Then L2-based importance should be used
    And the method should be logged
    And performance should match L2 characteristics

  Scenario: Enable early stopping independently
    Given a prompt with 10000 tokens
    And the --early-stop flag is set
    When I run compaction
    Then NNLS should use early stopping
    And iterations should be less than 10
    And convergence should be logged

  Scenario: Enable layer-wise budgets
    Given a prompt with 10000 tokens
    And the --layer-budget flag is set
    When I run compaction
    Then budgets should be allocated per layer
    And quality should improve vs uniform allocation
    And allocation should be logged

Feature: Quality Preservation
  As a researcher validating optimization effectiveness
  I want quality metrics to be computed and displayed
  So that I can verify optimizations don't degrade quality

  Scenario: Measure cosine similarity for L2
    Given a compacted cache using L2 importance
    And a reference query
    When I compute the output
    Then cosine similarity should be > 0.96
    And relative error should be < 0.1
    And metrics should be logged

  Scenario: Measure cosine similarity for hierarchical
    Given a compacted cache using hierarchical selection
    And a reference query
    When I compute the output
    Then cosine similarity should be > 0.95
    And relative error should be < 0.12
    And metrics should be logged

  Scenario: Compare quality across methods
    Given the same prompt and cache
    When I compact with baseline, L2, and hierarchical
    Then all methods should have cos_sim > 0.95
    And L2 should be within 5% of baseline
    And hierarchical should be within 8% of baseline

Feature: Performance Validation
  As a developer validating optimization claims
  I want benchmarks to measure actual speedup
  So that I can confirm O(n log n) scaling

  Scenario: Benchmark small token count
    Given 100 tokens in the cache
    When I run the optimization benchmark
    Then baseline, L2, and hierarchical should all be measured
    And times should be reported in milliseconds
    And speedup should be calculated

  Scenario: Benchmark medium token count
    Given 1000 tokens in the cache
    When I run the optimization benchmark
    Then L2 should be faster than baseline
    And hierarchical should be fastest
    And speedup should increase with token count

  Scenario: Benchmark large token count
    Given 10000 tokens in the cache
    When I run the optimization benchmark
    Then L2 should show > 10x speedup
    And hierarchical should show > 50x speedup
    And results should be saved to CSV

  Scenario: Validate sublinear scaling
    Given benchmark results at multiple token counts
    When I analyze the scaling
    Then baseline should show quadratic growth
    And L2 should show near-linear growth
    And hierarchical should show sublinear growth
    And complexity should be confirmed O(n log n)

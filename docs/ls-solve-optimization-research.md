# Least-Squares Solve Optimization for KV Cache Compaction

**Date:** 2026-04-05
**Scope:** Practical approaches to accelerate the dense LS value refit step (73-86% of compaction time)
**Target hardware:** AMD 8060S RDNA 3.5 APU (40 CUs, 128 GB unified memory, wavefront 32)

---

## Problem Statement

The value refit step solves:

```
min ||X * C_v - Y||^2    =>    C_v = (X^T X + ridge*I)^{-1} X^T Y
```

Where:
- X is (n_q x t): softmax attention weights over selected keys, n_q = 4-64 queries, t = kept tokens
- Y is (n_q x d_v): original attention-weighted value output
- C_v is (t x d_v): the refitted compacted values we solve for
- d_v = 64-256 (value dimension per head)

Current implementation (in `least_squares_solve()` at `kv-compact-math.h:463-557`):
1. Compute A^T*A: O(n_q * t^2) -- three nested loops, naive C
2. Compute A^T*b: O(n_q * t * d_v) -- three nested loops
3. Gaussian elimination with partial pivoting: O(t^3 + t^2 * d_v) on augmented matrix [A^T*A | A^T*b]

**Typical dimensions per solve (after chunked compaction, t_chunk ~ 256):**

| Scenario | n_q | t | d_v | A^T*A size | Solve flops |
|----------|-----|---|-----|------------|-------------|
| Qwen 3.5, 5x | 32 | 256 | 256 | 256x256 = 256 KB | ~55M |
| Qwen 3.5, 10x | 32 | 128 | 256 | 128x128 = 64 KB | ~7M |
| Llama 3.1, 5x | 32 | 256 | 128 | 256x256 = 256 KB | ~28M |
| Small model, 5x | 16 | 256 | 64 | 256x256 = 256 KB | ~14M |

**Frequency:** Per-head, per-chunk, per-layer. For Qwen 3.5 at 50k tokens: 2 KV heads * ~195 chunks * 10 attention layers = ~3,900 independent LS solves.

**Current timing share:** 73-86% of total compaction time (scoring_ms is 9-30%, nnls_ms which includes LS is 70-86%).

---

## Approach 1: Iterative Solvers (CG / LSQR)

### Description

Instead of forming and factoring the normal equations (O(t^3)), use an iterative method that only requires matrix-vector products with X^T X (or X and X^T directly).

**Conjugate Gradient on Normal Equations (CGNE):**
- Each iteration: one multiply by X^T and one by X -- O(n_q * t) per iteration
- Convergence depends on condition number kappa(X^T X)
- Need kappa iterations for kappa-relative residual reduction

**LSQR (equivalent to CG on X^T X, but numerically more stable):**
- Same iteration complexity, better numerical properties for ill-conditioned problems

### Analysis for our problem

The matrix X is a softmax matrix. Its singular value spectrum depends on:
- How concentrated the attention is (few dominant singular values)
- The ratio n_q/t (underdetermined when n_q < t, which happens for aggressive compaction)

**Key property: softmax matrices have rapidly decaying singular values.**

For a typical attention pattern where each query attends to 3-5 of the t selected keys:
- Effective rank r is 3-5, far less than min(n_q, t)
- Condition number can be very large (1e6+), but the "effective" condition number (ratio of sigma_1/sigma_r) is small
- CG converges to machine-precision in r iterations for a rank-r system

With the existing ridge regularization (1e-6), the bottom singular values are lifted, making kappa more moderate.

### Complexity comparison

| Method | Flops | Memory | Convergence |
|--------|-------|--------|-------------|
| Current (Gauss elim) | 2*n_q*t^2 + 2/3*t^3 + t^2*d_v | t^2 + t*d_v | Exact in 1 step |
| CGNE (k iters) | k * (4*n_q*t + 2*t*d_v) | t + n_q*t + d_v | Residual halved per ~kappa^{1/2} iters |
| LSQR (k iters) | k * (4*n_q*t + 2*t*d_v) | t + n_q*t + d_v | Same as CGNE |

For t=256, n_q=32, d_v=256, effective rank ~5:
- Direct: 2*32*256^2 + 2/3*256^3 + 256^2*256 = ~55M flops
- CG with 10 iters: 10 * (4*32*256 + 2*256*256) = ~1.6M flops -- **~35x fewer flops**

### Preconditioning

The ridge term (lambda * I) is already a form of Tikhonov preconditioning. More sophisticated options:
- **Jacobi (diagonal):** P = diag(X^T X). Costs O(n_q * t) to compute. Very effective for softmax X where diagonal dominance varies.
- **Incomplete Cholesky:** O(t^2) setup, but the matrix is already small. Overkill.
- **Low-rank preconditioner:** Since X has effective rank r ~ 5, compute top-r eigenvectors of X^T X, use as deflation preconditioner. Costs O(n_q * t * r) via power iteration. Near-exact in r iterations.

**Recommended:** Jacobi preconditioner. Near-zero overhead, typically halves CG iterations.

### Estimated speedup

- Flop reduction: 10-35x (depends on effective rank)
- Memory reduction: t^2 normal equations matrix no longer needed (saves 256 KB per solve at t=256)
- With 10 CG iterations: **10-20x speedup** on the LS solve
- On total compaction: **7-17x** (since LS is 73-86% of total)

### Quality impact

- CG with 10 iterations on a well-conditioned system: residual < 1e-8
- Ridge regularization helps convergence
- **Negligible quality impact** for k >= effective_rank iterations

### Implementation complexity

**Low.** Replace `least_squares_solve()` with a CGNE implementation:
```cpp
// Conjugate Gradient on Normal Equations
// Solves (X^T X + ridge*I) * x = X^T b
x = 0;
r = X^T * b;           // residual = RHS
p = r;                  // search direction
for (int k = 0; k < max_iter; k++) {
    Ap = X^T * (X * p) + ridge * p;  // matrix-vector product
    alpha = r^T r / (p^T Ap);
    x += alpha * p;
    r_new = r - alpha * Ap;
    if (||r_new|| < tol) break;
    beta = r_new^T r_new / (r^T r);
    p = r_new + beta * p;
    r = r_new;
}
```

The mat-vec `X * p` is O(n_q * t) and `X^T * v` is O(n_q * t). No need to form X^T*X explicitly.

### Literature

- CGNE/CGNR is standard for overdetermined LS (Hestenes & Stiefel, 1952)
- LSQR (Paige & Saunders, 1982) is the numerically stable variant
- Not explored in the KV compression literature specifically, but the softmax low-rank property is well-known (see Wang et al., "Linear attention" and related work)

### Verdict: **HIGHLY RECOMMENDED**

This is the single highest-impact, lowest-risk optimization. The softmax attention matrix X has very low effective rank (3-10 in practice), meaning CG converges in a handful of iterations. The implementation is straightforward and can fall back to direct solve for safety.

---

## Approach 2: Approximate LS / Early Termination

### Description

Instead of solving to machine precision, accept approximate solutions. Two variants:

**A) CG with aggressive early stopping (k = 3-5 iterations)**
- Exploit the low effective rank: 3 iterations covers rank 3, 5 covers rank 5
- Accept residual of 1e-2 to 1e-4 instead of 1e-8

**B) Direct solve in reduced precision**
- Use FP16 or BF16 for the normal equations formation and solve
- 2x throughput on most hardware, acceptable quality for a "refit" step

### Analysis

The value refit minimizes ||X * C_v - Y||^2. The question is: how precise does C_v need to be?

The quality of the compaction is ultimately measured by how well the *compacted* attention output matches the *original*. This is the same quantity the LS solve optimizes. So any approximation error in C_v directly becomes output error.

However, the LS solution is already an approximation -- it's fitting to n_q reference queries, not to all future queries. The "right" solution depends on the distribution of future queries, which is unknown. So an approximate LS solution is not necessarily worse than the exact one for future queries.

**Experiment to run:** Measure quality (cosine sim, PPL) with CG at k=3, 5, 10, 20 vs direct solve.

### Estimated speedup

- k=3 CG: ~60x fewer flops than direct, **~50x on LS**, **~40x on total**
- k=5 CG: ~35x fewer flops, **~30x on LS**, **~25x on total**
- FP16 direct: ~2x on LS, **~1.5x on total**

### Quality impact

- k=3: potentially visible degradation for difficult cases (high compression, dense attention)
- k=5: likely indistinguishable from exact for most cases
- k=10: effectively exact for all practical purposes
- FP16: depends on condition number. With ridge=1e-6 in FP16, may lose precision

### Implementation complexity

**Trivial.** Change `max_iter` in CG from t to a small constant.

### Verdict: **RECOMMENDED with k=5-10 as default, k=3 as "fast" mode**

The combination of CG + early termination is the practical sweet spot. Start with k=10 (near-exact), benchmark, then optionally reduce to k=5.

---

## Approach 3: Randomized / Dimensionality Reduction (Sketching)

### Description

Reduce the n_q dimension before solving. Instead of solving with full n_q x t system X, project to a smaller system.

**Random projection:** Multiply X and Y by a random (n_q x s) matrix S^T from the left:
```
X' = S^T * X   (s x t)
Y' = S^T * Y   (s x d_v)
```
Then solve the sketched system X' * C_v = Y'. Cost drops from O(n_q * t^2) to O(s * t^2).

**Subsampling:** Simply use fewer reference queries. If n_q=32 and we subsample to s=8, the solve is 4x cheaper.

### Analysis

The value of n_q is already small (4-64, typically 32). With cheap-Qref the typical n_q is T/2 capped at 64. After chunked compaction, n_q may be larger than the chunk T itself (since Q_ref covers all positions, not just the chunk).

The key question: does the n_q dimension need to be larger than t? For a well-determined system (n_q > t), yes. For our case:
- n_q = 32, t = 256: the system is **underdetermined** (more unknowns than equations)
- The ridge regularization makes the solution unique, but the system has only n_q effective constraints

Since the system is already underdetermined, sketching n_q down further loses information. The effective number of constraints is already only min(n_q, effective_rank_of_X).

**Better alternative:** sketch the *t* dimension. Since X has effective rank r << t, we can:
1. Compute top-r right singular vectors of X via randomized SVD: O(n_q * t * r)
2. Project C_v to the column space: C_v = V_r * z where V_r is t x r
3. Solve the reduced (n_q x r) system for z, then recover C_v

This is effectively what CG does naturally -- it finds the solution in the dominant Krylov subspace.

### Estimated speedup

- Subsampling n_q from 32 to 8: 4x on LS solve, **~3x on total**. Quality risk.
- Randomized SVD + reduced solve: similar to CG with r iterations. No advantage over CG.

### Quality impact

- Reducing n_q: risky, loses coverage of the query space
- Randomized SVD approach: same quality as CG (they are mathematically equivalent in the limit)

### Implementation complexity

Subsampling is trivial. Randomized SVD is moderate complexity with no advantage over CG.

### Verdict: **NOT RECOMMENDED**

CG (Approach 1) already captures the low-rank structure optimally. Sketching adds complexity without benefit. Subsampling n_q is a valid but independent optimization (reducing the number of reference queries, already explored via SnapKV window).

---

## Approach 4: Woodbury Identity / Incremental Updates

### Description

If we compact incrementally (adding or removing tokens from the compacted set), the normal equations matrix X^T X changes by a rank-1 or rank-k update. The Woodbury identity gives:

```
(A + U V^T)^{-1} = A^{-1} - A^{-1} U (I + V^T A^{-1} U)^{-1} V^T A^{-1}
```

This turns an O(t^3) re-solve into O(t^2 * k) where k is the rank of the update.

### Analysis

This applies when:
1. We already have the inverse (or factorization) of X^T X from a previous solve
2. We modify a small number of rows in X (add/remove a few tokens)
3. We need to re-solve

**Current pipeline:** Chunked compaction processes each chunk independently. There is no incremental relationship between chunks -- different tokens, different X matrices.

**Iterative refinement (refine_rounds > 0):** This swaps one key at a time, which is exactly the rank-1 update scenario. Currently, each refinement round re-solves from scratch (O(t^3)). With Woodbury, each swap costs O(t^2).

**Multi-round compaction:** If the same context is compacted multiple times (e.g., during online reasoning), the second round's X is a different matrix (different selected keys). No Woodbury benefit.

### Estimated speedup

- Iterative refinement with k=1 swap per round: O(t^2) instead of O(t^3) per round -- **t-fold speedup per round**
- For t=256: **256x faster** per refinement round
- But refinement is currently disabled by default (refine_rounds=0), so this only helps when explicitly enabled

### Quality impact

Mathematically exact -- Woodbury gives the same result as full re-solve.

### Implementation complexity

**Moderate.** Need to:
1. Cache the Cholesky factorization (or inverse) of X^T X between refinement rounds
2. Implement rank-1 update to the factorization
3. Apply the updated factorization to solve

Using Cholesky rank-1 update (dch1up/dch1dn from LINPACK): ~100 lines of code.

### Verdict: **CONDITIONALLY RECOMMENDED**

Only useful when iterative refinement is enabled. If refinement becomes important for quality at extreme compression ratios, this is a clean optimization. Not worth implementing until refinement is actually used in production.

---

## Approach 5: Low-Rank Structure Exploitation

### Description

The softmax matrix X has rapidly decaying singular values. This means X^T X has the same low-rank structure. We can exploit this:

**A) Truncated SVD solve:**
1. Compute rank-r SVD of X: X = U_r * Sigma_r * V_r^T
2. Solve the reduced system: z = Sigma_r^{-1} * U_r^T * Y
3. Recover: C_v = V_r * z

Cost: O(n_q * t * r + n_q * r * d_v) for the solve, plus O(n_q * t * r) for the randomized SVD.

**B) Nystrom approximation:**
X^T X ~ X^T[:, S] * (X[S, S])^{-1} * X[:, S]^T where S is a subset of rows.
Solve the approximate normal equations.

### Analysis

For a softmax matrix X with effective rank r:
- Truncated SVD solve: total cost O(n_q * t * r), vs O(n_q * t^2 + t^3) for direct
- With r=5, n_q=32, t=256: 40K vs 55M -- **~1400x fewer flops**

But: this is mathematically equivalent to CG with r iterations (CG finds the solution in the dominant r-dimensional Krylov subspace). CG does not require an explicit SVD computation.

**Practical advantage of truncated SVD over CG:** None, really. CG is simpler and achieves the same result without needing to compute SVD.

**Where low-rank structure matters:** It determines how many CG iterations are needed. The faster the singular value decay, the fewer CG iterations.

### Singular value decay in practice

For softmax attention matrices:
- Sparse attention (few keys dominate): very fast decay, effective rank 2-5
- Diffuse attention (many keys have similar weight): slower decay, effective rank 10-30
- Early layers tend to be more diffuse; later layers more concentrated

Measuring the actual spectrum on Qwen 3.5 would inform the CG iteration count.

### Estimated speedup

Same as CG (Approach 1) -- they are equivalent. The speedup is determined by the effective rank, not the method.

### Verdict: **NOT RECOMMENDED as separate approach**

The low-rank structure is the theoretical justification for why CG works well. CG implicitly exploits it. No need for explicit SVD.

---

## Approach 6: GPU Offload

### Description

The project already has ROCm/HIP support for:
1. Multi-head attention scoring (`kv_compact_hip_score_all_heads`) via rocBLAS strided-batched GEMM
2. Multi-head value refit target computation (`kv_compact_hip_refit_target_all_heads`) via batched GEMM

The LS solve itself is not GPU-accelerated. Options:

**A) Batched LS on GPU (rocSOLVER):**
Use rocSOLVER's batched LU or Cholesky factorization. Process all heads in a single batched call.

**B) Batched CG on GPU:**
Implement CG as a custom HIP kernel that processes all heads in parallel. Each workgroup handles one head's CG iteration.

**C) Keep solve on CPU, offload only matmul parts:**
The A^T*A and A^T*b formation are GEMM operations that can use rocBLAS. The factorization stays CPU-side.

### Analysis

**Problem dimensions per solve:** t=256, n_q=32, d_v=256. The A^T*A matrix is 256x256 (256 KB). This is very small for GPU -- kernel launch overhead may dominate.

**Batch size:** 2 KV heads * 10 attention layers * ~195 chunks = 3,900 independent solves. But chunked compaction processes chunks sequentially (or with OpenMP parallelism). The batch exposed to GPU at any moment is just n_head_kv = 2 solves.

**rocSOLVER batched solve:** With batch size 2, GPU utilization is extremely poor. The CUs would be mostly idle.

**Custom CG kernel:** Each CG iteration for one head does:
- SpMV-like: X * p where X is 32x256 (8K elements), p is 256
- This is far too small for GPU. Even on the APU with unified memory, the kernel launch overhead exceeds the compute.

**The real opportunity:** The 3,900 independent solves could be batched together if we restructure the pipeline to:
1. Collect all X, Y matrices for all heads, layers, and chunks
2. Launch a single massive batched solve
3. Collect results

But this requires significant pipeline restructuring (currently chunks are processed independently and can be parallel via OpenMP).

**Alternative: BLAS-optimized CPU solve.** The biggest win may be simply linking against a tuned BLAS/LAPACK:
- Apple Accelerate (on current M5 hardware): `dsysv` or `dposv` for Cholesky
- OpenBLAS on AMD: optimized BLAS for the matrix operations
- The current naive triple-loop implementation leaves enormous performance on the table

### Estimated speedup

| Approach | Speedup on LS | Speedup on total | Notes |
|----------|---------------|-------------------|-------|
| rocSOLVER batched (batch=2) | 0.5-2x | 0.4-1.7x | Kernel overhead dominates |
| Custom CG kernel (batch=2) | 0.5-1.5x | 0.4-1.3x | Same overhead problem |
| Restructured pipeline, batch=3900 | 5-20x | 4-17x | Major refactoring needed |
| Apple Accelerate (CPU BLAS) | 3-10x | 2.5-8.5x | Drop-in replacement |
| BLIS/OpenBLAS on AMD | 2-5x | 1.7-4.3x | Drop-in replacement |

### Quality impact

None. Same algorithm, different implementation.

### Implementation complexity

- **BLAS/LAPACK drop-in:** Low. Replace `least_squares_solve()` body with `cblas_sgemm` + LAPACK `sposv` calls. ~50 lines changed.
- **rocSOLVER batched:** Moderate. Need to restructure to batch all heads together.
- **Full pipeline restructure:** High. Collect all problems, dispatch to GPU, collect results.

### Verdict: **RECOMMENDED: CPU BLAS first, GPU batched later**

The immediate win is replacing the naive triple-loop implementation with optimized BLAS/LAPACK routines. On Apple hardware, Accelerate gives 3-10x on these operations. On AMD APU, BLIS or rocSOLVER with batching after pipeline restructure.

---

## Approach 7: Skipping LS Entirely with Better Key Selection

### Description

If key selection were perfect (the compacted keys perfectly represent the original attention distribution), the LS value refit would converge to C_v = V[selected] -- i.e., the original values at selected positions. The LS solve is only needed because:
1. Key selection is imperfect (the compacted attention distribution differs from the original)
2. Even with perfect key selection, the softmax normalization changes when you remove keys

### Analysis

**What happens without LS (C_v = V[selected]):**

This is the "V-copy" baseline. The existing benchmark (`bench-results-codegen.md`) shows:
- At 5x compression: quality is "Good" with LS refit. Without LS (V-copy): quality degrades noticeably.
- The paper's ablation (Section 6) ranks value refitting as "consistent improvement" -- it helps but is not the most critical component.

**Can key selection be improved enough to eliminate LS?**

The theoretical limit: even with perfect key selection, softmax(q @ K_sel) != softmax(q @ K_all). The normalization factor (sum of exponentials) changes. With perfect key selection, the *relative* attention weights are preserved, but the *absolute* values change, which affects the output when mixed with new tokens during generation.

The only way to perfectly avoid LS is to preserve the exact attention output, which requires solving for C_v. There is no key selection strategy, however perfect, that eliminates the need for value refitting.

**Practical middle ground:**

For moderate compression (5x, 10x), V-copy + beta gives acceptable quality. The LS solve becomes important at higher compression (20x+) where the attention distribution changes more dramatically.

The existing code already supports V-copy (skip the LS solve, use original V values for selected positions). This is used for all layers except layer 0 in the CLI tool.

### Estimated speedup

- Skipping LS: eliminates 73-86% of compaction time. **4-7x on total compaction.**
- This is already an option via V-copy. The question is quality tradeoff.

### Quality impact

- 5x compression: moderate degradation, may still be acceptable for some use cases
- 10x+ compression: significant degradation. LS refit is critical.
- The paper found value refitting to be a consistent improvement across all settings

### Verdict: **NOT RECOMMENDED for default, ALREADY AVAILABLE as fast mode**

V-copy is already implemented and works at moderate compression. But for quality-sensitive applications, LS is necessary. The goal should be making LS fast, not removing it.

---

## Approach 8: Block-Diagonal / Diagonal Approximation

### Description

Instead of solving the full normal equations (X^T X) * x = X^T b, approximate X^T X as block-diagonal or diagonal:

**Diagonal (independent solve per dimension):**
```
diag(X^T X) * x = X^T b   =>   x[i] = (X^T b)[i] / (X^T X)[i][i]
```

This is one Gauss-Seidel/Jacobi iteration. Cost: O(n_q * t) -- essentially free.

**Block-diagonal (group d_v dimensions into blocks of size B):**
Partition the t unknowns into blocks and solve each independently. Cost: O(t/B * B^3) = O(t * B^2).

### Analysis

**Diagonal approximation:**
This ignores the off-diagonal correlations in X^T X. For softmax X, the off-diagonal elements capture the interaction between different selected keys. When two selected keys are "competitors" (both have high attention from the same queries), the off-diagonal element is large and important.

The diagonal approximation effectively assumes independence between selected keys. This is reasonable only when keys are very diverse (low correlation).

**Quality test:** The diagonal approximation is equivalent to:
```
C_v[j] = sum_i X[i,j] * Y[i] / sum_i X[i,j]^2
```
This is a weighted average of Y over queries, weighted by attention to key j. It completely ignores that C_v[j] and C_v[k] must coordinate to jointly approximate Y.

For the attention matching problem, this coordination is critical. If query q1 attends strongly to keys j and k (with weights 0.6 and 0.4), the output is 0.6*V[j] + 0.4*V[k]. The diagonal approximation would independently set C_v[j] = Y[j]/0.6 and C_v[k] = Y[k]/0.4, which is incorrect.

**Block-diagonal with B = d_v:** This doesn't help because the LS problem is structured as t unknowns per output dimension, not d_v unknowns per key. The coupling is in the t dimension (between keys), not the d_v dimension.

### Estimated speedup

- Diagonal: ~1000x on LS solve, but quality loss is severe
- Block-diagonal: marginal improvement over diagonal, same fundamental issue

### Quality impact

**Diagonal approximation: unacceptable for most use cases.** It ignores the core of the attention matching problem (coordinating values across selected keys to jointly approximate the original output).

### Verdict: **NOT RECOMMENDED**

The coupling between selected keys is the fundamental reason LS is needed. Diagonal approximation ignores this coupling. Block-diagonal doesn't address the right coupling structure.

---

## Summary and Prioritized Recommendations

| # | Approach | LS Speedup | Total Speedup | Quality Impact | Complexity | Priority |
|---|----------|-----------|---------------|----------------|------------|----------|
| 1 | **CG iterative solver** | 10-35x | 7-17x | Negligible (k=10) | Low | **P0** |
| 2 | **Early termination (k=5)** | 30-50x | 25-40x | Small | Trivial | **P0** |
| 6a | **CPU BLAS/LAPACK** | 3-10x | 2.5-8.5x | None | Low | **P0** |
| 6b | **GPU batched CG** | 5-20x | 4-17x | None | High | P2 |
| 4 | **Woodbury updates** | t-fold per round | Only with refinement | None | Moderate | P3 |
| 7 | **Skip LS (V-copy)** | Infinite | 4-7x | Moderate to severe | Already done | Available |
| 3 | **Sketching** | -- | -- | -- | -- | Skip |
| 5 | **Low-rank SVD** | Same as CG | Same as CG | -- | -- | Skip |
| 8 | **Diagonal approx** | ~1000x | ~6x | Severe | Trivial | Skip |

### Recommended implementation order

1. **Replace naive matmul with BLAS calls** in `least_squares_solve()`. Use `cblas_sgemm` for A^T*A and A^T*b, and LAPACK `sposv` (Cholesky) or `sgesv` (LU) for the solve. On Apple hardware, link Accelerate. ~50 lines changed, 3-10x immediate speedup.

2. **Implement CGNE as alternative LS solver.** New function `least_squares_solve_cg()` that uses conjugate gradient on the normal equations. Default to k=10 iterations with Jacobi preconditioner. Benchmark against direct solve.

3. **Benchmark and validate.** Run the full quality benchmark suite (PPL, cosine sim, agreement) with CG at k=3, 5, 10, 20 and direct solve. Establish quality/speed Pareto frontier.

4. **If GPU pipeline restructure is justified later:** Batch all LS solves across heads + layers + chunks into a single GPU dispatch. Implement CG as a custom HIP kernel operating on the batch. This requires significant refactoring but could give another 2-5x on top of CG.

### Expected final result

With BLAS + CG(k=10):
- LS solve: **30-100x faster** than current naive implementation
- Total compaction at 50k tokens: **3-6x faster** (from ~3.3s to ~0.5-1.1s)
- Total compaction at 200k tokens: **3-6x faster** (from ~13s to ~2-4s)
- Quality: **indistinguishable** from current direct solve

With CG(k=5) as "fast" mode:
- Additional **~1.5x** on top of CG(k=10)
- Quality: likely within measurement noise for most scenarios

---

## Appendix: Why the Normal Equations Formation is Slow

The current implementation uses naive triple-nested loops:

```cpp
// A^T * A: O(n_q * t^2) -- naive triple loop
for (int i = 0; i < n; i++)      // n = t (kept tokens)
    for (int j = 0; j < n; j++)  // n = t
        for (int k = 0; k < m; k++)  // m = n_q
            AtA[i * n + j] += A[k * n + i] * A[k * n + j];
```

For t=256, n_q=32: this is 256 * 256 * 32 = 2.1M multiply-adds, but the inner loop has stride-n access pattern (poor cache utilization when n > L1 cache line / sizeof(float)). The L1 cache miss rate is ~50% for n=256.

A BLAS `sgemm` call uses cache-blocking (typically 32-64x64 micro-kernels) that achieves near-peak throughput. Even without CG, just switching to BLAS gives 3-10x on the A^T*A computation alone.

The A^T*A computation is O(n_q * t^2) = 2.1M flops for the standard case. The actual Gaussian elimination is O(t^3 / 3) = 5.6M flops. Combined: ~7.7M flops. With naive C code achieving ~1 GFLOPS and BLAS achieving ~10-50 GFLOPS on a single core, the difference is 10-50x.

CG avoids forming A^T*A entirely, so it sidesteps both the computation and the cache miss problem.

# Improvement Tracker

Full status of all 24 concepts from adjacent-concepts.md.

Reference each by number (e.g., "Do 3") to implement.

## Implemented

| # | Name | Section | Status | Commit |
|---|------|---------|--------|--------|
| 1 | Sensitivity-weighted key selection | Sec 18 (RPCholesky) | Done | fd1416c |
| 2 | Alternating minimization | Sec 10 | Done | 961983f |
| 3 | Submodular key selection (BumbleBee) | Sec 21 | Done | b825d9a |
| 4 | Token merging (ToMe / D2O hybrid) | Sec 20 | Done | 36f533c |
| 5 | Sinkhorn for mass matching | Sec 6 | Done | 82580e4 |
| 6 | K-means centroid keys | Sec 16 | Done | 547d428 |
| 7 | Carathéodory-informed budgets | Sec 22 | Done | 547d428 |

## Not implemented

| # | Name | Section | Difficulty | Value | Notes |
|---|------|---------|------------|-------|-------|
| 8 | CUR decomposition | Sec 1 | Medium | High | Alternative to NNLS: joint key+value selection via column/row factorization |
| 9 | Coresets | Sec 2 | Medium | Medium | Weighted subset with approximation guarantees; overlaps with submodular |
| 10 | Nyström approximation | Sec 3 | Medium | High | Low-rank attention matrix approx; WildCat (Sec 18) already uses this idea |
| 11 | Determinantal Point Processes (DPPs) | Sec 4 | Hard | Medium | Diversity-aware sampling; log-det is submodular (Sec 21 covers this) |
| 12 | CSSP & leverage scores | Sec 5 | Medium | High | Principled key selection via statistical leverage; could replace top-t |
| 13 | Kernel herding / MMD | Sec 7 | Medium | Low | Greedy MMD minimization; similar to submodular but different objective |
| 14 | Rate-distortion theory | Sec 8 | Hard | Low | Theoretical framework only; no direct algorithm |
| 15 | Frank-Wolfe method | Sec 9 | Medium | Medium | Sparse optimization for beta; alternative to NNLS with sparsity control |
| 16 | Attention head pruning | Sec 11 | Easy | Medium | Prune entire heads; orthogonal to within-head compaction |
| 17 | MoE merging | Sec 12 | Hard | Low | Cross-expert merging; not directly applicable to KV compaction |
| 18 | Matrix sketching (Frequent Directions) | Sec 13 | Medium | Medium | Online/streaming low-rank approx; useful for incremental compaction |
| 19 | Knowledge distillation | Sec 14 | Hard | Medium | Feature-level distillation; requires training, not inference-time |
| 20 | Compressed sensing | Sec 15 | Hard | Medium | Sparse recovery of attention; CS-VLM (2025) applies this directly |
| 21 | Expectation-Maximization | Sec 17 | Medium | Medium | Soft assignment alternative to hard selection; 2-3 EM iterations practical |
| 22 | Sparse GPs / inducing points | Sec 19 | Hard | Medium | Keys as inducing points; elegant but complex implementation |
| 23 | Information bottleneck | Sec 23 | Hard | Low | Principled compression-quality tradeoff; theoretical, hard to operationalize |
| 24 | Fast multipole / hierarchical attention | Sec 24 | Hard | Low | Hierarchical structure (MuSe); requires architectural changes |

## Priority candidates for next implementation

Based on RICE scoring (value vs. effort):

1. **CSSP & leverage scores (Sec 5)** — principled replacement for top-t scoring
2. **CUR decomposition (Sec 1)** — joint key+value factorization
3. **Frank-Wolfe (Sec 9)** — sparse beta fitting with convergence guarantees
4. **EM soft assignment (Sec 17)** — 2-3 iterations on top of existing pipeline
5. **Attention head pruning (Sec 11)** — easiest; prune heads before compaction

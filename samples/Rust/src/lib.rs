//! KV Cache Compaction Library
//!
//! This library provides efficient KV cache compaction for transformer models
//! using attention-based key selection and value refitting.

use std::collections::HashMap;
use std::time::Instant;

pub mod attention;
pub mod nnls;
pub mod compression;
pub mod streaming;

pub use attention::{AttentionScore, KeySelector};
pub use nnls::{NnlsSolver, BetaFitter};
pub use compression::{CompactOptions, CompactResult};
pub use streaming::{StreamingCompactor, StreamingConfig};

/// Configuration for KV cache compaction
#[derive(Debug, Clone)]
pub struct CompactConfig {
    /// Target compression ratio (0.0 to 1.0)
    pub ratio: f32,

    /// Minimum number of tokens to keep
    pub min_tokens: usize,

    /// Number of reference queries for quality estimation
    pub num_ref_queries: usize,

    /// Whether to use submodular key selection
    pub use_submodular: bool,

    /// Regularization strength for NNLS
    pub regularization: f32,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self {
            ratio: 0.2,
            min_tokens: 16,
            num_ref_queries: 64,
            use_submodular: false,
            regularization: 1e-4,
        }
    }
}

/// Main compaction engine
pub struct KvCompactor {
    config: CompactConfig,
    selector: KeySelector,
    fitter: BetaFitter,
}

impl KvCompactor {
    /// Create a new compactor with default configuration
    pub fn new() -> Self {
        Self::with_config(CompactConfig::default())
    }

    /// Create a new compactor with custom configuration
    pub fn with_config(config: CompactConfig) -> Self {
        Self {
            selector: KeySelector::new(),
            fitter: BetaFitter::new(config.regularization),
            config,
        }
    }

    /// Compact the KV cache for a single layer
    ///
    /// # Arguments
    /// * `keys` - Key tensor of shape [num_tokens, num_heads, head_dim]
    /// * `values` - Value tensor of shape [num_tokens, num_heads, head_dim]
    /// * `queries` - Reference query tensor for scoring
    ///
    /// # Returns
    /// Compaction result with selected keys and refitted values
    pub fn compact_layer(
        &self,
        keys: &[f32],
        values: &[f32],
        queries: &[f32],
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<CompactResult, CompactionError> {
        let start = Instant::now();

        // Calculate target token count
        let target_tokens = (num_tokens as f32 * self.config.ratio) as usize;
        let target_tokens = target_tokens.max(self.config.min_tokens);

        // Select top keys by attention score
        let selected_indices = self.selector.select_top_keys(
            keys,
            queries,
            num_tokens,
            num_heads,
            head_dim,
            target_tokens,
        )?;

        // Extract selected keys and values
        let selected_keys = extract_selected(keys, &selected_indices, num_heads, head_dim);
        let selected_values = extract_selected(values, &selected_indices, num_heads, head_dim);

        // Fit attention mass biases (beta) using NNLS
        let beta = self.fitter.fit_beta(
            keys,
            queries,
            &selected_indices,
            num_tokens,
            num_heads,
            head_dim,
        )?;

        // Refit values using least squares
        let refitted_values = self.refit_values(
            keys,
            values,
            &selected_indices,
            &beta,
            num_tokens,
            num_heads,
            head_dim,
        )?;

        // Calculate quality metrics
        let quality = self.calculate_quality(
            keys,
            values,
            &selected_keys,
            &refitted_values,
            queries,
            &selected_indices,
            num_tokens,
            num_heads,
            head_dim,
        )?;

        Ok(CompactResult {
            selected_indices,
            selected_keys,
            refitted_values,
            beta,
            quality,
            compaction_time: start.elapsed(),
        })
    }

    /// Calculate quality metrics (cosine similarity, relative error)
    fn calculate_quality(
        &self,
        original_keys: &[f32],
        original_values: &[f32],
        compacted_keys: &[f32],
        compacted_values: &[f32],
        queries: &[f32],
        selected_indices: &[usize],
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<QualityMetrics, CompactionError> {
        use std::f32::EPSILON;

        let mut cos_sim_sum = 0.0;
        let mut rel_err_sum = 0.0;

        for head in 0..num_heads {
            // Calculate attention for original and compacted
            let original_attn = compute_attention(
                original_keys,
                original_values,
                queries,
                num_tokens,
                head_dim,
                head,
            );

            let compacted_attn = compute_attention(
                compacted_keys,
                compacted_values,
                queries,
                selected_indices.len(),
                head_dim,
                head,
            );

            // Cosine similarity
            let dot_product = original_attn.iter()
                .zip(compacted_attn.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();

            let norm_a = original_attn.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b = compacted_attn.iter().map(|x| x * x).sum::<f32>().sqrt();

            let cos_sim = if norm_a > EPSILON && norm_b > EPSILON {
                dot_product / (norm_a * norm_b)
            } else {
                0.0
            };

            // Relative error
            let rel_err = original_attn.iter()
                .zip(compacted_attn.iter())
                .map(|(a, b)| {
                    if a.abs() > EPSILON {
                        (a - b).abs() / a.abs()
                    } else {
                        0.0
                    }
                })
                .sum::<f32>() / original_attn.len() as f32;

            cos_sim_sum += cos_sim;
            rel_err_sum += rel_err;
        }

        Ok(QualityMetrics {
            cosine_similarity: cos_sim_sum / num_heads as f32,
            relative_error: rel_err_sum / num_heads as f32,
        })
    }

    /// Refit values using least squares regression
    fn refit_values(
        &self,
        keys: &[f32],
        values: &[f32],
        selected_indices: &[usize],
        beta: &[f32],
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, CompactionError> {
        // Implementation of least squares value refitting
        // This is a placeholder - actual implementation would solve:
        // C_v* = argmin ||V - C_k * C_v*||^2
        let compacted_size = selected_indices.len() * num_heads * head_dim;
        Ok(vec![0.0; compacted_size])
    }
}

/// Extract selected tokens from tensor
fn extract_selected(
    tensor: &[f32],
    indices: &[usize],
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(indices.len() * num_heads * head_dim);

    for &idx in indices {
        let start = idx * num_heads * head_dim;
        let end = start + num_heads * head_dim;
        result.extend_from_slice(&tensor[start..end]);
    }

    result
}

/// Compute attention output for a single head
fn compute_attention(
    keys: &[f32],
    values: &[f32],
    queries: &[f32],
    num_tokens: usize,
    head_dim: usize,
    head_idx: usize,
) -> Vec<f32> {
    // Simplified attention computation
    // Actual implementation would compute softmax(Q @ K^T) @ V
    vec![0.0; num_tokens]
}

/// Quality metrics for compaction
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub cosine_similarity: f32,
    pub relative_error: f32,
}

/// Errors that can occur during compaction
#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    #[error("Invalid tensor dimensions")]
    InvalidDimensions,

    #[error("NNLS solver failed to converge")]
    NnlsConvergenceFailed,

    #[error("Insufficient tokens for compaction")]
    InsufficientTokens,

    #[error("Numerical error in computation")]
    NumericalError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CompactConfig::default();
        assert_eq!(config.ratio, 0.2);
        assert_eq!(config.min_tokens, 16);
    }

    #[test]
    fn test_compactor_creation() {
        let compactor = KvCompactor::new();
        assert_eq!(compactor.config.ratio, 0.2);
    }
}

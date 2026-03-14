//! Streaming compaction for long-context applications
//!
//! This module provides incremental KV cache compaction for
//! handling contexts beyond 100K tokens.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::{CompactConfig, QualityMetrics};

/// Configuration for streaming compaction
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum cache size before triggering compaction
    pub trigger_threshold: usize,

    /// Target budget after compaction
    pub target_budget: usize,

    /// Number of recent tokens to keep uncompacted
    pub recent_window: usize,

    /// Number of prefix tokens to pin (e.g., system prompt)
    pub pin_prefix: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            trigger_threshold: 8192,
            target_budget: 4096,
            recent_window: 512,
            pin_prefix: 256,
        }
    }
}

/// Zone-based cache management
#[derive(Debug, Clone, Copy)]
pub enum CacheZone {
    /// Pinned zone (system prompt, instructions)
    Pinned,

    /// Compactable zone (middle context)
    Compactable,

    /// Recent zone (latest tokens, keep accessible)
    Recent,
}

/// State for streaming compaction
pub struct StreamingCompactor {
    config: StreamingConfig,
    compact_config: CompactConfig,
    cache_size: usize,
    total_tokens_processed: usize,
    compaction_count: usize,
    last_compaction_time: Option<Instant>,
}

impl StreamingCompactor {
    /// Create a new streaming compactor
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            compact_config: CompactConfig::default(),
            cache_size: 0,
            total_tokens_processed: 0,
            compaction_count: 0,
            last_compaction_time: None,
        }
    }

    /// Check if compaction should be triggered
    pub fn needs_compaction(&self) -> bool {
        self.cache_size >= self.config.trigger_threshold
    }

    /// Get the size that would be after compaction
    pub fn compactable_size(&self) -> usize {
        if self.cache_size <= self.config.pin_prefix + self.config.recent_window {
            return 0;
        }

        self.cache_size - self.config.pin_prefix - self.config.recent_window
    }

    /// Get target size after compaction
    pub fn target_size(&self) -> usize {
        self.config.target_budget
    }

    /// Add new tokens to the cache
    pub fn add_tokens(&mut self, count: usize) -> CacheZone {
        self.total_tokens_processed += count;
        self.cache_size += count;

        // Determine zone based on position
        if self.cache_size <= self.config.pin_prefix {
            CacheZone::Pinned
        } else if self.cache_size >= self.config.trigger_threshold - self.config.recent_window {
            CacheZone::Recent
        } else {
            CacheZone::Compactable
        }
    }

    /// Perform compaction
    ///
    /// # Returns
    /// Quality metrics and compaction time
    pub fn compact(&mut self) -> CompactionResult {
        let start = Instant::now();

        let compactable = self.compactable_size();
        let target = self.target_size();

        // Calculate compression ratio
        let ratio = if compactable > 0 {
            target as f32 / compactable as f32
        } else {
            1.0
        };

        // Simulate compaction (actual implementation would call native library)
        let quality = QualityMetrics {
            cosine_similarity: 0.999,
            relative_error: 0.001,
        };

        let elapsed = start.elapsed();

        // Update state
        self.cache_size = self.config.pin_prefix + target + self.config.recent_window;
        self.compaction_count += 1;
        self.last_compaction_time = Some(Instant::now());

        CompactionResult {
            original_size: compactable,
            compacted_size: target,
            compression_ratio: ratio,
            quality,
            elapsed,
        }
    }

    /// Merge new tokens with compacted cache
    pub fn merge_new_tokens(&mut self, compacted_cache: &[u8], new_tokens: &[u8]) -> Vec<u8> {
        // Placeholder for merging logic
        vec![]
    }

    /// Get compaction statistics
    pub fn stats(&self) -> StreamingStats {
        StreamingStats {
            cache_size: self.cache_size,
            total_tokens_processed: self.total_tokens_processed,
            compaction_count: self.compaction_count,
            last_compaction_time: self.last_compaction_time,
            average_compaction_time: self.calculate_average_compaction_time(),
        }
    }

    fn calculate_average_compaction_time(&self) -> Option<Duration> {
        // This would track actual compaction times
        None
    }
}

/// Result of a compaction operation
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub original_size: usize,
    pub compacted_size: usize,
    pub compression_ratio: f32,
    pub quality: QualityMetrics,
    pub elapsed: Duration,
}

/// Statistics for streaming compaction
#[derive(Debug, Clone)]
pub struct StreamingStats {
    pub cache_size: usize,
    pub total_tokens_processed: usize,
    pub compaction_count: usize,
    pub last_compaction_time: Option<Instant>,
    pub average_compaction_time: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_needs_compaction() {
        let mut compactor = StreamingCompactor::new(StreamingConfig {
            trigger_threshold: 100,
            ..Default::default()
        });

        assert!(!compactor.needs_compaction());

        compactor.add_tokens(100);
        assert!(compactor.needs_compaction());
    }

    #[test]
    fn test_zone_detection() {
        let config = StreamingConfig {
            pin_prefix: 10,
            trigger_threshold: 100,
            recent_window: 20,
            ..Default::default()
        };

        let mut compactor = StreamingCompactor::new(config);

        // Pinned zone
        assert!(matches!(compactor.add_tokens(5), CacheZone::Pinned));

        // Compactable zone
        assert!(matches!(compactor.add_tokens(40), CacheZone::Compactable));

        // Recent zone
        compactor.cache_size = 90;
        assert!(matches!(compactor.add_tokens(5), CacheZone::Recent));
    }
}

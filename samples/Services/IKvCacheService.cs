using KVCompact.Shared.Models;

namespace KVCompact.Shared.Services;

/// <summary>
/// Service for managing KV cache compaction in LLM applications
/// </summary>
public interface IKvCacheService
{
    /// <summary>
    /// Compact the KV cache for the given session
    /// </summary>
    Task<CompactionResult> CompactCacheAsync(
        string sessionId,
        int originalTokenCount,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get compaction metrics for a session
    /// </summary>
    Task<CompactionMetrics?> GetMetricsAsync(string sessionId);

    /// <summary>
    /// Get recent compaction history
    /// </summary>
    Task<List<CompactionHistoryItem>> GetHistoryAsync(
        int count = 10,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Configure compaction parameters
    /// </summary>
    Task ConfigureAsync(CompactionConfig config);

    /// <summary>
    /// Get optimal compaction ratio based on context length
    /// </summary>
    double GetOptimalRatio(int tokenCount);

    /// <summary>
    /// Estimate memory savings from compaction
    /// </summary>
    double EstimateMemorySavings(int originalTokens, int compactedTokens);
}

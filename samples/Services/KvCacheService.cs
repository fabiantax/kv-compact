using Microsoft.Extensions.Logging;
using KVCompact.Shared.Models;

namespace KVCompact.Shared.Services;

/// <summary>
/// Implementation of KV cache compaction service using native interop
/// </summary>
public class KvCacheService : IKvCacheService
{
    private readonly ILogger<KvCacheService> _logger;
    private readonly CompactionConfig _config;
    private readonly Dictionary<string, CompactionMetrics> _sessionMetrics;
    private readonly List<CompactionHistoryItem> _history;

    public KvCacheService(ILogger<KvCacheService> logger)
    {
        _logger = logger;
        _config = new CompactionConfig();
        _sessionMetrics = new Dictionary<string, CompactionMetrics>();
        _history = new List<CompactionHistoryItem>();
    }

    public async Task<CompactionResult> CompactCacheAsync(
        string sessionId,
        int originalTokenCount,
        CancellationToken cancellationToken = default)
    {
        var startTime = DateTime.UtcNow;

        _logger.LogInformation("Starting compaction for session {SessionId} with {TokenCount} tokens",
            sessionId, originalTokenCount);

        try
        {
            // Calculate target token count
            var ratio = GetOptimalRatio(originalTokenCount);
            var targetTokens = Math.Max(16, (int)(originalTokenCount * ratio));

            // Call native compaction library
            var result = await CallNativeCompactionAsync(
                sessionId,
                originalTokenCount,
                targetTokens,
                cancellationToken);

            var elapsed = (DateTime.UtcNow - startTime).TotalMilliseconds;

            var metrics = new CompactionMetrics
            {
                SessionId = sessionId,
                OriginalTokenCount = originalTokenCount,
                CompactedTokenCount = targetTokens,
                CompressionRatio = (double)originalTokenCount / targetTokens,
                CompactionTimeMs = elapsed,
                QualityScore = result.CosineSimilarity,
                MemorySavedMB = EstimateMemorySavings(originalTokenCount, targetTokens),
                ProcessedLayers = result.ProcessedLayers,
                TotalLayers = result.TotalLayers,
                Timestamp = DateTime.UtcNow
            };

            _sessionMetrics[sessionId] = metrics;
            _history.Insert(0, new CompactionHistoryItem
            {
                SessionId = sessionId,
                Timestamp = metrics.Timestamp,
                CompressionRatio = metrics.CompressionRatio,
                QualityScore = metrics.QualityScore
            });

            _logger.LogInformation("Compaction completed: {Original} -> {Compacted} tokens in {TimeMs}ms",
                originalTokenCount, targetTokens, elapsed);

            return new CompactionResult
            {
                Success = true,
                Metrics = metrics,
                CompactedData = result.CompactedData
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Compaction failed for session {SessionId}", sessionId);
            return new CompactionResult
            {
                Success = false,
                Error = ex.Message
            };
        }
    }

    public Task<CompactionMetrics?> GetMetricsAsync(string sessionId)
    {
        return Task.FromResult(_sessionMetrics.TryGetValue(sessionId, out var metrics) ? metrics : null);
    }

    public Task<List<CompactionHistoryItem>> GetHistoryAsync(
        int count = 10,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(_history.Take(count).ToList());
    }

    public Task ConfigureAsync(CompactionConfig config)
    {
        _config.CompactRatio = config.CompactRatio;
        _config.MinTokenCount = config.MinTokenCount;
        _config.MaxTokenCount = config.MaxTokenCount;
        _config.QualityThreshold = config.QualityThreshold;
        return Task.CompletedTask;
    }

    public double GetOptimalRatio(int tokenCount)
    {
        // Adaptive ratio based on token count
        return tokenCount switch
        {
            < 100 => 0.5,      // 2x compression for small contexts
            < 1000 => 0.2,     // 5x compression for medium contexts
            < 10000 => 0.1,    // 10x compression for large contexts
            _ => 0.05          // 20x compression for very large contexts
        };
    }

    public double EstimateMemorySavings(int originalTokens, int compactedTokens)
    {
        // Estimate: ~16 bytes per token per layer (K+V in fp16)
        const double bytesPerTokenPerLayer = 16.0;
        const double layers = 36.0; // Qwen3-4B has 36 layers

        var originalMB = (originalTokens * bytesPerTokenPerLayer * layers) / (1024 * 1024);
        var compactedMB = (compactedTokens * bytesPerTokenPerLayer * layers) / (1024 * 1024);

        return originalMB - compactedMB;
    }

    private async Task<NativeCompactionResult> CallNativeCompactionAsync(
        string sessionId,
        int originalTokens,
        int targetTokens,
        CancellationToken cancellationToken)
    {
        // P/Invoke to native kv-compact library
        // This is a placeholder - actual implementation would call the native library
        return await Task.FromResult(new NativeCompactionResult
        {
            CosineSimilarity = 0.98,
            ProcessedLayers = 36,
            TotalLayers = 36,
            CompactedData = Array.Empty<byte>()
        });
    }

    private class NativeCompactionResult
    {
        public double CosineSimilarity { get; set; }
        public int ProcessedLayers { get; set; }
        public int TotalLayers { get; set; }
        public byte[] CompactedData { get; set; } = Array.Empty<byte>();
    }
}

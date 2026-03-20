# KV Compact - Sample Code

This directory contains example implementations and bindings for the KV cache compaction library in multiple languages.

## 📁 Directory Structure

```
samples/
├── BlazorComponents/     # ASP.NET Core Blazor components
│   ├── ChatMessage.razor
│   ├── StreamingText.razor
│   └── CompactionMonitor.razor
├── Services/              # C# service interfaces
│   ├── IKvCacheService.cs
│   └── KvCacheService.cs
├── Rust/                  # Rust implementation
│   └── src/
│       ├── lib.rs
│       └── streaming.rs
└── TypeScript/            # TypeScript/WASM bindings
    ├── src/
    │   ├── compaction.ts
    │   └── index.ts
    └── examples/
        └── browser-example.html
```

## 🚀 Quick Start

### C# / Blazor

```csharp
// Register the service
builder.Services.AddSingleton<IKvCacheService, KvCacheService>();

// Use in a component
@inject IKvCacheService KvCacheService

<button @onclick="CompactCache">Compact Cache</button>

@code {
    private async Task CompactCache()
    {
        var result = await KvCacheService.CompactCacheAsync(
            sessionId: "my-session",
            originalTokenCount: 1000
        );

        if (result.Success)
        {
            Console.WriteLine($"Compressed to {result.Metrics.CompactedTokenCount} tokens");
        }
    }
}
```

### Rust

```rust
use kv_compact::{KvCompactor, CompactConfig};

let config = CompactConfig {
    ratio: 0.2,
    min_tokens: 16,
    ..Default::default()
};

let compactor = KvCompactor::with_config(config);

let result = compactor.compact_layer(
    &keys,
    &values,
    &queries,
    num_tokens,
    num_heads,
    head_dim,
)?;

println!("Quality: {:.4}", result.quality.cosine_similarity);
```

### TypeScript

```typescript
import { KvCompactor, createCompactor } from './compaction';

// Create compactor with preset
const compactor = createCompactor('balanced');
await compactor.init();

// Compact a layer
const result = compactor.compactLayer(
    keys,
    values,
    queries,
    numTokens,
    numHeads,
    headDim,
);

if (result.success) {
    console.log(`Compression ratio: ${result.metrics.compressionRatio}x`);
}
```

## 📊 Language Comparison

| Feature | C# / Blazor | Rust | TypeScript / WASM |
|---------|-------------|------|-------------------|
| **Native Performance** | ✅ Yes | ✅ Yes | ⚠️ WASM overhead |
| **Streaming** | ✅ Supported | ✅ Supported | ✅ Supported |
| **Async/Await** | ✅ Native | ❌ Requires tokio | ✅ Native |
| **Memory Safety** | ✅ GC | ✅ Rust | ✅ WASM sandbox |
| **Browser Support** | ❌ Server only | ❌ Server only | ✅ Full |
| **Best For** | .NET apps | Systems programming | Web apps |

## 🔧 Configuration Examples

### Fast Compression (Lower Quality)

```csharp
var config = new CompactionConfig
{
    Ratio = 0.1,  // 10x compression
    QualityThreshold = 0.85
};
```

```rust
let config = CompactConfig {
    ratio: 0.1,
    min_tokens: 16,
    ..Default::default()
};
```

```typescript
const compactor = createCompactor('fast');
```

### High Quality (Less Compression)

```csharp
var config = new CompactionConfig
{
    Ratio = 0.3,  // 3.3x compression
    QualityThreshold = 0.98
};
```

```rust
let config = CompactConfig {
    ratio: 0.3,
    use_submodular: true,
    ..Default::default()
};
```

```typescript
const compactor = createCompactor('quality');
```

## 📈 Performance Characteristics

Based on Windows native tests with Qwen3-4B:

| Token Count | Compaction Time | Per Layer |
|-------------|-----------------|-----------|
| 100 | 96 ms | 2.7 ms |
| 1,000 | 10.8 s | 300 ms |
| 10,000 | ~108 s (est.) | ~3 s (est.) |

**Key Observations:**
- Compaction scales quadratically O(n²)
- Quality remains excellent (cos_sim > 0.99) even at 5x compression
- Memory savings: ~16 bytes per token per layer (fp16)

## 🎯 Use Cases

### 1. Long-Context Chat Applications

Use streaming compaction for conversations exceeding 8K tokens:

```typescript
const compactor = new StreamingCompactor({
  triggerThreshold: 8192,
  targetBudget: 4096,
  recentWindow: 512,
  pinPrefix: 256,  // System prompt
});
```

### 2. Document Analysis

For RAG applications with large document contexts:

```csharp
await kvCacheService.CompactCacheAsync(
    sessionId: documentId,
    originalTokenCount: documentTokens
);
```

### 3. Real-Time Agents

For agentic workflows with streaming generation:

```rust
let mut compactor = StreamingCompactor::new(config);
while streaming {
    compactor.add_tokens(new_tokens);
    if compactor.needs_compaction() {
        compactor.compact()?;
    }
}
```

## 🔗 Integration Examples

### With llama.cpp

```csharp
// Save KV state
var state = llama_state_seq_get_data(ctx);

// Compact
var compacted = await kvCacheService.CompactAsync(state);

// Write back
llama_state_seq_set_data(ctx, compacted);
```

### With LLM APIs

```typescript
// Store conversation history
const history = getConversationHistory();

// Compact before sending to API
const compacted = await compactor.compact(history);

// Send compacted context
const response = await llmApi.complete({
    context: compacted,
    max_tokens: 100
});
```

## 📚 Additional Resources

- **Paper**: "Fast KV Compaction via Attention Matching" (arXiv:2602.16284)
- **Main Library**: `../include/kv-compact-math.h`
- **CLI Tool**: `../src/kv-compact.cpp`
- **Tests**: `../tests/test-kv-compact-math.cpp`

## 🤝 Contributing

When adding bindings for new languages:

1. Follow the existing structure
2. Implement the core compaction interface
3. Add streaming support for long contexts
4. Include quality metrics
5. Provide usage examples

## 📄 License

Same as the main kv-compact project (MIT)

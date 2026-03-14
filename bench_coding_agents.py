"""
Realistic coding agent benchmark: 10 agents x 100K context.

Tests the serving throughput for a scenario where 10 coding agents each have
~100K tokens of context (codebase + instructions + conversation history).
Without compaction this requires 150-260 GB of KV cache -- impossible.
With compaction, we can fit all 10 agents and measure actual throughput.

We simulate compacted KV by using proportionally shorter prompts:
- 100K / 10x = 10K tokens per slot  (10x compaction)
- 100K / 20x = 5K tokens per slot   (20x compaction)
- 100K / 50x = 2K tokens per slot   (50x compaction)
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_BIN = r"C:\Users\fabia\Projects\llama.cpp\llama-flash-attn\build-win\bin\llama-server.exe"
HOST = "127.0.0.1"
PORT = 9200
N_PREDICT = 50
N_AGENTS = 10
TMP = os.environ.get('TEMP', '/tmp')

MODELS = {
    'Qwen3.5-4B': {
        'path': r'C:\Users\fabia\models\Qwen3.5-4B-Q4_K_M.gguf',
        'weights_gb': 2.7,
    },
    'Qwen3.5-9B': {
        'path': r'C:\Users\fabia\models\Qwen3.5-9B-Q4_K_M.gguf',
        'weights_gb': 5.7,
    },
    'Qwen3.5-35B-A3B': {
        'path': r'C:\Users\fabia\models\Qwen3.5-35B-A3B-Q4_K_M.gguf',
        'weights_gb': 20.0,
    },
}

BASE_CONTEXT = 100_000

# Bottom-up: start with most compacted (most likely to succeed), work up.
# This way we get results fast and can see where OOM starts.
CONFIGS = [
    # Most compacted first (guaranteed to fit)
    ("50x",            50,     2000, 10,   4096),
    ("20x",            20,     5000, 10,   8192),
    ("10x",            10,    10000, 10,  12288),
    ("5x",              5,    20000, 10,  24576),
    # Full context baselines (will OOM for most models -- that's the point)
    ("1x (1 agent)",    1,   100000, 1,  131072),
]


def generate_coding_prompt(n_tokens_target):
    """Generate a realistic coding agent prompt of approximately n_tokens_target tokens.

    ~2.3 tokens per word for code-heavy content (Qwen tokenizer).
    """
    n_words = int(n_tokens_target / 2.3)

    system_prompt = """You are a senior software engineer working on a large distributed system.
Your task is to review the codebase, understand the architecture, and implement the requested changes.
Follow the existing code style, add appropriate error handling, and ensure backward compatibility.

## Current Codebase Context

The following files are part of the project you're working on:
"""

    code_blocks = [
        '''
```python
# service/api/handlers.py
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class RequestContext:
    request_id: str
    user_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_trace(self, span_name: str) -> 'RequestContext':
        return RequestContext(
            request_id=self.request_id,
            user_id=self.user_id,
            timestamp=self.timestamp,
            metadata={**self.metadata, 'span': span_name}
        )

class ServiceHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self._lock = asyncio.Lock()
        self.metrics = MetricsCollector()

    async def handle_request(self, ctx: RequestContext, payload: Dict) -> Dict:
        logger.info(f"Processing request {ctx.request_id}")
        async with self._lock:
            if ctx.request_id in self.cache:
                self.metrics.record_cache_hit()
                return self.cache[ctx.request_id]
        result = await self._process(ctx, payload)
        async with self._lock:
            self.cache[ctx.request_id] = result
        return result

    async def _process(self, ctx: RequestContext, payload: Dict) -> Dict:
        start = time.monotonic()
        try:
            validated = self._validate(payload)
            transformed = await self._transform(ctx, validated)
            stored = await self._store(ctx, transformed)
            self.metrics.record_latency(time.monotonic() - start)
            return {"status": "ok", "data": stored}
        except ValidationError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Processing failed for {ctx.request_id}: {e}")
            raise
```
''',
        '''
```rust
// src/network/connection_pool.rs
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tokio::net::TcpStream;
use tokio::time::{timeout, Duration};

pub struct ConnectionPool {
    connections: Arc<RwLock<HashMap<String, Vec<PooledConnection>>>>,
    semaphore: Arc<Semaphore>,
    config: PoolConfig,
}

struct PooledConnection {
    stream: TcpStream,
    created_at: std::time::Instant,
    last_used: std::time::Instant,
    in_use: bool,
}

#[derive(Clone)]
pub struct PoolConfig {
    pub max_connections_per_host: usize,
    pub max_total_connections: usize,
    pub idle_timeout: Duration,
    pub connect_timeout: Duration,
}

impl ConnectionPool {
    pub fn new(config: PoolConfig) -> Self {
        ConnectionPool {
            connections: Arc::new(RwLock::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(config.max_total_connections)),
            config,
        }
    }

    pub async fn get(&self, host: &str) -> Result<PoolGuard, PoolError> {
        let _permit = self.semaphore.acquire().await
            .map_err(|_| PoolError::Exhausted)?;
        {
            let mut conns = self.connections.write().await;
            if let Some(pool) = conns.get_mut(host) {
                if let Some(conn) = pool.iter_mut().find(|c| !c.in_use) {
                    conn.in_use = true;
                    conn.last_used = std::time::Instant::now();
                    return Ok(PoolGuard { });
                }
            }
        }
        let stream = timeout(self.config.connect_timeout, TcpStream::connect(host))
            .await.map_err(|_| PoolError::Timeout)?
            .map_err(PoolError::Connect)?;
        let conn = PooledConnection {
            stream, created_at: std::time::Instant::now(),
            last_used: std::time::Instant::now(), in_use: true,
        };
        let mut conns = self.connections.write().await;
        conns.entry(host.to_string()).or_default().push(conn);
        Ok(PoolGuard { })
    }
}
```
''',
        '''
```typescript
// src/components/Dashboard.tsx
import React, { useState, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

interface MetricPoint {
  timestamp: number; value: number; label: string;
  metadata?: Record<string, unknown>;
}
interface DashboardProps {
  projectId: string;
  timeRange: { start: Date; end: Date };
  refreshInterval?: number;
}

const Dashboard: React.FC<DashboardProps> = ({ projectId, timeRange, refreshInterval = 30000 }) => {
  const queryClient = useQueryClient();
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [alertThresholds, setAlertThresholds] = useState<Record<string, number>>({});
  const { data: metrics, isLoading, error } = useQuery({
    queryKey: ['metrics', projectId, timeRange],
    queryFn: () => fetchMetrics(projectId, timeRange),
    refetchInterval: refreshInterval, staleTime: 10000,
  });
  const updateThreshold = useMutation({
    mutationFn: (params: { metric: string; threshold: number }) =>
      api.put(`/projects/${projectId}/thresholds`, params),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['metrics', projectId] }),
  });
  const processedData = useMemo(() => {
    if (!metrics) return [];
    return metrics.map((m: MetricPoint) => ({
      ...m, normalized: m.value / (alertThresholds[m.label] || 1),
      isAlert: m.value > (alertThresholds[m.label] || Infinity),
    }));
  }, [metrics, alertThresholds]);
  if (isLoading) return <LoadingSkeleton />;
  if (error) return <ErrorBoundary error={error} />;
  return (
    <div className="dashboard-container">
      <MetricGrid data={processedData} selected={selectedMetric}
        onSelect={useCallback((m: string) => setSelectedMetric(p => p === m ? null : m), [])} />
      {selectedMetric && <DetailPanel metric={selectedMetric}
        data={processedData.filter(d => d.label === selectedMetric)}
        threshold={alertThresholds[selectedMetric]}
        onThresholdChange={(v) => updateThreshold.mutate({ metric: selectedMetric, threshold: v })} />}
    </div>
  );
};
```
''',
        '''
```go
// internal/middleware/ratelimit.go
package middleware

import (
    "context"; "fmt"; "net/http"; "sync"; "time"
    "golang.org/x/time/rate"
)

type RateLimiter struct {
    visitors map[string]*visitor
    mu       sync.RWMutex
    rate     rate.Limit
    burst    int
}
type visitor struct { limiter *rate.Limiter; lastSeen time.Time }

func NewRateLimiter(r rate.Limit, burst int) *RateLimiter {
    rl := &RateLimiter{visitors: make(map[string]*visitor), rate: r, burst: burst}
    go rl.cleanupLoop()
    return rl
}

func (rl *RateLimiter) getVisitor(ip string) *rate.Limiter {
    rl.mu.RLock()
    v, exists := rl.visitors[ip]
    rl.mu.RUnlock()
    if exists { v.lastSeen = time.Now(); return v.limiter }
    rl.mu.Lock(); defer rl.mu.Unlock()
    limiter := rate.NewLimiter(rl.rate, rl.burst)
    rl.visitors[ip] = &visitor{limiter: limiter, lastSeen: time.Now()}
    return limiter
}

func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        limiter := rl.getVisitor(extractIP(r))
        if !limiter.Allow() {
            w.Header().Set("Retry-After", fmt.Sprintf("%d", int(time.Second/time.Duration(rl.rate))))
            http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
            return
        }
        ctx := context.WithValue(r.Context(), rateLimitKey, &RateLimitInfo{
            Remaining: limiter.Tokens(), Limit: float64(rl.rate),
        })
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
```
''',
    ]

    parts = [system_prompt]
    words_so_far = len(system_prompt.split())
    module_num = 1

    while words_so_far < n_words:
        for block in code_blocks:
            if words_so_far >= n_words:
                break
            header = f"\n### Module {module_num}: Service Component {module_num}\n"
            parts.append(header)
            parts.append(block)
            words_so_far += len(header.split()) + len(block.split())
            module_num += 1

    parts.append("\n## Task\nReview the codebase and implement error handling, circuit breakers, and tracing.\n")
    full_text = '\n'.join(parts)
    words = full_text.split()[:n_words]
    return ' '.join(words)


def start_server(model_path, n_slots, ctx_per_slot, extra_args=None):
    """Start llama-server with Q8_0 KV cache quantization and wait for ready."""
    total_ctx = n_slots * ctx_per_slot
    cmd = [
        SERVER_BIN, "-m", model_path,
        "-c", str(total_ctx), "-np", str(n_slots),
        "-cb", "-ngl", "99", "--no-warmup",
        "-ctk", "q8_0", "-ctv", "q8_0",  # quantize KV cache to halve memory
        "--host", HOST, "--port", str(PORT),
    ]
    if extra_args:
        cmd.extend(extra_args)

    log_path = os.path.join(TMP, "bench_agents.log")
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )

    for _ in range(180):
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2) as r:
                h = json.loads(r.read())
                if h.get('status') == 'ok':
                    return proc
        except:
            pass
        # Check if process died (OOM)
        if proc.poll() is not None:
            return None
        time.sleep(1)

    proc.kill()
    return None


def send_completion(payload_bytes):
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/completion",
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def bench_concurrent(payload_bytes, n_concurrent):
    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(send_completion, payload_bytes) for _ in range(n_concurrent)]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.perf_counter() - t0

    total_tokens = 0
    errors = 0
    per_slot_tps = []
    for r in results:
        t = r.get('timings', {})
        n = t.get('predicted_n', 0)
        tps = t.get('predicted_per_second', 0)
        if n > 0:
            total_tokens += n
            per_slot_tps.append(tps)
        else:
            errors += 1

    agg_tps = total_tokens / elapsed if elapsed > 0 else 0
    avg_slot = sum(per_slot_tps) / len(per_slot_tps) if per_slot_tps else 0
    return {
        'total_tokens': total_tokens,
        'elapsed_s': elapsed,
        'agg_tps': agg_tps,
        'errors': errors,
        'per_slot_tps': per_slot_tps,
        'avg_slot_tps': avg_slot,
    }


def kill_server():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True, timeout=10)
    except:
        pass
    time.sleep(3)


def main():
    selected_models = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    print("=" * 78)
    print("  CODING AGENT BENCHMARK: 10 Agents x 100K Context")
    print("  KV cache: Q8_0 quantized | Simulating post-compaction sizes")
    print("=" * 78)
    print()

    # Generate prompts (bottom-up: smallest first)
    prompt_cache = {}
    for label, ratio, target_kv, n_slots, ctx_per_slot in CONFIGS:
        if target_kv not in prompt_cache:
            print(f"  Generating ~{target_kv//1000}K-token coding prompt...", end=" ", flush=True)
            text = generate_coding_prompt(target_kv)
            payload = json.dumps({
                'prompt': text,
                'n_predict': N_PREDICT,
                'temperature': 0.0,
                'cache_prompt': True,
            }).encode()
            prompt_cache[target_kv] = payload
            print(f"({len(text.split())} words)")
    print()

    for model_name in selected_models:
        if model_name not in MODELS:
            print(f"  Unknown model: {model_name}")
            continue

        model_info = MODELS[model_name]
        model_path = model_info['path']

        if not os.path.exists(model_path):
            print(f"  SKIP {model_name}: not found at {model_path}")
            print()
            continue

        print("=" * 78)
        print(f"  {model_name} (Q4_K_M, {model_info['weights_gb']:.1f} GB weights)")
        print(f"  10 coding agents, 100K original context each, KV cache Q8_0")
        print("=" * 78)
        hdr = (f"{'Compact':<16} {'Agents':>6} {'KV/agent':>9} {'TotalCtx':>9} "
               f"{'Agg tok/s':>10} {'Per-agent':>10} {'Err':>4}")
        print(hdr)
        print("-" * 78)

        for label, ratio, target_kv, n_slots, ctx_per_slot in CONFIGS:
            total_ctx = n_slots * ctx_per_slot
            if total_ctx >= 1024:
                total_str = f"{total_ctx // 1024}K"
            else:
                total_str = str(total_ctx)
            kv_str = f"{target_kv // 1000}K" if target_kv >= 1000 else str(target_kv)

            kill_server()
            print(f"{label:<16} {n_slots:>6} {kv_str:>9} {total_str:>9} ",
                  end="", flush=True)

            proc = start_server(model_path, n_slots, ctx_per_slot)
            if proc is None:
                print(f"{'--':>10} {'--':>10} {'':>4}  OOM")
                continue

            try:
                payload = prompt_cache[target_kv]
                send_completion(payload)  # warmup
                r = bench_concurrent(payload, n_slots)

                if r['errors'] == n_slots:
                    print(f"{'--':>10} {'--':>10} {n_slots:>4}  ALL_FAIL")
                else:
                    err_s = str(r['errors']) if r['errors'] else ""
                    print(f"{r['agg_tps']:>10.1f} {r['avg_slot_tps']:>10.1f} {err_s:>4}")
            finally:
                proc.kill()
                proc.wait()
                time.sleep(2)

        print()
        kill_server()

    print("=" * 78)
    print("  COMPLETE")
    print("=" * 78)


if __name__ == '__main__':
    main()

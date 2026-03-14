"""
Coding agent benchmark v2: practical multi-slot testing.

Tests each model at its feasible scale, reports per-slot speed at various
KV sizes (simulating compaction), and projects aggregate throughput.

Key insight: Qwen3.5 hybrid models (attn + DeltaNet) have limited n_seq_max.
We test at each model's actual limit and extrapolate for 10 agents.
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
TMP = os.environ.get('TEMP', '/tmp')

MODELS = {
    'Qwen3.5-4B': {
        'path': r'C:\Users\fabia\models\Qwen3.5-4B-Q4_K_M.gguf',
        'weights_gb': 2.7,
        'max_slots': 4,  # test up to this
    },
    'Qwen3.5-9B': {
        'path': r'C:\Users\fabia\models\Qwen3.5-9B-Q4_K_M.gguf',
        'weights_gb': 5.7,
        'max_slots': 4,
    },
    'Qwen3.5-35B-A3B': {
        'path': r'C:\Users\fabia\models\Qwen3.5-35B-A3B-Q4_K_M.gguf',
        'weights_gb': 20.0,
        'max_slots': 2,  # known limit
    },
}

# Test matrix: per-slot KV sizes simulating compaction of 100K context
# (ratio, kv_tokens, ctx_alloc) — ctx_alloc includes headroom for generation
KV_CONFIGS = [
    ("100K (1x)",    100000, 104000),
    ("50K  (2x)",     50000,  52000),
    ("20K  (5x)",     20000,  22000),
    ("10K  (10x)",    10000,  12000),
    ("5K   (20x)",     5000,   6000),
    ("2K   (50x)",     2000,   4000),
    ("1K   (100x)",    1000,   2048),
]


def generate_prompt(n_tokens_target):
    """Generate coding prompt targeting n_tokens_target tokens (~2.4 tok/word for Qwen)."""
    n_words = max(50, int(n_tokens_target / 2.4))

    base = """You are a senior engineer. Review this distributed system codebase:

```python
import asyncio, logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    name: str
    port: int = 8080
    workers: int = 4
    timeout: float = 30.0
    retry_limit: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConnectionPool:
    def __init__(self, max_size: int = 100):
        self._pool: Dict[str, List] = {}
        self._lock = asyncio.Lock()
        self._max = max_size

    async def acquire(self, host: str):
        async with self._lock:
            if host in self._pool and self._pool[host]:
                return self._pool[host].pop()
        return await self._create(host)

    async def release(self, host: str, conn):
        async with self._lock:
            self._pool.setdefault(host, []).append(conn)

class RequestHandler:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.pool = ConnectionPool()
        self.metrics = {}

    async def handle(self, method: str, path: str, body: Optional[bytes] = None):
        start = asyncio.get_event_loop().time()
        try:
            conn = await self.pool.acquire(f"{self.config.name}:{self.config.port}")
            response = await asyncio.wait_for(
                self._send(conn, method, path, body),
                timeout=self.config.timeout
            )
            await self.pool.release(f"{self.config.name}:{self.config.port}", conn)
            elapsed = asyncio.get_event_loop().time() - start
            self._record_metric(path, elapsed, True)
            return response
        except asyncio.TimeoutError:
            self._record_metric(path, self.config.timeout, False)
            raise
        except Exception as e:
            logger.error(f"Request failed: {method} {path}: {e}")
            raise

    def _record_metric(self, path, latency, success):
        key = f"{path}:{'ok' if success else 'err'}"
        if key not in self.metrics:
            self.metrics[key] = {'count': 0, 'total_ms': 0}
        self.metrics[key]['count'] += 1
        self.metrics[key]['total_ms'] += latency * 1000
```

```rust
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct CacheManager {
    data: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_entries: usize,
    ttl_seconds: u64,
}

struct CacheEntry {
    value: Vec<u8>,
    created_at: std::time::Instant,
    access_count: u64,
}

impl CacheManager {
    pub fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        CacheManager {
            data: Arc::new(RwLock::new(HashMap::new())),
            max_entries, ttl_seconds,
        }
    }

    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let mut data = self.data.write().await;
        if let Some(entry) = data.get_mut(key) {
            if entry.created_at.elapsed().as_secs() < self.ttl_seconds {
                entry.access_count += 1;
                return Some(entry.value.clone());
            }
            data.remove(key);
        }
        None
    }

    pub async fn set(&self, key: String, value: Vec<u8>) {
        let mut data = self.data.write().await;
        if data.len() >= self.max_entries {
            self.evict_lru(&mut data);
        }
        data.insert(key, CacheEntry {
            value, created_at: std::time::Instant::now(), access_count: 0,
        });
    }

    fn evict_lru(&self, data: &mut HashMap<String, CacheEntry>) {
        if let Some(key) = data.iter()
            .min_by_key(|(_, e)| e.access_count)
            .map(|(k, _)| k.clone())
        {
            data.remove(&key);
        }
    }
}
```

Implement circuit breakers, health checks, and graceful shutdown for all services above.
"""

    words = base.split()
    if len(words) >= n_words:
        return ' '.join(words[:n_words])

    # Pad by repeating with variations
    result_words = list(words)
    iteration = 1
    while len(result_words) < n_words:
        result_words.append(f"\n\n### Extension Module {iteration}\n")
        for w in words[50:]:  # skip the system prompt part
            result_words.append(w)
            if len(result_words) >= n_words:
                break
        iteration += 1

    return ' '.join(result_words[:n_words])


def start_server(model_path, n_slots, ctx_per_slot, timeout_s=120):
    total_ctx = n_slots * ctx_per_slot
    cmd = [
        SERVER_BIN, "-m", model_path,
        "-c", str(total_ctx), "-np", str(n_slots),
        "-cb", "-ngl", "99", "--no-warmup",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--host", HOST, "--port", str(PORT),
    ]
    log_path = os.path.join(TMP, "bench_agents2.log")
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)

    for _ in range(timeout_s):
        if proc.poll() is not None:
            return None  # died (OOM)
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2) as r:
                h = json.loads(r.read())
                if h.get('status') == 'ok':
                    return proc
        except:
            pass
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


def bench(payload_bytes, n_concurrent):
    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(send_completion, payload_bytes) for _ in range(n_concurrent)]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.perf_counter() - t0

    total_tok = sum(r.get('timings', {}).get('predicted_n', 0) for r in results)
    errors = sum(1 for r in results if r.get('timings', {}).get('predicted_n', 0) == 0)
    per_slot = [r.get('timings', {}).get('predicted_per_second', 0)
                for r in results if r.get('timings', {}).get('predicted_n', 0) > 0]
    agg = total_tok / elapsed if elapsed > 0 else 0
    avg = sum(per_slot) / len(per_slot) if per_slot else 0
    return agg, avg, errors


def kill_server():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                       capture_output=True, timeout=10)
    except:
        pass
    time.sleep(3)


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    print("=" * 80)
    print("  CODING AGENT BENCHMARK v2: Qwen3.5 Hybrid Models")
    print("  KV cache Q8_0 | Simulating 100K context at various compaction ratios")
    print("=" * 80)

    # Pre-generate all prompts
    prompt_cache = {}
    for label, kv_tok, ctx_alloc in KV_CONFIGS:
        if kv_tok not in prompt_cache:
            text = generate_prompt(kv_tok)
            prompt_cache[kv_tok] = json.dumps({
                'prompt': text,
                'n_predict': N_PREDICT,
                'temperature': 0.0,
                'cache_prompt': False,  # avoid cache confusion
            }).encode()
    print(f"  Prompts ready: {', '.join(f'{k//1000}K' for k in sorted(prompt_cache.keys()))}")
    print()

    for model_name in selected:
        if model_name not in MODELS:
            continue
        info = MODELS[model_name]
        if not os.path.exists(info['path']):
            print(f"  SKIP {model_name}: not found")
            continue

        max_slots = info['max_slots']

        print("=" * 80)
        print(f"  {model_name} ({info['weights_gb']:.1f} GB) | max tested slots: {max_slots}")
        print("=" * 80)

        # Phase 1: Per-slot speed at different KV sizes (1 slot)
        print()
        print(f"  Phase 1: Per-slot generation speed (1 slot, varying KV size)")
        print(f"  {'KV size':<14} {'Per-slot tok/s':>14} {'Status':<12}")
        print(f"  {'-'*14} {'-'*14} {'-'*12}")

        single_slot_speeds = {}

        for label, kv_tok, ctx_alloc in KV_CONFIGS:
            kill_server()
            print(f"  {label:<14} ", end="", flush=True)

            proc = start_server(info['path'], 1, ctx_alloc, timeout_s=180)
            if proc is None:
                print(f"{'--':>14} {'OOM':>12}")
                continue

            try:
                payload = prompt_cache[kv_tok]
                send_completion(payload)  # warmup
                agg, avg, errors = bench(payload, 1)
                if errors:
                    print(f"{'--':>14} {'FAIL':>12}")
                else:
                    print(f"{avg:>14.1f} {'OK':>12}")
                    single_slot_speeds[kv_tok] = avg
            finally:
                proc.kill()
                proc.wait()

        # Phase 2: Multi-slot throughput at compacted sizes
        print()
        print(f"  Phase 2: Multi-slot throughput ({max_slots} slots, compacted KV)")
        print(f"  {'KV size':<14} {'Slots':>5} {'Agg tok/s':>10} {'Per-slot':>10} {'Status':<8}")
        print(f"  {'-'*14} {'-'*5} {'-'*10} {'-'*10} {'-'*8}")

        multi_results = {}

        for label, kv_tok, ctx_alloc in KV_CONFIGS:
            if kv_tok > 20000:  # skip huge contexts for multi-slot
                continue
            kill_server()
            print(f"  {label:<14} {max_slots:>5} ", end="", flush=True)

            proc = start_server(info['path'], max_slots, ctx_alloc, timeout_s=180)
            if proc is None:
                print(f"{'--':>10} {'--':>10} {'OOM':>8}")
                continue

            try:
                payload = prompt_cache[kv_tok]
                send_completion(payload)  # warmup
                agg, avg, errors = bench(payload, max_slots)
                if errors >= max_slots:
                    print(f"{'--':>10} {'--':>10} {'FAIL':>8}")
                else:
                    err_s = f"{errors}err" if errors else "OK"
                    print(f"{agg:>10.1f} {avg:>10.1f} {err_s:>8}")
                    multi_results[kv_tok] = (agg, avg)
            finally:
                proc.kill()
                proc.wait()

        # Phase 3: Projection for 10 agents
        print()
        print(f"  Phase 3: Projected throughput for 10 coding agents x 100K context")
        print(f"  {'Compaction':<14} {'1-slot tok/s':>12} {'Proj 10-agent':>14} {'vs 1x':>8}")
        print(f"  {'-'*14} {'-'*12} {'-'*14} {'-'*8}")

        baseline = single_slot_speeds.get(100000, single_slot_speeds.get(50000, 0))

        for label, kv_tok, ctx_alloc in KV_CONFIGS:
            speed = single_slot_speeds.get(kv_tok)
            if speed is None:
                continue
            # Conservative projection: 10 agents, assume ~70% efficiency at 10 slots
            proj = speed * 10 * 0.7
            ratio = proj / (baseline * 10 * 0.7) if baseline > 0 else 0
            ratio_s = f"{ratio:.1f}x" if baseline > 0 else "--"
            print(f"  {label:<14} {speed:>12.1f} {proj:>14.1f} {ratio_s:>8}")

        print()
        kill_server()

    print("=" * 80)
    print("  COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

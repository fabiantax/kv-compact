"""
Coding agent benchmark v3: simple and reliable.

One server, one slot, large fixed context. Vary prompt length to measure
how KV cache size affects per-token generation speed. This directly shows
the speed benefit of compaction without any OOM/slot complications.

Then project aggregate throughput for 10 agents based on measured speeds.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

SERVER_BIN = r"C:\Users\fabia\Projects\llama.cpp\llama-flash-attn\build-win\bin\llama-server.exe"
HOST = "127.0.0.1"
PORT = 9200
N_PREDICT = 50
TMP = os.environ.get('TEMP', '/tmp')

MODELS = {
    'Qwen3.5-4B': {
        'path': r'C:\Users\fabia\models\Qwen3.5-4B-Q4_K_M.gguf',
        'weights_gb': 2.7,
        'ctx_size': 65536,  # fixed context allocation
    },
    'Qwen3.5-9B': {
        'path': r'C:\Users\fabia\models\Qwen3.5-9B-Q4_K_M.gguf',
        'weights_gb': 5.7,
        'ctx_size': 32768,
    },
    'Qwen3.5-35B-A3B': {
        'path': r'C:\Users\fabia\models\Qwen3.5-35B-A3B-Q4_K_M.gguf',
        'weights_gb': 20.0,
        'ctx_size': 32768,
    },
}

# Word counts targeting specific token counts (at ~2.4 tok/word for Qwen)
# We'll use the server's actual token count from the response
PROMPT_WORD_COUNTS = [
    ("~1K",     400),
    ("~2K",     800),
    ("~5K",    2000),
    ("~10K",   4000),
    ("~20K",   8000),
]


def generate_prompt(n_words):
    """Generate a coding prompt with approximately n_words words."""
    base = """You are a senior engineer reviewing this distributed system codebase.

```python
import asyncio, logging, time
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

struct CacheEntry { value: Vec<u8>, created_at: std::time::Instant, access_count: u64 }

impl CacheManager {
    pub fn new(max_entries: usize, ttl_seconds: u64) -> Self {
        CacheManager { data: Arc::new(RwLock::new(HashMap::new())), max_entries, ttl_seconds }
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
        if data.len() >= self.max_entries { self.evict_lru(&mut data); }
        data.insert(key, CacheEntry { value, created_at: std::time::Instant::now(), access_count: 0 });
    }
}
```

Implement circuit breakers and health checks for the above services.
"""

    words = base.split()
    if len(words) >= n_words:
        return ' '.join(words[:n_words])

    result = list(words)
    i = 1
    while len(result) < n_words:
        result.append(f"\n### Module {i}\n")
        result.extend(words[20:])  # repeat without preamble
        i += 1

    return ' '.join(result[:n_words])


def start_server(model_path, ctx_size, timeout_s=120):
    cmd = [
        SERVER_BIN, "-m", model_path,
        "-c", str(ctx_size), "-np", "1",
        "-ngl", "99", "--no-warmup",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--host", HOST, "--port", str(PORT),
    ]
    log_path = os.path.join(TMP, "bench_agents3.log")
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)

    for _ in range(timeout_s):
        if proc.poll() is not None:
            return None
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


def send_completion(text, n_predict=50):
    payload = json.dumps({
        'prompt': text,
        'n_predict': n_predict,
        'temperature': 0.0,
        'cache_prompt': False,
    }).encode()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


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
    print("  CODING AGENT BENCHMARK v3: Per-slot Speed vs KV Cache Size")
    print("  How much faster is generation after compaction?")
    print("=" * 80)
    print()

    # Pre-generate prompts
    prompts = {}
    for label, n_words in PROMPT_WORD_COUNTS:
        prompts[label] = generate_prompt(n_words)
    print(f"  Prompts: {', '.join(f'{l} ({len(prompts[l].split())}w)' for l, _ in PROMPT_WORD_COUNTS)}")
    print()

    for model_name in selected:
        if model_name not in MODELS:
            continue
        info = MODELS[model_name]
        if not os.path.exists(info['path']):
            print(f"  SKIP {model_name}: not found")
            continue

        kill_server()
        print("=" * 80)
        print(f"  {model_name} ({info['weights_gb']:.1f} GB, ctx={info['ctx_size']//1024}K, KV Q8_0)")
        print("=" * 80)
        print()

        # Start one server, keep it running for all prompt sizes
        print(f"  Starting server...", end=" ", flush=True)
        proc = start_server(info['path'], info['ctx_size'])
        if proc is None:
            print("FAILED (OOM?)")
            continue
        print("ready.")
        print()

        print(f"  {'Prompt':<8} {'KV tokens':>10} {'Prefill tok/s':>14} {'Gen tok/s':>11} {'Gen tokens':>11}")
        print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*11} {'-'*11}")

        results = []
        for label, n_words in PROMPT_WORD_COUNTS:
            text = prompts[label]
            print(f"  {label:<8} ", end="", flush=True)

            # Warmup (short)
            send_completion("Hello", n_predict=5)

            # Actual benchmark
            r = send_completion(text, n_predict=N_PREDICT)

            if 'error' in r:
                print(f"{'--':>10} {'FAIL':>14}")
                continue

            t = r.get('timings', {})
            prompt_n = t.get('prompt_n', 0)
            prompt_tps = t.get('prompt_per_second', 0)
            gen_n = t.get('predicted_n', 0)
            gen_tps = t.get('predicted_per_second', 0)

            if gen_n == 0:
                print(f"{prompt_n:>10} {'FAIL':>14}")
                continue

            print(f"{prompt_n:>10} {prompt_tps:>14.1f} {gen_tps:>11.1f} {gen_n:>11}")
            results.append((label, prompt_n, gen_tps))

        proc.kill()
        proc.wait()

        # Projection
        if len(results) >= 2:
            print()
            print(f"  --- Projection: 10 Coding Agents x 100K Context ---")
            print(f"  Original context: 100K tokens per agent")
            print()
            print(f"  {'Compaction':<12} {'KV/agent':>10} {'tok/s/agent':>12} "
                  f"{'10-agent agg':>13} {'vs baseline':>12}")
            print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*13} {'-'*12}")

            baseline_tps = results[-1][2]  # largest context = baseline (worst case)

            for label, prompt_n, gen_tps in results:
                ratio = 100000 / max(prompt_n, 1)
                ratio_label = f"{ratio:.0f}x" if ratio >= 2 else "1x"
                # 10 agents at 70% batching efficiency
                agg_10 = gen_tps * 10 * 0.7
                speedup = gen_tps / baseline_tps if baseline_tps > 0 else 0
                print(f"  {ratio_label:<12} {prompt_n:>10} {gen_tps:>12.1f} "
                      f"{agg_10:>13.1f} {speedup:>11.1f}x")

            # Memory comparison
            print()
            print(f"  --- Memory Analysis ---")
            print(f"  Full 100K x 10 agents (F16 KV): impossible (>100 GB KV alone)")
            print(f"  With 10x compaction (10K KV): ~10x less KV memory")
            print(f"  With 50x compaction (2K KV):  ~50x less KV memory")

        print()
        kill_server()

    print("=" * 80)
    print("  COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

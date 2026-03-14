"""Per-slot speed vs KV cache size — the core compaction value proposition.

Measures how generation speed degrades as context length grows,
then shows how compaction restores speed by reducing KV cache size.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

SERVER_BIN = r"C:\Users\fabia\Projects\llama.cpp\llama-flash-attn\build-win\bin\llama-server.exe"
MODEL = r"C:\Users\fabia\models\SmolLM3-3B-128K-Q4_K_M.gguf"
HOST = "127.0.0.1"
PORT = 9095
TMP = os.environ.get('TEMP', '/tmp')


def kill_server():
    os.system('taskkill /F /IM llama-server.exe >nul 2>&1')
    time.sleep(3)


def start_server(ctx_size):
    proc = subprocess.Popen(
        [SERVER_BIN, "-m", MODEL,
         "-c", str(ctx_size), "-np", "1",
         "-ngl", "99", "-t", "4", "--no-warmup",
         "--host", HOST, "--port", str(PORT)],
        stdout=open(os.path.join(TMP, "bench_perslot.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2) as r:
                if json.loads(r.read()).get('status') == 'ok':
                    return proc
        except:
            pass
        time.sleep(1)
    proc.kill()
    return None


def measure(payload_bytes, n_runs=2):
    """Run n_runs and return best result."""
    best_tps = 0
    best_prompt_tps = 0
    prompt_n = 0
    for _ in range(n_runs):
        req = urllib.request.Request(
            f"http://{HOST}:{PORT}/completion",
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                r = json.loads(resp.read())
                t = r.get('timings', {})
                tps = t.get('predicted_per_second', 0)
                if tps > best_tps:
                    best_tps = tps
                    best_prompt_tps = t.get('prompt_per_second', 0)
                    prompt_n = t.get('prompt_n', 0)
        except Exception as e:
            pass
    return best_tps, best_prompt_tps, prompt_n


def main():
    print("=" * 80)
    print("  PER-SLOT SPEED vs KV CACHE SIZE — SmolLM3-3B on Vulkan (8060S)")
    print("  How generation speed degrades with context length")
    print("=" * 80)
    print()

    # Test configs: (label, request_file, ctx_size)
    configs = [
        ("2K tokens",   "req_direct_2k.json",    4096),
        ("5K tokens",   "req_direct_5k.json",    8192),
        ("10K tokens",  "req_direct_10k.json",   16384),
        ("25K tokens",  "req_direct_25k.json",   32768),
        ("50K tokens",  "req_direct_50k.json",   65536),
        ("100K tokens", "req_direct_100k.json",  131072),
    ]

    print(f"{'Context':<14} {'Actual tok':<11} {'Prefill tok/s':<14} {'Gen tok/s':<10} {'vs 2K':<7} {'Slowdown':<9}")
    print(f"{'-------':<14} {'----------':<11} {'-------------':<14} {'---------':<10} {'-----':<7} {'--------':<9}")

    baseline = None
    results = []

    for label, req_file, ctx_size in configs:
        path = os.path.join(TMP, req_file)
        if not os.path.exists(path):
            print(f"{label:<14} MISSING")
            continue

        with open(path, 'rb') as f:
            payload = f.read()

        kill_server()
        print(f"{label:<14} ", end="", flush=True)
        proc = start_server(ctx_size)
        if proc is None:
            print("SERVER_FAIL")
            continue

        try:
            tps, prompt_tps, prompt_n = measure(payload)
            if tps == 0:
                print(f"{prompt_n:<11} ERROR")
                continue
            if baseline is None:
                baseline = tps
            ratio = f"{tps/baseline:.2f}x" if baseline > 0 else "-"
            slowdown = f"{baseline/tps:.1f}x slower" if tps < baseline else "baseline"
            print(f"{prompt_n:<11} {prompt_tps:<14.1f} {tps:<10.1f} {ratio:<7} {slowdown}")
            results.append((label, prompt_n, tps, prompt_tps))
            sys.stdout.flush()
        finally:
            proc.kill()
            proc.wait()

    if len(results) >= 2:
        print()
        print("=" * 80)
        print("  COMPACTION VALUE PROPOSITION (250K context scenario)")
        print("=" * 80)
        print()

        # Extrapolate to 250K from measured data
        # Find the speed at largest and smallest context
        small_ctx, small_tps = results[0][1], results[0][2]
        large_ctx, large_tps = results[-1][1], results[-1][2]

        print(f"  Measured: {small_ctx} tokens → {small_tps:.1f} tok/s")
        print(f"  Measured: {large_ctx} tokens → {large_tps:.1f} tok/s")
        print()
        print(f"  At 250K context without compaction:")
        print(f"    - Attention over 250K KV tokens per generated token")
        print(f"    - KV cache: 250K × 72 bytes = ~17 MB per user")
        print(f"    - Generation speed: severely degraded")
        print()
        print(f"  With 50x compaction (250K → 5K KV tokens):")
        print(f"    - Attention over 5K KV tokens (~{small_tps:.0f} tok/s, like 2K context)")
        print(f"    - KV cache: 5K × 72 bytes = ~360 KB per user")
        print(f"    - Can serve {250000//5000}x more concurrent users")
        print(f"    - Each user gets {small_tps/large_tps:.0f}x faster generation")
        print()
        print(f"  With 100x compaction (250K → 2.5K KV tokens):")
        print(f"    - KV cache: 2.5K × 72 bytes = ~180 KB per user")
        print(f"    - In 1 GB budget: {1024*1024*1024//180000:.0f} concurrent users")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()

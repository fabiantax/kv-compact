"""Aggressive throughput benchmark: push Strix Halo to the limit.

Tests realistic coding context (10K tokens = several files) with compaction.
Maximizes concurrent users within a large memory budget.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_BIN = r"C:\Users\fabia\Projects\llama.cpp\llama-flash-attn\build-win\bin\llama-server.exe"
MODEL = r"C:\Users\fabia\models\SmolLM3-3B-128K-Q4_K_M.gguf"
HOST = "127.0.0.1"
PORT = 9095
N_PREDICT = 50
TMP = os.environ.get('TEMP', '/tmp')


def kill_server():
    os.system('taskkill /F /IM llama-server.exe >nul 2>&1')
    time.sleep(2)


def start_server(n_slots, ctx_per_slot):
    total_ctx = n_slots * ctx_per_slot
    proc = subprocess.Popen(
        [SERVER_BIN, "-m", MODEL,
         "-c", str(total_ctx), "-np", str(n_slots),
         "-cb", "-ngl", "99", "-t", "4", "--no-warmup",
         "--host", HOST, "--port", str(PORT)],
        stdout=open(os.path.join(TMP, "bench_aggressive.log"), "w"),
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
    # Warmup
    send_completion(payload_bytes)
    # Measure
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(send_completion, payload_bytes) for _ in range(n_concurrent)]
        results = [f.result() for f in as_completed(futures)]
    elapsed = time.perf_counter() - t0

    total_tokens = sum(r.get('timings', {}).get('predicted_n', 0) for r in results)
    errors = sum(1 for r in results if r.get('timings', {}).get('predicted_n', 0) == 0)
    slot_tps = [r.get('timings', {}).get('predicted_per_second', 0)
                for r in results if r.get('timings', {}).get('predicted_n', 0) > 0]
    agg = total_tokens / elapsed if elapsed > 0 else 0
    avg_slot = sum(slot_tps) / len(slot_tps) if slot_tps else 0
    return agg, avg_slot, errors, elapsed


def main():
    prompt_size = sys.argv[1] if len(sys.argv) > 1 else '10k'
    full_tokens = {'1k': 881, '10k': 10500}[prompt_size]

    # Aggressive memory budgets — use the machine's 128GB RAM
    if prompt_size == '10k':
        # 10K prompt = realistic coding context (several files)
        configs = [
            # (label, keep%, n_slots, ctx_per_slot)
            ("1x full",     100,   4,  12000),   #  4 ×  12K =  48K — baseline
            ("1x full",     100,   8,  12000),   #  8 ×  12K =  96K
            ("1x full",     100,  16,  12000),   # 16 ×  12K = 192K
            ("2.5x",         40,  16,   4800),   # 16 × 4.8K =  77K
            ("2.5x",         40,  32,   4800),   # 32 × 4.8K = 154K
            ("5x",           20,  32,   2400),   # 32 × 2.4K =  77K
            ("5x",           20,  64,   2400),   # 64 × 2.4K = 154K
            ("10x",          10,  64,   1200),   # 64 × 1.2K =  77K
            ("10x",          10, 128,   1200),   #128 × 1.2K = 154K
            ("20x",           5, 128,    600),   #128 × 600  =  77K
            ("20x",           5, 256,    600),   #256 × 600  = 154K
        ]
    else:
        configs = [
            ("1x full",     100,   8,  2048),
            ("1x full",     100,  16,  2048),
            ("1x full",     100,  32,  2048),
            ("5x",           20,  32,   512),
            ("5x",           20,  64,   512),
            ("5x",           20, 128,   512),
            ("10x",          10, 128,   256),
            ("10x",          10, 256,   256),
        ]

    print(f"=== AGGRESSIVE Scaling: {prompt_size} prompt ({full_tokens} tok) ===")
    print(f"=== AMD Ryzen AI Max+ 395, Radeon 8060S, 128GB RAM, Vulkan ===")
    print()
    print(f"{'Label':<10} {'Slots':<6} {'KV/slot':<8} {'TotalCtx':<9} {'Agg tok/s':<10} {'Per-slot':<9} {'vs base':<8} {'Err':<4}")
    print(f"{'-----':<10} {'-----':<6} {'-------':<8} {'--------':<9} {'---------':<10} {'--------':<9} {'-------':<8} {'---':<4}")

    baseline_agg = None
    for label, keep_pct, n_slots, ctx_per_slot in configs:
        kv_tokens = int(full_tokens * keep_pct / 100)
        req_file = os.path.join(TMP, f"req_{prompt_size}_keep{keep_pct}.json")
        if not os.path.exists(req_file):
            print(f"{label:<10} {n_slots:<6} {kv_tokens:<8} MISSING REQUEST FILE")
            continue

        with open(req_file, 'rb') as f:
            payload = f.read()

        kill_server()
        total_ctx = n_slots * ctx_per_slot
        print(f"{label:<10} {n_slots:<6} {kv_tokens:<8} {total_ctx:<9} ", end="", flush=True)
        proc = start_server(n_slots, ctx_per_slot)
        if proc is None:
            print("SERVER_FAIL")
            continue

        try:
            agg, avg_slot, errors, elapsed = bench(payload, n_slots)
            if baseline_agg is None:
                baseline_agg = agg
            ratio = f"{agg/baseline_agg:.1f}x" if baseline_agg and baseline_agg > 0 else "-"
            err_str = str(errors) if errors > 0 else ""
            print(f"{agg:<10.1f} {avg_slot:<9.1f} {ratio:<8} {err_str}")
            sys.stdout.flush()
        finally:
            proc.kill()
            proc.wait()

    print(f"\n=== Done ===")


if __name__ == '__main__':
    main()

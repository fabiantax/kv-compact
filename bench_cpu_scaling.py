"""CPU scaling benchmark: fixed memory budget, vary slots via compaction."""
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


def start_server(n_slots, ctx_per_slot):
    total_ctx = n_slots * ctx_per_slot
    proc = subprocess.Popen(
        [SERVER_BIN, "-m", MODEL,
         "-c", str(total_ctx), "-np", str(n_slots),
         "-cb", "-ngl", "0", "--no-warmup",
         "--host", HOST, "--port", str(PORT)],
        stdout=open(os.path.join(TMP, "bench_cpu_scale.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):
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
    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(send_completion, payload_bytes) for _ in range(n_concurrent)]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.perf_counter() - t0

    total_tokens = sum(r.get('timings', {}).get('predicted_n', 0) for r in results)
    errors = sum(1 for r in results if r.get('timings', {}).get('predicted_n', 0) == 0)
    slot_tps = [r.get('timings', {}).get('predicted_per_second', 0)
                for r in results if r.get('timings', {}).get('predicted_n', 0) > 0]
    agg = total_tokens / elapsed if elapsed > 0 else 0
    avg_slot = sum(slot_tps) / len(slot_tps) if slot_tps else 0
    return agg, avg_slot, errors


def kill_server():
    os.system('taskkill /F /IM llama-server.exe >nul 2>&1')
    time.sleep(2)


def main():
    prompt_size = sys.argv[1] if len(sys.argv) > 1 else '1k'
    full_tokens = {'1k': 881, '10k': 10500}[prompt_size]

    if prompt_size == '1k':
        budget = 32768
        configs = [
            ("1x",   100, 8,  4096),   # 8 × 4K = 32K
            ("2.5x",  40, 20, 1600),   # 20 × 1.6K = 32K
            ("5x",    20, 40,  800),   # 40 × 800 = 32K
            ("10x",   10, 48,  680),   # 48 × 680 ≈ 32K
            ("20x",    5, 48,  680),   # same slots, smaller KV
            ("50x",    2, 48,  680),   # same slots, minimal KV
        ]
    else:
        budget = 24576
        configs = [
            ("1x",   100, 2,  12288),
            ("2.5x",  40, 5,  4800),
            ("5x",    20, 10, 2400),
            ("10x",   10, 20, 1200),
            ("20x",    5, 40,  600),
        ]

    print(f"=== CPU Scaling: {prompt_size} prompt ({full_tokens} tok), budget={budget} ===")
    print(f"{'Ratio':<7} {'Slots':<6} {'KV tok':<7} {'Agg tok/s':<10} {'Per-slot':<10} {'vs 1x':<7} {'Err':<4}")
    print(f"{'-----':<7} {'-----':<6} {'------':<7} {'---------':<10} {'--------':<10} {'-----':<7} {'---':<4}")

    baseline_agg = None
    for label, keep_pct, n_slots, ctx_per_slot in configs:
        kv_tokens = int(full_tokens * keep_pct / 100)
        req_file = os.path.join(TMP, f"req_{prompt_size}_keep{keep_pct}.json")
        if not os.path.exists(req_file):
            print(f"{label:<7} {n_slots:<6} {kv_tokens:<7} NO_FILE")
            continue

        with open(req_file, 'rb') as f:
            payload = f.read()

        kill_server()
        print(f"{label:<7} {n_slots:<6} {kv_tokens:<7} ", end="", flush=True)
        proc = start_server(n_slots, ctx_per_slot)
        if proc is None:
            print("SERVER_FAIL")
            continue

        try:
            agg, avg_slot, errors = bench(payload, n_slots)
            if baseline_agg is None:
                baseline_agg = agg
            ratio = f"{agg/baseline_agg:.1f}x" if baseline_agg > 0 else "-"
            err_str = str(errors) if errors > 0 else ""
            print(f"{agg:<10.1f} {avg_slot:<10.1f} {ratio:<7} {err_str}")
            sys.stdout.flush()
        finally:
            proc.kill()
            proc.wait()

    print("\n=== Done ===")


if __name__ == '__main__':
    main()

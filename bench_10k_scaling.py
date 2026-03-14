"""10K prompt scaling benchmark: fixed memory budget, vary slots via compaction."""
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
TOTAL_BUDGET = 24576  # fixed memory budget in tokens

# Configs: (label, keep_pct, n_slots, ctx_per_slot)
# Budget = n_slots * ctx_per_slot ≈ TOTAL_BUDGET
CONFIGS = [
    ("1x",   100, 2,  12288),  # 2 × 12288 = 24576
    ("2.5x",  40, 5,  4800),   # 5 × 4800  = 24000
    ("5x",    20, 10, 2400),   # 10 × 2400 = 24000
    ("10x",   10, 20, 1200),   # 20 × 1200 = 24000
    ("20x",    5, 40,  600),   # 40 × 600  = 24000
    ("50x",    2, 80,  300),   # 80 × 300  = 24000
]


def start_server(n_slots, ctx_per_slot):
    total_ctx = n_slots * ctx_per_slot
    proc = subprocess.Popen(
        [SERVER_BIN, "-m", MODEL,
         "-c", str(total_ctx), "-np", str(n_slots),
         "-cb", "-ngl", "99", "--no-warmup",
         "--host", HOST, "--port", str(PORT)],
        stdout=open(os.path.join(TMP, "bench10k_scale.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    # Wait for ready
    for _ in range(60):
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
    return {
        'total_tokens': total_tokens,
        'elapsed_s': elapsed,
        'agg_tps': agg_tps,
        'errors': errors,
        'per_slot_tps': per_slot_tps,
    }


def main():
    print("=== 10K Prompt Scaling Benchmark (Fixed Memory Budget) ===")
    print(f"Model: SmolLM3 3B Q4_K_M (Vulkan)")
    print(f"Budget: {TOTAL_BUDGET} total ctx tokens")
    print(f"Gen: {N_PREDICT} tokens per request")
    print()
    print(f"{'Ratio':<7} {'Slots':<6} {'Ctx/s':<7} {'KV tok':<7} {'Agg tok/s':<10} {'Per-slot':<10} {'Err':<4}")
    print(f"{'-----':<7} {'-----':<6} {'-----':<7} {'------':<7} {'---------':<10} {'--------':<10} {'---':<4}")

    for label, keep_pct, n_slots, ctx_per_slot in CONFIGS:
        kv_tokens = int(10500 * keep_pct / 100)
        req_file = os.path.join(TMP, f"req_10k_keep{keep_pct}.json")

        if not os.path.exists(req_file):
            print(f"{label:<7} {n_slots:<6} {ctx_per_slot:<7} {kv_tokens:<7} NO_FILE")
            continue

        with open(req_file, 'rb') as f:
            payload = f.read()

        # Start server with this config
        print(f"{label:<7} {n_slots:<6} {ctx_per_slot:<7} {kv_tokens:<7} ", end="", flush=True)
        proc = start_server(n_slots, ctx_per_slot)
        if proc is None:
            print("SERVER_FAIL")
            continue

        try:
            # Warmup
            send_completion(payload)

            # Benchmark
            r = bench_concurrent(payload, n_slots)
            avg_slot = sum(r['per_slot_tps']) / len(r['per_slot_tps']) if r['per_slot_tps'] else 0
            err_str = str(r['errors']) if r['errors'] > 0 else ""
            print(f"{r['agg_tps']:<10.1f} {avg_slot:<10.1f} {err_str}")
            sys.stdout.flush()
        finally:
            proc.kill()
            proc.wait()
            time.sleep(2)  # let port clear

    print()
    print("=== Done ===")


if __name__ == '__main__':
    main()

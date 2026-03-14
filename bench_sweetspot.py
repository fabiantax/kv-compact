"""Find the sweet spot: max users with still-usable per-slot speed.

Tests 10K code context compacted 10x (1050 KV tokens/slot).
Sweeps: 1, 2, 4, 8, 10, 16, 24, 32, 48, 64 slots.
Reports per-slot tok/s alongside aggregate.
"""
import json, os, subprocess, sys, time, io, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

STOCK = r"C:\Users\fabia\AppData\Local\Temp\llama-stock\llama-server.exe"
MODEL = r"C:\Users\fabia\models\SmolLM3-3B-128K-Q4_K_M.gguf"
HOST, PORT = "127.0.0.1", 9095
TMP = os.environ.get('TEMP', '/tmp')

def kill():
    os.system('taskkill /F /IM llama-server.exe >nul 2>&1')
    time.sleep(3)

def start(n_slots, ctx_per_slot):
    total = n_slots * ctx_per_slot
    proc = subprocess.Popen(
        [STOCK, "-m", MODEL, "-c", str(total), "-np", str(n_slots),
         "-cb", "-ngl", "99", "-t", "4", "--no-warmup",
         "--host", HOST, "--port", str(PORT)],
        stdout=open(os.path.join(TMP, "bench_sweet.log"), "w"),
        stderr=subprocess.STDOUT)
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

def send(payload):
    req = urllib.request.Request(f"http://{HOST}:{PORT}/completion",
        data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def bench(payload, n):
    send(payload)  # warmup
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(send, payload) for _ in range(n)]
        results = [f.result() for f in as_completed(futures)]
    elapsed = time.perf_counter() - t0
    total_tok = sum(r.get('timings', {}).get('predicted_n', 0) for r in results)
    errors = sum(1 for r in results if r.get('timings', {}).get('predicted_n', 0) == 0)
    slot_tps = [r.get('timings', {}).get('predicted_per_second', 0)
                for r in results if r.get('timings', {}).get('predicted_n', 0) > 0]
    agg = total_tok / elapsed if elapsed > 0 else 0
    avg = sum(slot_tps) / len(slot_tps) if slot_tps else 0
    mn = min(slot_tps) if slot_tps else 0
    return agg, avg, mn, errors

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 72)
    print("  SWEET SPOT: Per-user speed vs concurrency")
    print("  SmolLM3-3B | 10K code context | 10x compaction (1050 KV/slot)")
    print("  Stock llama.cpp b8334 | Vulkan | Radeon 8060S")
    print("=" * 72)
    print()

    # 10x compacted 10K prompt
    path = os.path.join(TMP, "req_10k_keep10.json")
    if not os.path.exists(path):
        print("MISSING req_10k_keep10.json"); return
    with open(path, 'rb') as f:
        payload = f.read()

    print(f"{'Slots':<7} {'Agg tok/s':<11} {'Per-slot avg':<13} {'Per-slot min':<13} {'Usable?':<10}")
    print(f"{'─' * 7} {'─' * 11} {'─' * 13} {'─' * 13} {'─' * 10}")

    for n_slots in [1, 2, 4, 8, 10, 16, 24, 32, 48, 64]:
        ctx_per_slot = 1200  # 1050 KV + generation headroom
        kill()
        proc = start(n_slots, ctx_per_slot)
        if proc is None:
            print(f"{n_slots:<7} SERVER_FAIL")
            continue
        try:
            agg, avg, mn, errors = bench(payload, n_slots)
            err = f" ({errors}err)" if errors > 0 else ""
            if avg >= 30:
                usable = "GREAT"
            elif avg >= 15:
                usable = "OK"
            elif avg >= 8:
                usable = "slow"
            else:
                usable = "unusable"
            print(f"{n_slots:<7} {agg:<11.1f} {avg:<13.1f} {mn:<13.1f} {usable}{err}")
            sys.stdout.flush()
        finally:
            proc.kill()
            proc.wait()

    # Also test with 5x compaction (2100 KV) at sweet spot
    print()
    print("─── 5x compaction (2100 KV/slot) for comparison ───")
    print()
    path5 = os.path.join(TMP, "req_10k_keep20.json")
    if not os.path.exists(path5):
        print("MISSING req_10k_keep20.json"); return
    with open(path5, 'rb') as f:
        payload5 = f.read()

    print(f"{'Slots':<7} {'Agg tok/s':<11} {'Per-slot avg':<13} {'Per-slot min':<13} {'Usable?':<10}")
    print(f"{'─' * 7} {'─' * 11} {'─' * 13} {'─' * 13} {'─' * 10}")

    for n_slots in [1, 2, 4, 8, 10, 16]:
        ctx_per_slot = 2300
        kill()
        proc = start(n_slots, ctx_per_slot)
        if proc is None:
            print(f"{n_slots:<7} SERVER_FAIL")
            continue
        try:
            agg, avg, mn, errors = bench(payload5, n_slots)
            err = f" ({errors}err)" if errors > 0 else ""
            if avg >= 30:
                usable = "GREAT"
            elif avg >= 15:
                usable = "OK"
            elif avg >= 8:
                usable = "slow"
            else:
                usable = "unusable"
            print(f"{n_slots:<7} {agg:<11.1f} {avg:<13.1f} {mn:<13.1f} {usable}{err}")
            sys.stdout.flush()
        finally:
            proc.kill()
            proc.wait()

    print()
    print("=" * 72)
    print("  The sweet spot: max slots where per-user speed stays > 15 tok/s")
    print("=" * 72)

if __name__ == '__main__':
    main()

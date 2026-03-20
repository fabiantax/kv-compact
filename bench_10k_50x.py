"""Quick 50x-only test for 10K prompts."""
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
TMP = os.environ.get('TEMP', '/tmp')

N_SLOTS = 80
CTX_PER_SLOT = 300
TOTAL_CTX = N_SLOTS * CTX_PER_SLOT

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

def main():
    req_file = os.path.join(TMP, "req_10k_keep2.json")
    with open(req_file, 'rb') as f:
        payload = f.read()

    # Check prompt fits
    prompt_data = json.loads(payload)
    prompt_len = len(prompt_data.get('prompt', ''))
    print(f"Prompt chars: {prompt_len}, est tokens: ~{prompt_len//4}")
    print(f"Server: {N_SLOTS} slots × {CTX_PER_SLOT} ctx = {TOTAL_CTX} total")

    # Start server
    proc = subprocess.Popen(
        [SERVER_BIN, "-m", MODEL,
         "-c", str(TOTAL_CTX), "-np", str(N_SLOTS),
         "-cb", "-ngl", "99", "--no-warmup",
         "--host", HOST, "--port", str(PORT)],
        stdout=open(os.path.join(TMP, "bench50x.log"), "w"),
        stderr=subprocess.STDOUT,
    )

    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2) as r:
                if json.loads(r.read()).get('status') == 'ok':
                    break
        except:
            pass
        time.sleep(1)

    try:
        # Warmup
        send_completion(payload)

        # Bench
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=N_SLOTS) as pool:
            futures = [pool.submit(send_completion, payload) for _ in range(N_SLOTS)]
            results = [f.result() for f in as_completed(futures)]
        elapsed = time.perf_counter() - t0

        total_tokens = 0
        errors = 0
        slot_tps = []
        for r in results:
            t = r.get('timings', {})
            n = t.get('predicted_n', 0)
            if n > 0:
                total_tokens += n
                slot_tps.append(t.get('predicted_per_second', 0))
            else:
                errors += 1

        agg = total_tokens / elapsed if elapsed > 0 else 0
        avg_slot = sum(slot_tps) / len(slot_tps) if slot_tps else 0
        print(f"50x: {N_SLOTS} slots, agg={agg:.1f} tok/s, per-slot={avg_slot:.1f}, errors={errors}")

    finally:
        proc.kill()
        proc.wait()

if __name__ == '__main__':
    main()

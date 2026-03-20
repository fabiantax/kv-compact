# kv-compact

## Todo List (Mandatory)

This project REQUIRES maintaining a task list via `TaskCreate`/`TaskList`/`TaskUpdate`. Always:
1. Create tasks for multi-step work before starting
2. Mark tasks `in_progress` when beginning work
3. Mark tasks `completed` when done
4. Clean up stale tasks at session end

**Current Session Tasks:**
- See `TaskList` output for active tasks
- Tasks persist across conversation compression

**Previous Session Status:**
- Session compaction auto-saves context
- Check MEMORY.md for project memory

---

## Session Handover

See `HANDOVER.md` for full project state, what's done/not done, and recommended
next steps. See `docs/user-stories.md` for the complete backlog (US-1 through US-18).

### Quick Start
```bash
# Test-only build (no model needed)
cmake -B build -DKV_COMPACT_BUILD_TOOL=OFF && cmake --build build
./build/test-kv-compact-math && ./build/test-kv-compact-adapter
```

### Key Files
- `include/kv-compact-math.h` — Core algorithm (header-only, zero deps)
- `include/kv-compact-adapter.h` — GQA/MLA/hybrid adapter abstraction
- `include/kv-compact-state.h` — llama.cpp state parser/writer
- `plan.md` — 5-phase streaming compaction roadmap
- `docs/improvement-tracker.md` — Implementation status matrix

---

## Qwen 3.5

Qwen 3.5 (released 2026-02-16) is Alibaba's latest model family. It does NOT use GQA or MLA — it uses a **Gated DeltaNet + full attention hybrid** architecture.

### Architecture
- **3 out of 4 layers**: Gated DeltaNet (linear attention, recurrent-style hidden state — not a traditional KV cache)
- **Every 4th layer**: Standard full attention (has a normal KV cache)
- Combines Mamba2's gated decay mechanism with a delta rule for updating hidden states
- Sparse Mixture-of-Experts (MoE) variants available

### Models
- **Dense**: Qwen3.5-0.8B, 2B, 4B, 9B, 27B
- **MoE**: Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B
- 256K context, 201 languages, thinking + non-thinking modes

### Unsloth support
- Unsloth provided day-zero GGUF quants for all variants
- Unsloth Dynamic 2.0 quants are SOTA on nearly all bit levels
- QLoRA (4-bit) training is NOT recommended for Qwen 3.5 (higher quantization error)
- Training uses custom Mamba Triton kernels (slower compile times, especially on T4)

### Implications for kv-compact
- Full-attention layers (every 4th) have standard KV cache — existing GQA adapter can work
- DeltaNet layers store gated recurrent state, not KV pairs — already "compressed" by nature
- The `LayerClassifier` should handle per-layer adapter dispatch (different adapter per layer type)
- Need to investigate how llama.cpp stores Qwen 3.5 state for DeltaNet vs attention layers



## fab-swarm Nano-Agent Integration

This project uses fab-swarm nano-agents for fast operations. **Prefer these tools over built-in equivalents:**

| Instead of | Use | Why |
|------------|-----|-----|
| `Grep` for simple searches | `nano_search` | 100x faster, no token overhead |
| `Glob` for file finding | `nano_glob` | Direct filesystem, instant |
| `Read` for file contents | `nano_read` | No context window usage |
| Manual `cargo test` | `nano_run` | Structured output |

### When to use nano-agents:
- Simple pattern searches → `nano_search`
- Finding files by pattern → `nano_glob`
- Reading file contents → `nano_read`
- Counting lines/files → `nano_count`
- Running tests/build → `nano_run`
- Simple find/replace → `nano_replace`

### When to use built-in tools:
- Complex regex with context → `Grep`
- Need to reason about contents → `Read`
- Multi-step file operations → Built-in tools

---

## llama-cli Process Management

### Problem: Stuck llama-cli.exe Processes

Multiple llama-cli.exe processes can become stuck, consuming memory and GPU resources. This happens when:
- Benchmarks are interrupted (Ctrl+C not properly handled)
- GPU backend fails to release shared memory
- Multiple instances run simultaneously
- Process exits but orphaned child processes remain

### Prevention

**Before running benchmarks:**
```bash
# Check for existing processes
ps aux | grep -i llama | grep -v grep

# If any exist, kill them first
taskkill.exe /IM llama-cli.exe /F
```

**Use safe benchmark parameters:**
```bash
# Use shorter token counts for testing
-n 32    # Instead of -n 512 for quick tests
-c 2048  # Reasonable context size

# Add timeout flag (if supported)
--timeout 300
```

**Run one benchmark at a time:**
- Wait for completion before starting next
- Check process status between runs

### Recovery: Killing Stuck Processes

**Method 1: Task Manager (GUI)**
```
1. Ctrl+Shift+Esc → Open Task Manager
2. Go to "Details" tab
3. Find llama-cli.exe
4. Right-click → End Task
```

**Method 2: PowerShell (Admin)**
```powershell
# Kill all llama processes
Get-Process | Where-Object {$_.ProcessName -like '*llama*'} | Stop-Process -Force

# Or by specific PID
Stop-Process -Id 1084 -Force
```

**Method 3: Bash (Git Bash/WSL)**
```bash
# Try normal kill first
pkill -9 llama-cli

# If that fails, use Windows command
taskkill.exe /IM llama-cli.exe /F

# Kill by process group (for orphaned processes)
ps aux | grep -i llama | awk '{print $2}' | xargs kill -9
```

**Method 4: Force Kill Process Group**
```bash
# Find the process group ID
ps aux | grep -i llama

# Kill the entire process group
kill -9 -<PGID>  # e.g., kill -9 -1047
```

### Verification

```bash
# Confirm all processes are gone
ps aux | grep -i llama | grep -v grep | wc -l

# Should output: 0
```

### If All Else Fails

**Restart GPU driver (Windows):**
```powershell
# As Administrator
Restart-Computer -Force  # Last resort
```

### Helper Scripts

Use the provided wrapper scripts to automate process management:

**Windows Batch:**
```cmd
scripts\safe-llama-run.bat -m model.gguf -c 2048 -n 32 -p "test"
```

**Git Bash:**
```bash
./scripts/safe-llama-run.sh -m model.gguf -c 2048 -n 32 -p "test"
```

These scripts automatically:
1. Check for existing llama-cli processes
2. Kill any stuck processes before running
3. Verify clean exit after completion

**Or use GPU cleanup tools:**
- AMD: Reset GPU from AMD Software
- NVIDIA: `nvidia-smi --gpu-reset` (if available)

### Workflow Integration

**Before ANY llama-cli command:**
```bash
# 1. Check existing processes
if ps aux | grep -i llama | grep -v grep > /dev/null; then
    echo "Warning: Existing llama-cli processes found!"
    echo "Kill them first: taskkill.exe /IM llama-cli.exe /F"
    # Optionally auto-kill:
    taskkill.exe /IM llama-cli.exe /F
fi

# 2. Run benchmark
./llama-cli.exe ...

# 3. Verify clean exit
ps aux | grep -i llama | grep -v grep
```

---

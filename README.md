# perfAnalysis — P1: LLM Workload Characterization via Trace-Driven Cache Simulation

**Goal:** Capture real memory traces from `llama2.c`, replay them through a cache
simulator you wrote, validate against hardware `perf` counters, and answer:
*why is LLM decode memory-bound, where exactly does it break, and what cache /
prefetcher design would help?*

Platform assumed: **native Ubuntu**, x86-64, GCC 11+, Python 3.10+.

---

## Layout

```
src/          C++17 trace-driven cache simulator
workloads/    micro/ (triad, ptr_chase, gemm_naive) + llama2c/ integration
pintool/      Intel Pin tool for trace capture
configs/      machine description + sweep specs
scripts/      env setup, hw detection, synthetic traces, sweeps, plots
traces/       (gitignored) captured traces
results/      (gitignored) CSVs and figures
REPORT.md     the writeup
```

---

## Trace format

Packed, 18 bytes per record:

```c
struct TraceEntry {
  uint64_t pc;    // instruction pointer (needed by the stride prefetcher)
  uint64_t addr;  // effective address, OR phase payload when type==2
  uint8_t  type;  // 0 = READ, 1 = WRITE, 2 = PHASE
  uint8_t  size;  // access size in bytes
};
```

Phase records encode `0xDEAD000000000000 | (phase_id << 32) | aux`, where `aux`
is the token position (for PREFILL/DECODE) or the layer index (for ATTN/FFN).

Budget: ~18 bytes/access. 500M accesses = ~9 GB raw, ~2 GB after `zstd -3`.
Pin slows execution **50-100x** — size your `-n` accordingly.

---

## Phases and gates

**Do not skip a gate.** Each one prevents a class of bug that is brutal to find later.

### Phase A — Environment & ground truth
```bash
sudo ./scripts/setup_env.sh          # perf perms, ASLR off, freq pinned
./scripts/detect_hw.sh               # writes configs/my_machine.yaml
# fill in the latency TODOs using lmbench lat_mem_rd or Intel MLC
./workloads/llama2c/fetch_llama2c.sh
./scripts/perf_baseline.sh           # writes results/hw_baseline.csv
```
**Gate:** you have measured cache latencies, STREAM bandwidth, and a `perf` MPKI baseline.

### Phase B — Simulator first (no Pin yet)
```bash
make
python3 scripts/gen_synthetic_trace.py
python3 scripts/verify_sim.py        # must exit 0
```
**Gate:** the simulator reproduces all analytically-known synthetic patterns exactly.

### Phase C — Trace capture
```bash
# see pintool/README.md for Pin install
cd pintool && make obj-intel64/llmtrace.so TARGET=intel64 && cd ..
$PIN_ROOT/pin -t pintool/obj-intel64/llmtrace.so -o traces/llama110m.bin \
    -- workloads/llama2c/llama2.c/run stories110M.bin -n 30 -i "Once upon a time"
python3 scripts/verify_tracer.py
```
**Gate:** pintool load/store counts match `perf` within ~5%.

### Phase D — Microbenchmark validation
Validate `triad` (line size), `ptr_chase` (capacity/replacement), `gemm_naive`
(set indexing/associativity) against `perf`.
**Gate:** MPKI within 15% of hardware. Each benchmark localizes a different bug class.

### Phase E — The study
```bash
python3 scripts/sweep.py configs/sweep_size.yaml
python3 scripts/plot.py
python3 scripts/roofline.py
```
E1 prefill vs decode - E2 KV-cache growth - E3 roofline - E4 model scaling -
E5 prefetchers - E6 tiled-matmul co-design - E7 AMAT.

### Phase F — Report
Fill in `REPORT.md`.

---

## The prediction to validate (do this on paper first)

For `stories110M`: 12 layers, dim 768, seq_len 1024, fp32.

```
KV bytes per token position = 2 * layers * dim * 4
                            = 2 * 12 * 768 * 4  =  73,728 B  ~= 73.7 KB
```

For a 12 MB LLC the KV cache stops fitting at:

```
pos = 12,582,912 / 73,728  ~=  170 tokens
```

Weights are ~440 MB — roughly 36x the LLC, streamed once per decoded token with
zero reuse. **Predict first, then measure in E2.** Matching a hand-derived
prediction is the strongest result in the report.

---

## Simulator CLI

| Flag | Meaning |
|---|---|
| `--trace <path>` / `--stdin` | input trace |
| `--config <yaml>` | machine description |
| `--phase <id>` | analyse only this phase |
| `--pos-min/--pos-max <n>` | restrict to a token-position window |
| `--warmup <n>` | replay n accesses, then reset stats |
| `--prefetcher none\|nextline\|stride` | prefetcher model |
| `--degree <n>` | prefetch degree |
| `--json <path>` | machine-readable stats |
| `--amat` | compute AMAT from config latencies |

---

## Troubleshooting

- `perf` permission denied - rerun `scripts/setup_env.sh`; check `perf_event_paranoid`.
- Pin fails to attach - `sudo sysctl -w kernel.yama.ptrace_scope=0`.
- Trace counts low vs `perf` - you are missing multi-operand instructions (AVX,
  `rep movs`). Loop over `INS_MemoryOperandCount`, not just operand 0.
- Non-reproducible set indices between runs - ASLR is back on.
- Markers optimised away - check `volatile`, verify with
  `objdump -d run | grep __p1_marker`.

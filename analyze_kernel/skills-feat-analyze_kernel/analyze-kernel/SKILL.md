---
name: analyze-kernel
description: |
  Analyze, benchmark, and profile any JAX/Pallas kernel on TPU.
  Produces a structured report with theoretical FLOPs, correctness vs reference,
  wall-clock timing, xplane device timing, MXU utilization, and bottleneck analysis.
  Use when profiling kernel performance, debugging hardware utilization,
  or comparing kernel implementations.
---

# Analyze JAX/Pallas Kernel

Analyze the kernel specified by: `$ARGUMENTS`

## Setup

1. Read `CLAUDE.md` and `AGENTS.md` to determine the SSH host alias and remote repo path.
2. Sync the local repo to the TPU before benchmarking. Prefer a full repo sync with cache/build excludes:
   ```
   rsync -az --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude .mypy_cache /path/to/PallasKernels/ <tpu-host>:/home/gcpuser/PallasKernels/
   ```
   Replace the local path, host, and remote path with values from repo docs when they differ.
3. Install/update the remote TPU environment with uv, then force-upgrade `libtpu` to `0.0.41`:
   ```
   ssh <tpu-host> 'cd <remote-repo> && /home/gcpuser/.local/bin/uv sync --extra tpu --extra tests && /home/gcpuser/.local/bin/uv pip install "libtpu==0.0.41"'
   ```
4. Deploy utility scripts to the TPU:
   ```
   rsync -avz ${CLAUDE_SKILL_DIR}/scripts/ <tpu-host>:<remote-repo>/tools/kernel_analyzer/
   ```

## Workflow

### Phase 1 — Investigate the kernel

Read the kernel source code. Determine:

- Function signature, input shapes and dtypes
- `pallas_call` grid spec, BlockSpec layout
- All matmuls inside the kernel body: shape `(M,K)@(K,N)`, loop iteration count, grid cell count
- Optimization techniques used (load `${CLAUDE_SKILL_DIR}/references/checklist.md` Phase 3 for the catalog)
- Suitable reference implementation for correctness comparison
- Suitable baseline(s) for speedup comparison

Compute theoretical FLOPs: `sum(2 × M × K × N × iterations × grid_cells)` for all matmuls.

**Cross-check FLOPs**: If a pure-JAX reference implementation exists, JIT-compile it and call `compiled.cost_analysis()` to get `model_flops`. XLA CAN count FLOPs for standard JAX ops — it only reports 0 for Pallas `custom-call`. Note any discrepancy.

**Present the matmul inventory and proposed baselines to the user for confirmation before proceeding.**

### Phase 2 — Test correctness

Write a Python script on the TPU that:
1. Generates test inputs matching the kernel's signature (random normal, scale weights by 0.02)
2. Runs the kernel and reference implementation
3. Computes `cos_sim` / `max_diff` using `tools/kernel_analyzer/correctness.py` on the TPU

Run via SSH. Test at small shapes first (e.g., S=1024, H=8) for fast iteration.

Thresholds: cos_sim > 0.9999 = PASS, > 0.99 = MARGINAL, < 0.99 = FAIL.

### Phase 3 — Benchmark

Write a Python script on the TPU that:
1. Runs the kernel and baselines using `tools/kernel_analyzer/benchmark_utils.bench()`
2. Reports timing stats (min, median, mean, std, p5, p95)
3. Computes speedup ratios using median times

Run via SSH. Note if wall clock is unreliable for sub-10ms kernels (host dispatch overhead ~100us).

### Phase 4 — Trace and device timing

Write a Python script on the TPU that:
1. Captures a JAX profiler trace with warmup (3 iters) + 5 traced iterations
2. Parses the xplane file using `tools/kernel_analyzer/xplane_parser.get_per_op_breakdown()`

**Always** run with these environment variables:
```
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
```

Run via SSH. Load `${CLAUDE_SKILL_DIR}/references/hardware-specs.md` for TPU constants. Compute MXU%:
```
MXU% = theoretical_FLOPs / (device_time_seconds × peak_TFLOPS × 1e12) × 100
```

### Phase 5 — Report

Assemble the report following `${CLAUDE_SKILL_DIR}/references/report-format.md`. Include all 6 sections:
1. Computation flow and matmul inventory
2. Correctness results
3. Benchmark results and speedup
4. Hardware utilization (FLOPs, device time, MXU%)
5. Bottleneck analysis with optimization suggestions
6. Trace file path and per-op breakdown

Save the report alongside the kernel source or at a location the user specifies.

## When to load references

- `references/checklist.md` — at the start of Phase 1
- `references/hardware-specs.md` — in Phase 4 when computing MXU/roofline
- `references/report-format.md` — in Phase 5 for report assembly
- Do NOT load all references upfront

## Critical rules

- **All Python runs on TPU via SSH**, never locally on the host machine
- **No hardcoded remote paths** — read CLAUDE.md/AGENTS.md to determine SSH host and remote repo path
- **XLA reports 0 FLOPs for Pallas custom-call** — count FLOPs manually by reading the kernel code; cross-check against JAX reference's `cost_analysis()` when available
- **Always use the LIBTPU_INIT_ARGS** flags for every trace capture
- **Always upgrade libtpu to 0.0.41 after remote `uv sync`** — older versions may reject `--xla_enable_custom_call_region_trace`.
- **Wall clock valid for >10ms kernels only** — for sub-10ms, note that xplane `device_duration_ps` is needed
- **xplane map fields** — `event_metadata`/`stat_metadata` are MAP fields, access via `.items()` or `[key]`, not iteration
- **Remote script writes** — when writing helper scripts on the TPU, prefer a quoted heredoc that rewrites the whole script. This avoids local shell expansion and command-parser issues with parentheses:
  ```
  ssh <tpu-host> 'bash -s' <<'SH'
  cat > <remote-repo>/tools/kernel_analyzer/<script>.py <<'PY'
  # script body
  PY
  SH
  ```

## Utility scripts

Located in `${CLAUDE_SKILL_DIR}/scripts/`, deployed to TPU at `<remote-repo>/tools/kernel_analyzer/`:

| Script | Functions |
|--------|-----------|
| `correctness.py` | `cos_sim(a, b)`, `max_diff(a, b)`, `mean_diff(a, b)`, `check_pass(cos)` |
| `benchmark_utils.py` | `bench(fn, warmup, iters)`, `compute_stats(times)`, `format_stats(stats)` |
| `xplane_parser.py` | `find_xplane(dir)`, `get_per_op_breakdown(dir)`, `get_op_time(dir, name)` |

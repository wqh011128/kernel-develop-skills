---
name: analyze-kernel
description: "Analyze correctness, benchmark performance, manual FLOPs, device timing, MFU, memory traffic, and bottlenecks for JAX/Pallas kernels on TPU. Use when profiling, comparing implementations, validating a tuning hypothesis, or producing structured reports under tmp/{kernel_name}_{date}/experiments/{method_name}/results/performance."
---

# Analyze JAX/Pallas Kernel

Use this skill for evidence-based performance analysis. Coordinate with `$profile-pallas-xprof` for profile capture, but keep manual FLOPs and source-level reasoning as the source of truth when XProf counters are incomplete or inconsistent. Write generated reports in Chinese by default; keep code identifiers, commands, metric names, and profiler event names unchanged.

## Required Output Location

Write analysis outputs under the method being analyzed:

```text
tmp/{kernel_name}_{YYYYMMDD}/experiments/{method_name}/results/performance/
```

Do not write new reports to a top-level `results/reports/` directory.

## Analysis Workflow

1. Read the kernel source, wrapper, tests, benchmark runner, and docs.
2. Confirm the operator contract, shapes, dtypes, layout, masks, padding, and reference.
3. Build a manual FLOPs and memory-traffic model when applicable.
4. Read correctness artifacts before using performance numbers.
5. Read benchmark distributions: median, mean, std, p5, p95, warmup, iterations, baseline.
6. Read XProf/device timing when available: full device time, custom-call time, collectives, copy/reshape/transpose, HBM/DMA, VMEM spill/fill, MXU, vector load/store, XLU.
7. For TPU profiles, inspect both `.xplane.pb` derived data and `*.trace.json.gz` when available. If one path is unavailable, state why.
8. Explicitly answer whether Vector ALU or Scalar ALU dominates, whether MXU utilization is low, whether communication overlaps useful compute, and whether host dispatch/device launch overhead matters.
9. Identify the bottleneck with evidence.
10. Propose optimization hypotheses with validation metrics and rejection conditions.
11. Write the report under `results/performance/`.
12. Update `docs/results.md` and, if a tuning decision changes, `docs/optimization.md`.

For XProf/Roofline/FLOPs/Bytes/Memory/Trace interpretation, load `references/diagnostic-matrix.md` and use the "phenomenon -> possible causes -> next checks" pattern. Do not jump from a slow median directly to block-size tuning.

## Required Deep XProf Questions

For TPU/Pallas work, every performance report must include a short answer for:

```text
ALU pressure:
  Vector ALU, Scalar ALU, exp/online-softmax, mask, index, and fusion overhead.

MXU utilization:
  manual FLOPs versus device time, custom-call count, tile size, padding, and whether low MXU is expected for the shape.

Memory and spills:
  Vector Load/Store, HBM/DMA, VMEM spill/fill, and copy/reshape/transpose.

Communication overlap:
  collective start/done timing, overlap with local compute, exposed communication time, and whether collectives serialize.

Host/device split:
  host dispatch, device launch, PjRt execution, and whether wall-clock timing is dominated by host overhead.

Bottleneck ranking:
  rank the top bottlenecks with evidence and name the next metric that should move.
```

## Report Contents

Write reports in Chinese by default unless the user explicitly asks for another language. Keep reports structured and scannable. Include these sections in this order:

```text
1. Conclusion summary:
   one-paragraph decision and bottleneck conclusion

2. Kernel contract:
   semantics, shapes, dtypes, layout, masks, padding, reference

3. Correctness status:
   test command, pass/fail, tolerances, artifact paths

4. Benchmark summary:
   median/mean/std/p5/p95, baseline, target, speedup, measurement policy

5. XProf/device timing:
   local XProf URL if running, profile path, custom-call, collectives, copies, fusion/control

6. FLOPs/MFU/memory model:
   manual FLOPs, MFU/utilization estimate, memory or communication estimate

7. Deep XProf answers:
   ALU pressure, MXU utilization, memory/spills, communication overlap, host/device split

8. Bottleneck analysis:
   evidence-backed bottleneck, not intuition

9. Optimization hypotheses:
   accepted, rejected, and next hypotheses with validation metrics

10. Artifact paths:
   correctness, benchmark, xprof, performance report, updated docs
```

For Pallas custom-calls, do not blindly trust XProf FLOPs. Cross-check with source-level manual FLOPs, CostEstimate, or JAX cost analysis when possible.

## User-Facing Result

Reply in Chinese and include:

```text
local XProf URL, if running
local XProf artifact path, if available
kernel analysis report path
benchmark artifact path
correctness artifact path
short bottleneck conclusion
```

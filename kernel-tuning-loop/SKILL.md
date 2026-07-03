---
name: kernel-tuning-loop
description: "Run a structured kernel tuning loop for JAX/Pallas/TPU/GPU kernels after a trusted reference and initial implementation exist. Coordinates correctness, benchmark, analyze-kernel reports, remote XProf capture, hypothesis-driven optimization, and experiment logging under tmp/{kernel_name}_{date}/experiments/{method_name}/results."
---

# Kernel Tuning Loop

Use this skill after a kernel has a trusted reference and an initial implementation. If the reference is not trusted, stop tuning and fix correctness first.

## Required Experiment Results Layout

Every tuning attempt must write artifacts under one method:

```text
tmp/{kernel_name}_{YYYYMMDD}/experiments/{method_name}/
  README.md
  results/
    benchmark/
    correctness/
    xprof/
    performance/
```

Do not mix artifacts from different implementation strategies in one folder.

Use `results/correctness/` for correctness artifacts, `results/benchmark/` for benchmark outputs, `results/xprof/` for XProf captures, and `results/performance/` for analysis reports.

## Iteration Loop

Run one focused experiment per iteration:

1. State one hypothesis and expected metric movement.
2. Check consensus sources or existing repo patterns when the hypothesis is not already proven by local evidence.
3. Classify the bottleneck as compute/MXU, HBM, VMEM, communication, launch/control, or mixed.
4. If the hypothesis depends on communication-compute overlap, ring/pipeline scheduling, async copy, DMA, remote transfer, prefetch, or expression-order tuning, use `$optimize-kernel-from-evidence`, load its overlap references, and run `scripts/overlap_feasibility.py` on C/M/S/O probes before changing the full kernel.
5. Identify the baseline artifact and target artifact paths.
6. Make the smallest code change that tests the hypothesis.
7. Run correctness first.
8. Run benchmark with the same shape, dtype, warmup, iteration count, and baseline policy.
   If a single sweep shows a large win for a configuration that should be equivalent to the current default, run a focused repeat before accepting the tuning result.
9. Capture XProf when wall time is small, communication is involved, or results are ambiguous.
10. Analyze full time and component time, not only the target custom-call.
11. Accept, reject, or keep investigating.
12. Update docs and experiment README in the same turn.

## Required Updates

When tuning changes results, update:

```text
docs/results.md
docs/optimization.md
docs/fail-notes.md, if a concise reusable pitfall was found
experiments/{method_name}/README.md
```

Use `results/performance/` for `$analyze-kernel` reports and `results/xprof/` for `$profile-pallas-xprof` artifacts.

## Decision Rules

Accept an optimization only when:

```text
correctness still passes
the target metric improves against a stable baseline
full device time does not regress unless the user explicitly accepts the tradeoff
the result is reproducible enough for the development stage
the experiment record explains why the change is kept
```

Reject an optimization when:

```text
correctness fails
benchmark movement is noise
local custom-call speedup is offset by collectives, copies, reshapes, or control overhead
complexity increases without measurable benefit
the result depends on an untrusted reference or missing edge-case coverage
```

For overlap-related work, also reject when:

```text
T_comm_est > T_compute_est and chunk/tile/communication strategy has not been adjusted
candidate_overlap_step is equivalent to serial_compute_then_comm
communication-done is hidden but full latency does not improve
no-communication multi-step compute is already slower than the baseline
state round-trip, custom-call fragmentation, copy/layout, or scratch traffic offsets the communication saving
```

## User-Facing Result

Reply in Chinese. Use a structured result with these fields:

```text
Conclusion:
  accepted/rejected/investigating and why

Current hypothesis:
  one focused hypothesis and expected metric movement

Correctness:
  pass/fail, command summary, artifact path

Benchmark:
  key median comparisons, speedup/regression, artifact path

XProf:
  local URL if running
  profile artifact path
  visible/readiness status
  if UI is not running, exact reason and recovery command/path

Analysis report:
  analyze-kernel/performance report path

Docs updated:
  docs and experiment README updated

Next step:
  next hypothesis with rejection condition
```

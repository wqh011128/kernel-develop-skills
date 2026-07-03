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
4. Identify the baseline artifact and target artifact paths.
5. Make the smallest code change that tests the hypothesis.
6. Run correctness first.
7. Run benchmark with the same shape, dtype, warmup, iteration count, and baseline policy.
   If a single sweep shows a large win for a configuration that should be equivalent to the current default, run a focused repeat before accepting the tuning result.
8. Capture XProf when wall time is small, communication is involved, or results are ambiguous.
9. Analyze full time and component time, not only the target custom-call.
10. Accept, reject, or keep investigating.
11. Update docs and experiment README in the same turn.

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

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
2. Identify the baseline artifact and target artifact paths.
3. Make the smallest code change that tests the hypothesis.
4. Run correctness first.
5. Run benchmark with the same shape, dtype, warmup, iteration count, and baseline policy.
6. Capture XProf when wall time is small, communication is involved, or results are ambiguous.
7. Analyze full time and component time, not only the target custom-call.
8. Accept, reject, or keep investigating.
9. Update docs and experiment README in the same turn.

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

Reply in Chinese and include:

```text
experiment directory
correctness artifact path
benchmark artifact path
XProf local URL, if running
XProf artifact path, if available
analyze-kernel report path, if generated
decision and next hypothesis
```

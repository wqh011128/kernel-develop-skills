---
name: implement-kernel-from-plan
description: "Implement a JAX/Pallas/TPU/GPU kernel only after an approved design and validation plan exists. Use when coding from tmp/{kernel_name}_{date}/docs or an equivalent plan. Implements correctness-first, follows repository patterns, stores each strategy under tmp/{kernel_name}_{date}/experiments/{method_name}, and updates docs with evidence."
---

# Implement Kernel From Plan

Use this skill only after the design docs or equivalent plan exists. If the math or reference is missing, return to `$kernel-design-docs` or `$kernel-goal-discovery`.

## Required Experiment Layout

Each distinct implementation strategy must live under:

```text
tmp/{kernel_name}_{YYYYMMDD}/experiments/{method_name}/
  code/
  README.md
  results/
    benchmark/
    correctness/
    xprof/
    performance/
```

Rules:

```text
method_name: descriptive strategy name, not impl1 except as a temporary discussion placeholder.
README.md: always uppercase.
code/: source snapshots, patches, runners, and helper scripts for this strategy.
results/correctness/: correctness logs and metrics.
results/benchmark/: benchmark JSON, CSV, logs.
results/xprof/: raw profiles, tarballs, local server logs, XProf-derived JSON.
results/performance/: analyze-kernel reports and derived performance summaries.
```

Do not place new experiment artifacts in a top-level `results/` directory.

## Read Before Coding

Read the relevant docs first:

```text
docs/rfc.md
docs/math.md
docs/develop-plan.md
docs/results.md, if existing baselines or validation notes exist
docs/impl-notes.md, if continuing prior implementation
docs/optimization.md, if continuing tuning
```

## Implementation Order

1. Inspect repository patterns for similar kernels, tests, benchmark helpers, layout conventions, dtype policy, registry entry points, and profiling conventions.
2. Implement or verify a trusted reference/wrapper first.
3. Implement the smallest useful target kernel or communication wrapper.
4. Run small-shape correctness against the trusted reference.
5. Expand correctness to representative and edge cases.
6. Add benchmark and optional profile runner.
7. Only then optimize.

## Correctness Gate

Do not discuss speedup or MFU until correctness passes for the intended semantics. Minimum correctness record:

```text
shape
dtype
seed
reference
tolerance
max abs diff
relative error or cosine when useful
PASS/FAIL
```

If correctness fails, record the failure under `results/correctness/` and summarize the actionable cause in `docs/fail-notes.md` only when understood.

## Objective Engineering Rules

Past kernel tricks are hypotheses, not facts. Keep changes explainable:

```text
change one main variable at a time
preserve existing single-kernel behavior unless explicitly asked
separate communication wrapper from local kernel math when practical
avoid assuming reshape, padding, or XProf FLOPs are the bottleneck without evidence
prefer repository style over new ad-hoc structure
```

## Documentation Updates

After implementation or verification changes, update:

```text
docs/impl-notes.md
docs/results.md, when correctness/benchmark/profile changes
docs/optimization.md, when an optimization decision changes
docs/fail-notes.md, when a concise pitfall is learned
experiments/{method_name}/README.md
```

Report in Chinese with changed files, correctness status, benchmark/profile artifacts if generated, and remaining gaps.

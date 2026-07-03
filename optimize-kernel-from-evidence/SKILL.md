---
name: optimize-kernel-from-evidence
description: "Evidence-driven optimization for JAX/Pallas/TPU/GPU kernels. Use after a kernel has a trusted mathematical contract, reference implementation, correctness tests, and at least one baseline benchmark or profile. Chooses one optimization hypothesis at a time and records results under tmp/{kernel_name}_{date}/experiments/{method_name}/results."
---

# Optimize Kernel From Evidence

Drive optimization from evidence, not intuition. Keep correctness first, change one main variable per iteration, compare against a stable baseline, and record accepted and rejected hypotheses.

## Required Experiment Layout

Use:

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

Each `method_name` represents one implementation strategy or tuning hypothesis. Store raw artifacts inside that experiment. Summarize cross-experiment conclusions in `docs/results.md` and tuning decisions in `docs/optimization.md`.

## Hard Gates

Do not optimize until all gates are true:

```text
operator semantics and mathematical derivation are documented
a JAX/framework reference exists and is trusted
correctness tests compare target implementation against the reference
at least one baseline benchmark or device profile exists
target shape, dtype, accumulation, hardware/backend, and tolerance are explicit
```

If the reference may be wrong, stop and fix the reference first. Never use a possibly incorrect optimized kernel to validate itself.

## Iteration Loop

1. State one hypothesis.
2. Name the expected metric movement, such as lower full device time, lower collective time, fewer spills, less HBM traffic, or lower compile overhead.
3. Identify baseline and target artifact paths.
4. Make the smallest code change that tests the hypothesis.
5. Run correctness before benchmark or profile.
6. Run benchmark with unchanged measurement policy.
7. Capture XProf when communication is involved, wall time is small, or results are ambiguous.
8. Compare full time and component time, not only the optimized custom-call.
9. Accept, reject, or keep investigating based on predefined metrics.
10. Update experiment README and top-level docs.

Do not combine unrelated changes such as layout, tile size, masking, communication pattern, and dtype policy unless they cannot be separated.

## Evidence Hierarchy

Prefer this order:

```text
mathematical contract and trusted reference
correctness tests over representative and edge shapes
device timing and benchmark distributions
XProf component timing for collectives, copies, reshapes, and custom-calls
manual FLOPs, memory traffic, and source-level reasoning
XProf FLOPs or compiler estimates after cross-checking
```

If evidence conflicts, record the conflict and add a targeted experiment.

## Skill Self-Improvement Boundary

Only update a skill when the lesson is general, objective, and likely useful for future kernels:

```text
new correctness gate
profile interpretation rule
reproducibility workflow improvement
artifact layout or benchmark field convention
```

Do not put shape-specific conclusions, kernel-specific bugs, one-off commands, cluster details, long logs, or raw trace summaries into skills.

## User-Facing Result

Reply in Chinese and include:

```text
optimization hypothesis
correctness status
benchmark/profile comparison
decision
experiment path
XProf URL/path, if available
performance report path, if available
docs updated
```

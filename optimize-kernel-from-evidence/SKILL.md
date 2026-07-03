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
3. Search or inspect consensus sources before inventing a tuning direction when time allows: official framework/backend docs, XProf/Roofline docs, current repo patterns, and high-signal open-source implementations.
4. Select the relevant kernel-type tuning reference from `references/` when the kernel category is clear.
5. Classify the current bottleneck with a roofline-style category: compute/MXU-bound, HBM-bound, VMEM/shared-memory-bound, communication-bound, launch/control-bound, or mixed.
6. Identify baseline and target artifact paths.
7. Make the smallest code change that tests the hypothesis.
8. Run correctness before benchmark or profile.
9. Run benchmark with unchanged measurement policy.
10. Capture XProf when communication is involved, wall time is small, or results are ambiguous.
11. Compare full time and component time, not only the optimized custom-call.
12. Accept, reject, or keep investigating based on predefined metrics.
13. Update experiment README and top-level docs.

Do not combine unrelated changes such as layout, tile size, masking, communication pattern, and dtype policy unless they cannot be separated.

## Kernel-Type Tuning References

Load only the reference matching the current kernel category:

```text
attention kernels, softmax, GQA/MHA/MQA, ring/prefix attention:
  references/attention-kernels.md

matmul-like kernels, GEMM, batched matmul, projection kernels:
  references/matmul-kernels.md

reduction or scan kernels, prefix sums, normalization, reductions:
  references/reduction-kernels.md

elementwise, fused activation, pointwise transform kernels:
  references/elementwise-kernels.md
```

If a kernel spans categories, load the smallest set of references that covers the bottleneck being tested. Do not load every reference by default.

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

## Roofline-Guided Optimization Classes

Use the bottleneck class to choose tactics:

```text
Compute/MXU-bound:
  improve tile shapes, increase useful matmul work per launch, reduce padding waste, improve accumulator/layout choices.

HBM-bound:
  reduce materialized intermediates, improve data reuse, avoid all-gather materialization when it dominates, reduce reads/writes and layout copies.

VMEM/shared-memory-bound:
  reduce scratch footprint, spills/fills, accumulator size, and excessive per-tile state.

Communication-bound:
  reduce collective bytes, collective count, exposed done/sync time, and improve overlap with useful local compute.

Launch/control-bound:
  reduce custom-call count, host dispatch, JAX fusion/control overhead, dynamic branches, and tiny kernels.

Mixed:
  change one bottleneck at a time and require XProf evidence that the intended component moved.
```

## Consensus Sources To Prefer

When a tuning direction is not obvious from local evidence, prefer primary or high-signal sources:

```text
OpenXLA XProf overview and roofline docs:
  https://openxla.org/xprof
  https://openxla.org/xprof/roofline_model

Google Cloud TPU performance guide:
  https://docs.cloud.google.com/tpu/docs/performance-guide

JAX Pallas documentation:
  https://docs.jax.dev/en/latest/pallas/index.html

Project-local kernels, tests, benchmark runners, and existing experiment reports.
```

Use these sources to choose the class of optimization, not to override local correctness or benchmark evidence.

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

Reply in Chinese. Use a structured result with these fields:

```text
Conclusion:
  accepted/rejected/investigating

Optimization hypothesis:
  one hypothesis, target metric, rejection condition

Evidence:
  correctness status
  benchmark/profile comparison
  XProf component comparison when available

Decision:
  why the change is kept, reverted, or still under investigation

Artifacts:
  experiment path
  correctness path
  benchmark path
  XProf local URL/path, or failure reason plus recovery command
  performance report path

Docs:
  docs updated

Next step:
  next smallest hypothesis
```

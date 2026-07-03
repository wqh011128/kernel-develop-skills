---
name: kernel-dev-lifecycle
description: "Orchestrate the full kernel development lifecycle from goal discovery to tuned implementation. Use when building, porting, optimizing, or repeatedly developing JAX/Pallas/TPU/GPU kernels. Enforces tmp/{kernel_name}_{date}/docs plus experiments/{method_name}/results layout, coordinates goal discovery, design docs, implementation, correctness, benchmark, XProf, analysis, and evidence-driven tuning."
---

# Kernel Dev Lifecycle

Use this as the top-level workflow for kernel development. It coordinates the stage skills; it does not replace them.

## Required Workspace Layout

Every kernel project must use this layout unless the user explicitly gives a different root:

```text
tmp/{kernel_name}_{YYYYMMDD}/
  docs/
    README.md
    rfc.md
    math.md
    results.md
    fail-notes.md
    impl-notes.md
    optimization.md
  experiments/
    {method_name}/
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
kernel_name: lowercase snake_case.
method_name: descriptive strategy name, such as all_gather_reference, ring_pallas, tiled_streaming.
README.md: always uppercase.
Do not create `develop-plan.md` for new work; `rfc.md` includes the development plan.
No new top-level results/ directory.
No old numbered docs such as 00-rfc.md or 03-validation-and-profiling.md.
```

Raw artifacts belong under exactly one `experiments/{method_name}/results/` directory. Top-level docs summarize evidence across experiments; they are not raw-log dumps.

## Stage Order

1. Goal discovery: use `$kernel-goal-discovery`.
2. Design docs: use `$kernel-design-docs`.
3. Implementation: use `$implement-kernel-from-plan`.
4. Correctness, benchmark, XProf, and analysis: use `$analyze-kernel` and `$profile-pallas-xprof` as needed.
5. Optimization: use `$optimize-kernel-from-evidence` and `$kernel-tuning-loop`.
6. Delivery: report implementation status, correctness, benchmark, XProf URL/path, analysis path, accepted/rejected optimizations, and next steps.

## Hard Gates

Do not proceed past these gates:

```text
No confirmed goal -> no design or implementation.
No mathematical derivation -> no kernel implementation.
No trusted JAX/framework reference -> no optimization.
Correctness failing -> no performance conclusion.
No stable baseline -> no speedup claim.
No device/profile timing for small kernels -> no MFU/utilization conclusion.
No experiment record -> do not keep an optimization as proven.
```

## Document Update Contract

When adding or changing artifacts, update related docs in the same turn:

```text
Goal or scope change:
  docs/rfc.md
  docs/README.md

Plan, ownership, rollout, validation strategy, or acceptance gate change:
  docs/rfc.md
  docs/optimization.md, only when the change is driven by experimental evidence

Math, mask, dtype, padding, or equivalence change:
  docs/math.md
  docs/impl-notes.md

Implementation change:
  docs/impl-notes.md
  experiments/{method_name}/README.md

Correctness, benchmark, XProf, or performance report:
  docs/results.md
  experiments/{method_name}/README.md

Optimization decision:
  docs/optimization.md
  docs/results.md
  docs/fail-notes.md, only for concise reusable pitfalls
```

## Documentation Quality Contract

Every substantial iteration must keep docs coherent, structured, and evidence-linked:

```text
README.md:
  update current status, current best, experiment index, and active XProf/report paths.

rfc.md:
  update only goal/scope/design/task/decision-log level changes.

math.md:
  update only when semantics, masking, dtype, padding, or equivalence changes.

results.md:
  summarize correctness, benchmark, XProf, and current-best evidence; do not paste raw logs.

optimization.md:
  record hypothesis, bottleneck class, evidence, accept/reject decision, and next hypothesis.

impl-notes.md:
  record API/file/layout/communication/kernel-boundary changes.

fail-notes.md:
  record concise root causes and what not to repeat; avoid long failure transcripts.
```

Use Chinese by default. Keep facts, evidence, decisions, and next actions in separate sections. Prefer tables for matrices and short code blocks for metrics, commands, shapes, and paths. Every performance claim must point to an experiment artifact or XProf/analysis report.

## Skill Self-Improvement Boundary

Update general skills only when a lesson is objective, reusable across kernels, and small enough to improve the process without overfitting. Put kernel-specific shapes, numbers, bugs, paths, cluster details, and failed experiment details in the current kernel workspace, not in skills.

At the end of each substantial kernel iteration, run this skill-evolution check:

```text
1. Identify lessons from docs/results.md, docs/optimization.md, docs/fail-notes.md, XProf reports, and correctness failures.
2. Classify each lesson as kernel-specific or reusable.
3. Keep kernel-specific lessons in the current kernel workspace only.
4. For reusable lessons, patch the smallest relevant skill section instead of rewriting stable content.
5. Prefer new guardrails, validation gates, artifact fields, or profiling interpretation rules over kernel-specific tuning recipes.
6. Re-run quick_validate.py for every modified skill.
7. Report which skill changed and why; if no reusable lesson exists, say no skill update was made.
```

## User-Facing Output

Reply in Chinese unless the user asks otherwise. For result-related responses, include:

```text
experiment directory
benchmark artifact path
correctness artifact path
XProf local URL, if running
XProf artifact path, if available
kernel analysis report path, if generated
docs updated
```

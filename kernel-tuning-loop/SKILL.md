---
name: kernel-tuning-loop
description: "Test one or more falsifiable JAX/Pallas/TPU/GPU kernel optimizations against a trusted reference and stable baseline. Use for a single evidence-backed change, repeated bounded tuning, communication-overlap feasibility, autonomous research, portfolio exploration, or kernel-genome evaluation. Applies correctness and full-latency gates and uses kernel-foundry state when experiments repeat."
---

# Kernel Tuning Loop

## Choose one mode

| Mode | Use when | State |
| --- | --- | --- |
| `single` | One observed bottleneck and one attributable optimization | One experiment record; no research queue |
| `research` | Two or more hypotheses, genome/portfolio work, or autonomous iteration | Bounded `$kernel-foundry` research state |

## Entry gates

Require a confirmed operator contract, independent trusted reference, representative correctness pass, stable baseline, explicit objectives, and comparable measurement policy. Stop and repair the missing gate instead of creating a tuning queue.

## Run one attributable experiment

1. State the observed phenomenon, plausible causes, selected cause, expected metric movement, and rejection condition.
2. Read only the relevant reference: `attention-kernels.md`, `matmul-kernels.md`, `reduction-kernels.md`, or `elementwise-kernels.md`.
3. Make the smallest attributable change. Do not combine layout, tile, mask, communication, and dtype changes unless their interaction is the hypothesis.
4. Run correctness, then the unchanged measurement policy, then XProf only when ambiguity or component timing requires it.
5. Compare full-device/end-to-end and target-component metrics. Accept, reject, or mark inconclusive with artifacts and complexity trade-offs.

## Start bounded research when experiments repeat

Use `$kernel-foundry` research state with explicit experiment, TPU-hour, and correctness-failure budgets:

```shell
python <kernel-foundry>/scripts/kernel_foundry.py research init \
  --state <research.json> --project <kernel> --mode research \
  --contract <contract-path> --objective full_device_ms:min \
  --max-experiments <n> --max-tpu-hours <hours> --max-failures <n>
```

For every hypothesis, record one expected metric movement and one rejection condition. Use `research add`, `next`, `start`, and `complete`; never mark an experiment accepted when correctness fails. Let `research status` maintain the Pareto frontier.

## Iteration discipline

1. Diagnose the observed phenomenon and plausible causes.
2. Test one attributable cause; use one-gene mutations for genome search unless interactions are the hypothesis.
3. Run correctness before comparable benchmark/profile.
4. Compare full-device/end-to-end time as well as the target component.
5. Accept, reject, or mark inconclusive with artifacts and TPU cost.
6. Stop when the budget is exhausted or no queued hypothesis has a falsifiable expected movement.

Use `$profile-pallas-xprof` for ambiguous, communication-heavy, or sub-millisecond results. Use `$analyze-kernel` for bottleneck interpretation. Use `$kernel-foundry` portfolio, causal, genome, and guardrail commands only after their evidence requirements are met.

## Communication-overlap gate

For ring/pipeline, async copy, DMA, prefetch, remote transfer, or expression-order claims, read `references/overlap-feasibility.md` and `references/communication-overlap-patterns.md`. Measure compute-only, communication-only, serial, and candidate steps before implementing a full pipeline:

```shell
python scripts/overlap_feasibility.py \
  --compute-ms <C> --comm-ms <M> --serial-ms <S> --candidate-ms <O> \
  --full-baseline-ms <baseline> --full-candidate-ms <candidate> \
  --json-out <overlap.json> --markdown-out <overlap.md>
```

Source order is not execution order. Require profiler or structural evidence, memory-residency feasibility, dependency independence, buffer readiness/ownership, and full-latency improvement. A smaller communication-done event alone is not success.

For `single`, report hypothesis, correctness, comparable evidence, decision, artifacts, and next falsifying check. For `research`, also report budget status, queue, accepted/rejected/inconclusive counts, Pareto frontier, and newly promoted guardrails.

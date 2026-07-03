---
name: kernel-foundry
description: "Run an evidence-driven kernel engineering system that turns confirmed failures into executable guardrails, fuzzes operator semantics, manages bounded autonomous experiments, builds shape-specific kernel portfolios, analyzes controlled source-to-HLO changes, and proposes traceable kernel-genome mutations. Use for repeated JAX/Pallas/TPU/GPU kernel development where the agent should improve the process from evidence instead of accumulating prompt rules."
---

# Kernel Foundry

Foundry TPU adapters and replay evidence must use the environment documented by the target repository README. Environment setup is outside Foundry; do not install or upgrade dependencies inside an experiment.

Use this skill as the executable learning layer beneath the kernel stage skills. Keep repository rules and operator semantics authoritative; never let the foundry infer either from performance results.

## Choose the smallest capability

| Need | Command |
| --- | --- |
| Turn a confirmed failure into a reusable check | `guardrail promote`, then `guardrail check` |
| Search for semantic or numerical counterexamples | `fuzz run` |
| Run bounded hypothesis-driven experiments | `research init/add/next/start/complete/status` |
| Compare skill variants on historical work | `replay score` |
| Build an evidence-backed exact dispatch table | `portfolio build` |
| Relate one controlled source change to HLO/metric movement | `causal analyze` |
| Propose traceable one-gene implementation mutations | `genome propose` |
| Test whether the system can reproduce production kernels without reading target implementations | run a sealed holdout challenge from `assets/challenges/` |

Run commands through:

```shell
python kernel-foundry/scripts/kernel_foundry.py <capability> <command> ...
```

Read `references/architecture.md` only when choosing workflow mode or integrating multiple capabilities. Read `references/adapter-contracts.md` before authoring a fuzz adapter or JSON input.

## Non-negotiable boundaries

1. Read every applicable `AGENTS.md` in the actual kernel checkout before edits. A local skill never substitutes for the remote repository contract.
2. Use `quick` mode for a bounded fix, `standard` for a new/ported kernel, and `research` only for repeated experiments. Do not create research paperwork for a quick task.
3. Require an independent trusted reference before optimization. The candidate kernel cannot validate itself.
4. Require correctness before accepting performance, portfolio, causal, or genome conclusions.
5. Treat causal output as controlled association unless the experiment changes exactly one intended variable and rules out confounders.
6. Promote a failure only when its root cause is confirmed, reproducible evidence exists, and prevention is expressed as an executable rule or replay case.
7. Keep kernel-specific numbers and failures in the kernel workspace. Put only generalized guardrails in the shared registry.
8. Let programs validate schemas, transitions, budgets, artifacts, and metric rules. Use model reasoning for semantics, diagnosis, hypothesis generation, and interpretation.
9. A kernel reproduction claim requires target-source isolation, independent adversarial correctness, holdout shapes, repeated comparable benchmarks, and a frozen verdict threshold. Report `competitive` separately from `dominant`.

## Improvement loop

After a substantial iteration:

1. Record the result and raw evidence in the experiment workspace.
2. If a failure is understood, encode a failure record and run `guardrail promote`.
3. Prove the new guardrail catches the historical case and does not fire on its passing control.
4. Add the case to replay evaluation.
5. Change skill text only when routing or escalation behavior changes; do not paste the lesson into multiple skills.

After modifying this skill family, run `python kernel-foundry/scripts/validate_family.py` to validate Python syntax, every skill structure, and all discovered regression suites.

Report the selected mode, commands run, produced artifacts, guardrails added, replay impact, and unresolved semantic or hardware uncertainties.

---
name: kernel-foundry
description: "运行证据驱动的 kernel 工程学习系统：将已确认失败转为可执行 guardrail，进行算子语义 fuzz，管理有边界的自主实验，构建按 shape 分发的 kernel portfolio，分析受控 source-to-HLO 变化，并提出可追踪的 kernel genome 变异。用于重复性的 JAX/Pallas/TPU/GPU kernel 开发，让流程从证据中改进，而不是不断堆积提示词规则。"
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

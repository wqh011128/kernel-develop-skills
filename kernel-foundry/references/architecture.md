# Kernel Foundry Architecture

## Contents

1. Workflow modes
2. Evidence flow
3. Capability boundaries
4. Promotion policy
5. Adoption sequence
6. Production reproduction challenge

## 1. Workflow modes

Choose by uncertainty and repetition, not kernel prestige.

| Mode | Use when | Required evidence | Documentation |
| --- | --- | --- | --- |
| `quick` | Localized fix or known implementation change | repository contract, trusted oracle, targeted correctness | existing project docs; add a short decision record only if behavior changes |
| `standard` | New/ported kernel or semantic/API change | operator contract, reference, correctness matrix, benchmark when performance is claimed | one contract/design record plus evidence summary; reuse repository conventions |
| `research` | Multiple competing hypotheses, tuning, portfolio, causal, or genome work | stable baseline, explicit budgets, isolated experiments, correctness, comparable metrics | machine-readable research state plus concise human summary |

Legacy workspaces with seven kernel documents remain valid. Do not migrate them mechanically. New work should create only documents required by the selected mode and repository contract.

## 2. Evidence flow

```text
operator contract + repository rules
  -> trusted reference
  -> semantic fuzzing and correctness
  -> stable baseline
  -> bounded experiment controller
  -> benchmark/profile evidence
  -> accept/reject/Pareto decision
  -> confirmed failure record
  -> executable guardrail + replay case
```

JSON state is the system of record for budgets, experiment transitions, guardrails, and replay scores. Markdown summarizes decisions for humans; it must not duplicate raw state.

## 3. Capability boundaries

- Guardrails enforce facts already learned. They do not invent root causes.
- Semantic fuzzing finds counterexamples. It does not prove correctness.
- Research state controls budget and transitions. It does not choose a hypothesis without model or human reasoning.
- Portfolio generation emits exact evidence-backed dispatch rules. Generalization to unseen shapes requires explicit validation.
- Causal analysis aggregates controlled pairs. It must not label observational correlation as hardware/compiler causality.
- Genome mutation proposes traceable candidates. A mutation is not accepted until independent correctness and objective metrics pass.

## 4. Promotion policy

Promote a failure only if all are true:

```text
status == confirmed
root cause is stated
at least one durable evidence path exists
a reproduction or replay case exists
scope and non-applicable cases are explicit
the candidate guardrail is declarative and executable
a passing control demonstrates acceptable false-positive behavior
```

Reject promotion for a single unexplained compiler failure, one shape-specific threshold, a personal path/host, or advice that cannot be tested.

## 5. Adoption sequence

1. Build a replay set from historical failures and successful controls.
2. Start with guardrails and semantic fuzzing; both improve correctness without autonomous code changes.
3. Enable research mode with low experiment/TPU budgets and human approval before `start`.
4. Add portfolio, controlled causal pairs, and genome mutation after measurement policy is stable.
5. Expand autonomy only when replay results show fewer violations and no increase in wrong performance conclusions.

## 6. Production reproduction challenge

Use a holdout challenge to test implementation ability rather than helper mechanics:

1. Freeze target kernels, allowed contract sources, workloads, budgets, metrics, and verdict thresholds before implementation.
2. Author candidates without target implementation, registry-runner, or target-doc access. Keep baseline evaluation separate from candidate authorship; procedural blindness is weaker than filesystem-enforced isolation.
3. Require an independent oracle, adversarial values, and shapes not used during design. Score candidate and repository correctness coverage independently; a baseline failure must not count as a candidate failure.
4. Benchmark only after both implementations pass the same default workload. Use multiple rounds and independent process repeats; require no hidden workload regression for a dominance claim.
5. Compare compile coverage, custom-call/HLO structure, full-device latency, numerical error, and reproducibility. `competitive` means useful parity, not superiority.

`assets/challenges/pallas_holdout_reproduction.py` is the first registry-backed TPU example. Its target-specific candidates are evidence fixtures, not generalized guardrails or production code.

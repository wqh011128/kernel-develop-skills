---
name: kernel-dev-lifecycle
description: "将 JAX/Pallas/TPU/GPU kernel 工作路由到最小且安全的流程。用于端到端实现、移植、调试、优化或重复研究；自动选择 quick、standard 或 research 模式，执行仓库、语义、正确性、证据、预算和交付门禁，并将可执行学习交给 kernel-foundry。"
---

# Kernel Development Lifecycle

Coordinate the work; do not duplicate stage instructions or force one document layout on every kernel.

## 1. Establish authority

Before research or edits:

1. Locate the actual local or remote kernel checkout.
2. Read every applicable `AGENTS.md` from repository root to target files.
3. Inspect branch, dirty status, tests, registry/config, CI, HLO export, IR upload, and delivery conventions.
4. Treat repository rules as authoritative. Read `references/repository-contract-review.md` only when discovery is unclear.
5. Build a delivery ledger from every applicable `AGENTS.md` Definition of Done item and every explicit user requirement. Re-read the applicable files after a rebase, branch change, or scope change.
6. For a fresh checkout, read the target repository README and follow its environment setup command exactly. Do not manually install or upgrade dependencies from this skill.
7. Resolve one artifact directory before any write: use the user's directory when provided; otherwise use `<kernel>/docs/` inside the target repository. RFCs, design docs, failure records, research state, and experiment artifacts use this directory. Never write them into the skills repository.

No accessible repository contract means no repository edit. Report what could not be inspected.

## 2. Select one mode

| Mode | Use when | Required process |
| --- | --- | --- |
| `quick` | Bounded fix with known semantics and reference | inspect contract, edit, targeted correctness, repository checks |
| `standard` | New/ported kernel or semantic/API change | confirm operator contract, design only what is uncertain, implement, correctness matrix, benchmark if claimed |
| `research` | Repeated tuning, competing implementations, portfolio, causal, or genome work | stable baseline plus bounded `kernel-foundry research` state |

Do not create a seven-document workspace for `quick`. Preserve legacy workspaces; do not migrate them mechanically.

## 3. Route only required stages

- Unresolved semantics, shapes, dtype, tolerance, hardware, or integration: use `$kernel-goal-discovery`.
- A non-trivial semantic or architectural change: use `$kernel-design-docs`.
- Confirmed contract and implementation request: use `$implement-kernel-from-plan`.
- Correctness/performance diagnosis: use `$analyze-kernel`; capture with `$profile-pallas-xprof` only when needed.
- One evidence-backed optimization: use `$kernel-tuning-loop` in `single` mode.
- Repeated experiments: use `$kernel-tuning-loop` in `research` mode with `$kernel-foundry` state.

## 4. Hard gates

```text
repository contract not reviewed -> no repository edit
operator semantics or trusted reference unresolved -> no optimization
correctness failing -> no performance conclusion
HLO custom-call audit missing or unexplained outer HLO ops present -> do not report completion
measurement policy or baseline unstable -> no speedup claim
experiment not reproducible -> do not accept it as current best
research budget exhausted -> do not start another experiment
failure root cause unconfirmed -> do not promote a shared guardrail
applicable AGENTS.md or Definition of Done item unaccounted for -> do not report completion
required correctness, TPU snapshot, CPU golden/HLO dump, Ruff, or pre-commit command not executed -> do not report completion
required command failed, produced no inspectable artifact, or was silently skipped -> report blocked
```

For every kernel-affecting change, the following acceptance surfaces are mandatory, even when
the repository CI does not discover them automatically:

1. Correctness: run the repository's focused correctness test. For the standard Pallas test
   layout, use the TPU snapshot command below; if the repository has a different entry point,
   pass that exact command to the gate and record the replacement.
2. TPU snapshot: run exactly `python scripts/test_all.py -i <test-file> -o <snapshot-dir>
   --snapshot -c correctness` (use the repository's Python executable, such as `.venv/bin/python`).
   Inspect the result directory and archive; a zero exit code without a result is not a pass.
3. CPU golden/HLO upload contract: run exactly
   `python tools/dump_golden_and_hlo_cpu.py --commit-msg "<ir-upload tag>" --commit
   "$(git rev-parse HEAD)" --out-dir <cpu-dump-dir> --strict`. Inspect the generated dump
   and record whether strict validation passed. The commit may be the current pre-change HEAD;
   this command is an artifact-generation contract, not permission to commit.
4. Ruff check and format, followed by `pre-commit run --all-files`.
5. HLO audit: inspect every TPU snapshot and CPU dump `*_before_opt.hlo`. Record
   the custom-call count and `custom_call_target` values, enumerate every outer
   HLO opcode other than `custom-call`, and compare same-named TPU/CPU dumps.
   Structural opcodes such as `parameter`, `tuple`, and `get-tuple-element` are
   expected. Any other outer opcode (for example `reshape`, `convert`, `copy`,
   `dot`, `gather`, or `reduce`) blocks delivery until its exact opcode is
   explained and explicitly acknowledged with the delivery gate. A zero
   custom-call count or TPU/CPU count/target mismatch is a blocker.

Use `scripts/kernel_delivery_gate.py` as the mechanical audit and executor. With `--run`, it
must receive `--tpu-test-file`, `--snapshot-root`, and `--cpu-dump-out`; missing required
inputs are blockers rather than implicit skips:

```shell
python scripts/kernel_delivery_gate.py --repo <repo> --kernel <kernel> \
  --config <config> --test <test> --snapshot-root <snapshot> \
  --tpu-test-file tests/kernels/test_<kernel>.py \
  --cpu-dump-out <cpu-dump-dir> \
  --commit-message <draft.txt> --pr-text <draft.txt> --run \
  --json-out <delivery-gate.json>
```

Use repeatable `--allow-extra-hlo-op <opcode>` only after inspecting the
reported instruction and recording why it is an intentional wrapper-level
operation. The gate JSON's `hlo_audit` section is the authoritative HLO report.

The gate JSON is the authoritative feedback artifact. The handoff must summarize every check
with its exact command, return code, stdout/stderr tail, artifact path, and pass/fail status.
It must also report each HLO file's custom-call count/targets and all non-custom
outer HLO opcodes, including acknowledged ones.

## 5. Close delivery against the repository contract

Before declaring the work complete:

1. Re-read every applicable `AGENTS.md`, then inspect the final diff and worktree status.
2. Reconcile every delivery-ledger item as `pass`, `not applicable` with a reason, or `blocked`, citing the exact command or artifact. A zero exit code is insufficient when the contract requires inspecting generated snapshots or artifacts.
3. For kernel-affecting changes, derive the exact IR-upload tag syntax from the repository parser, validator, CI, or documented examples. One tag represents one runnable upload matrix item: package, registered kernel, config, test module, and device count. Internal Pallas calls, custom calls, or HLO phases covered by that same item do not each need a tag. Use one tag only when one config/test invocation covers all phases; emit multiple tags for distinct matrix items.
4. Include the exact tag or tags in the handoff, even when no PR is being opened.
5. Inspect `.github/workflows`, pre-commit, Ruff, typing, and test configuration; run every applicable project-native CI command. Treat a missing required tool or an unexecuted applicable CI surface as blocked, not silently skipped.
6. Leave the worktree uncommitted unless the user separately and explicitly authorizes a commit. For implementation changes, draft a message from `assets/commit_message_template.txt` with the exact sections below; do not invent a free-form format:

   ```text
   feat[TOOL]: <imperative summary>

   Task:
   - <task>

   Solution:
   - <solution>

   Test:
   - <exact command and result>

   JIRA: COMPIL-XXXX
   ```

   Include the mandatory acceptance commands in `Test` bullets. Do not put the IR-upload tag
   in the commit message; report it separately in the delivery handoff/PR metadata. Use
   `JIRA: COMPIL-XXXX` only as a placeholder and explicitly remind the user to replace it.
   Validate the draft with the delivery gate. Drafting text never authorizes `git commit`, push,
   or PR creation.

The handoff must include the delivery ledger, exact CPU and accelerator commands, local CI results, tag decisions, the validated commit-message draft with a JIRA reminder, and whether the worktree remains uncommitted.

## 6. Make learning automatic and safe

Every kernel task ends with a learning checkpoint, even when the task succeeds:

1. Scan the task ledger, failed commands, correctness mismatches, performance regressions, blocked checks, and user corrections.
2. Classify each lesson as `local` (only this kernel), `candidate` (possibly reusable but not proven), or `confirmed` (reproduced root cause, durable evidence, and a passing control).
3. For `candidate`, write a machine-readable failure record in the resolved artifact directory; do not change shared skills.
4. For `confirmed`, use `$kernel-foundry` to promote the record to an executable guardrail, check it on the failing facts and passing control, and add the case to replay evaluation.
5. Only after replay/eval shows a measurable improvement may a shared skill rule or helper be proposed. Keep the raw failure record, guardrail result, replay result, and proposed diff together.

Do not paste a lesson into several skills, and do not silently edit shared `SKILL.md` files during a kernel task. The automatic part is evidence capture, classification, guardrail/replay compilation, and eval; shared prompt changes remain an explicit, reviewable change.

Report mode, repository contract sources, correctness, claimed performance evidence, per-item Definition of Done status, exact IR-upload tag decisions, uncommitted status and commit-message draft, foundry state/guardrails when used, and unresolved risks.

---
name: kernel-design-docs
description: "Create the minimum decision and semantic contract needed for a new or changed JAX/Pallas/TPU/GPU kernel. Use after goals are confirmed when math, distributed equivalence, API boundaries, validation, rollout, or competing designs require durable documentation. Adapts to quick, standard, or research mode and preserves existing repository conventions instead of forcing a fixed seven-document package."
---

# Kernel Design Records

Document uncertainty and durable decisions, not the entire workflow.

## Select the document surface

| Mode | Default |
| --- | --- |
| `quick` | No new design file; update existing project docs only if externally visible behavior changes |
| `standard` | One RFC based on the repository template or bundled `references/RFC_template.md` |
| `research` | The RFC plus machine-readable `kernel-foundry research` state and a concise evidence summary |

Keep an existing `docs/README.md`, `RFC.md`, `math.md`, `results.md`, `fail-notes.md`, `impl-notes.md`, and `optimization.md` layout coherent when the project already uses it. Do not create or migrate to that layout by default.

## RFC template gate

When an RFC is required, use the repository's applicable RFC template first; otherwise read and copy `references/RFC_template.md`. Preserve its numbered structure, fill it from repository evidence and the confirmed operator contract, remove instructional placeholders, and mark a genuinely non-applicable section with a short reason instead of silently deleting it. Keep the document UTF-8. Do not invent a different RFC outline ad hoc.

## Required content by risk

- Always record source of truth, input/output contract, target shapes/dtypes, trusted reference, tolerance, and falsifying tests.
- Add mathematical derivation when local/block/rank computation is not obviously equivalent to global semantics.
- Add mask, padding, normalization, accumulation, and numerical-stability rules when relevant.
- Add API/module boundaries and compatibility only when they change.
- Add alternatives and rollout only for a real architectural or integration decision.
- Put raw correctness, benchmark, HLO, and XProf data in artifacts; summarize only the evidence supporting decisions.

Do not duplicate repository facts or machine-readable research state in Markdown. Mark unresolved backend behavior as an experiment, not a design conclusion.

The design is sufficient when another engineer can implement the intended semantics, identify the oracle, run falsifying tests, and know which choices remain experimental. Report created/updated records, unresolved decisions, and the next executable validation.

---
name: kernel-design-docs
description: "Create a complete Chinese design-document package for a new or modified kernel under tmp/{kernel_name}_{date}/docs. Use after kernel goals are confirmed and before implementation. Produces README, a full RFC that includes the development plan, mathematical derivation, results plan, pitfalls, implementation notes, and optimization notes, while enforcing the standard experiments layout."
---

# Kernel Design Docs

Use this skill after the kernel goal has been confirmed. Do not start implementation from this skill unless the user explicitly asks to continue.

## Required Output Layout

Create or update:

```text
tmp/{kernel_name}_{YYYYMMDD}/docs/
  README.md
  rfc.md
  math.md
  results.md
  fail-notes.md
  impl-notes.md
  optimization.md
```

Also create the experiment root if it does not exist:

```text
tmp/{kernel_name}_{YYYYMMDD}/experiments/
```

Rules:

```text
kernel_name: lowercase snake_case.
README.md: always uppercase.
Use exactly the docs above for new work.
Do not create `develop-plan.md` for new work; the plan belongs inside `rfc.md`.
Do not create old numbered docs such as 00-rfc.md.
Do not create a top-level results/ directory for new artifacts.
```

## Research Requirement

Before writing `math.md`, research enough primary or high-signal sources to verify the operator semantics:

```text
current repository implementations, tests, benchmarks, and registry
official framework/backend documentation
papers or algorithm notes when relevant
at least one readable open-source implementation unless the op is project-private
```

If math is uncertain, mark the uncertainty and ask the user or add a validation gate. Do not present uncertain math as a conclusion.

## Document Responsibilities

## Document Writing Contract

All docs must be written in Chinese by default, encoded as UTF-8, and structured for future agents to continue work without reading raw logs first.

Use these rules for every doc:

```text
Use stable headings with numbered sections when the doc is long.
Put the conclusion or current status before details.
Separate status, facts, evidence, decisions, and next actions.
Use tables for matrices and short code blocks for commands, paths, shapes, and metrics.
Keep raw logs out of docs; link artifact paths instead.
When a result changes, update both the experiment README and the relevant top-level summary doc.
Do not mix design, results, failure notes, and optimization history in one doc.
Do not leave mojibake or mixed-language placeholder text in final docs.
```

Apply this per-section writing logic:

```text
Status sections:
  state what is accepted, rejected, unverified, or blocked.
  include the latest known implementation and artifact path.

Evidence sections:
  cite commands, shapes, tolerances, benchmark medians, XProf/report paths, or source references.
  never claim speedup, correctness, MFU, memory bottleneck, or overlap without an artifact.

Decision sections:
  record the decision, the reason, and the accept/reject condition that was used.
  do not hide neutral or rejected experiments if they changed the next plan.

Next-action sections:
  list ordered hypotheses or tasks.
  each task must have a validation gate and a rejection condition when it affects performance.

Reference sections:
  distinguish repository facts, official docs, papers, and inferred conclusions.
  mark unresolved math or backend behavior as uncertainty, not fact.
```

Use these content boundaries:

```text
README.md is a navigation and status index; it is not a design proposal or raw result report.
rfc.md is the high-level contract and plan; it is not an experiment log.
math.md is the semantic proof; it is not a performance tuning note.
results.md is the evidence summary; it is not an optimization diary.
optimization.md is the hypothesis/decision loop; it is not a place for raw benchmark dumps.
impl-notes.md is the code/layout/API boundary record; it is not a user-facing README.
fail-notes.md is for concise reusable pitfalls within this kernel workspace; it is not a full failure transcript.
```

`README.md`:

```text
Use this structure:

# <kernel> Docs

## 1. Current Status
current best, correctness status, performance status, active XProf URL/path

## 2. Scope
in scope, out of scope, hard constraints

## 3. Docs Map
one table mapping each doc to its purpose

## 4. Experiments Index
one table: experiment, status, purpose, key artifact/report

## 5. Current Best / Next Step
current implementation, why it is current best, next hypothesis

## 6. Legacy / Migration Notes
only if legacy artifacts exist
```

`rfc.md`:

```text
Write in Chinese. Use this exact structure:

# RFC XXXX: <title>

## 1. Summary
- current existing capability
- current missing capability
- this RFC proposal
- change boundary

## 2. Problem Statement
Table columns: work domain, current pain point, business/engineering impact.

## 3. Context
### 3.1 Current behavior / architecture
### 3.2 Relevant background and constraints
### 3.3 Technical environment
### 3.4 Technical positioning

## 4. Current Status / Progress
- completed
- verified
- not completed
- high uncertainty

## 5. Goals
### Goal 1
### Goal 2

## 6. Non-Goals

## 7. Proposed Design
### 7.1 Responsibility matrix
### 7.2 Core flow
### 7.3 Configuration interface
### 7.4 Module design
### 7.5 Error handling and validation
### 7.6 Compatibility strategy

## 8. Interfaces / Contracts

## 9. Alternatives Considered

## 10. Risks / Trade-offs

## 11. Validation / Testing Plan

## 12. Rollout / Migration Plan

## 13. Tasks / Ownership

## 14. Open Questions

## 15. Decision Log
```

The RFC is the single high-level planning document. Put phases, go/no-go gates, acceptance criteria, rollback/rejection conditions, and ownership in sections 11-15. Do not duplicate them in a separate plan document.

`math.md`:

```text
Use this structure:

# Math: <kernel>

## 1. Symbols And Shapes
symbol definitions and input/output shapes

## 2. Global Semantics
complete mathematical formula

## 3. Local / Block / Distributed Semantics
partitioning, block equations, rank equations

## 4. Equivalence Proof
prove global semantics equals local/block/rank semantics

## 5. Masking, Padding, And Boundaries
causal, padding, sequence boundary, invalid element behavior

## 6. Dtype And Numerical Stability
input dtype, accumulator dtype, LSE/softmax rules, tolerance expectations

## 7. Reference Pseudocode
minimal executable-style reference logic

## 8. Data Flow
diagram or stepwise data flow when useful

## 9. Required Edge Cases
edge cases that must be tested
```

For attention, scan, reduction, or distributed kernels, explicitly prove the equivalence between global semantics and local/block/rank semantics.

`results.md`:

```text
Use this structure:

# Results: <kernel>

## 1. Current Verdict
accepted current best, rejected paths, and whether performance claims are proven

## 2. Correctness Matrix
table: experiment, command, shapes, tolerance, status, artifact

## 3. Benchmark Summary
table: experiment, shape, baseline, target, median, speedup, artifact

## 4. XProf / Analysis Summary
local URL, profile path, analysis report, bottleneck class, key component movement

## 5. Current Best
current best implementation and why

## 6. Open Validation Gaps
missing shapes, missing profiles, unproven claims
```

`fail-notes.md`:

```text
Use this structure:

# Fail Notes: <kernel>

## 1. Pitfall Index
table: pitfall, affected experiment, status, short lesson

## 2. Rejected Directions
one section per rejected direction: hypothesis, evidence, shortest root cause, what not to repeat

## 3. Correctness Failures
only understood failures; include symptom, cause, fix

## 4. Reusable Kernel-Specific Lessons
short lessons useful for this kernel workspace
```

Do not paste long raw logs here.

`impl-notes.md`:

```text
Use this structure:

# Implementation Notes: <kernel>

## 1. File And API Boundaries
new files, untouched files, public/experimental APIs

## 2. Data Layout And Dtypes
physical layout, padding, dtype and accumulator choices

## 3. Communication / Kernel Split
what is done by framework collectives and what is done by Pallas/local kernels

## 4. Implementation Variants
accepted and experimental APIs, status, and integration state

## 5. Known Constraints
hard-coded assumptions, unsupported cases, compile/runtime caveats
```

`optimization.md`:

```text
Use this structure:

# Optimization: <kernel>

## 1. Baseline
stable baseline, target shapes, metrics, artifact paths

## 2. Bottleneck Classification
roofline class, XProf evidence, component ranking

## 3. Accepted Optimizations
one section per accepted change: hypothesis, evidence, decision, retained code path

## 4. Rejected / Neutral Optimizations
one section per rejected change: hypothesis, evidence, reason, what not to repeat

## 5. Current Hypothesis Queue
ordered next experiments with acceptance and rejection conditions

## 6. Process Notes
experiment process notes that affect next-step thinking
```

## Output To User

Report in Chinese:

```text
docs root
created/updated docs
unresolved questions
recommended next stage
```

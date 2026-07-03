---
name: kernel-design-docs
description: "Create a complete Chinese design-document package for a new or modified kernel under tmp/{kernel_name}_{date}/docs. Use after kernel goals are confirmed and before implementation. Produces README, RFC, mathematical derivation, development plan, validation/results plan, pitfalls, implementation notes, and optimization notes, while enforcing the standard experiments layout."
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
  develop-plan.md
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

`README.md`:

```text
quick docs map
current status
scope
experiment index
current best implementation, if any
current XProf URL/path, if any
legacy or migration notes, if any
```

`rfc.md`:

```text
goal
non-goals
input/output contract
target scenario
hardware/backend
first implementation strategy
expected deliverables
acceptance criteria
major risks
```

`math.md`:

```text
symbol definitions
input/output shapes
complete mathematical formula
mask, causal, padding, boundary behavior
dtype and accumulation rules
parallel/block/distributed equivalence derivation
numerical stability requirements
reference pseudocode
data-flow diagram when useful
edge cases that must be tested
```

For attention, scan, reduction, or distributed kernels, explicitly prove the equivalence between global semantics and local/block/rank semantics.

`develop-plan.md`:

```text
phases
files to change or create
what is intentionally out of scope
go/no-go gates
rollback or rejection conditions
```

`results.md`:

```text
correctness matrix and status
benchmark summary
XProf summary and local URL/path
analyze-kernel report path
current best implementation
open validation gaps
```

`fail-notes.md`:

```text
concise pitfalls
rejected directions
root-cause summaries
lessons that are useful for this kernel
```

Do not paste long raw logs here.

`impl-notes.md`:

```text
implementation boundaries
public APIs
layout and dtype choices
communication and kernel split
new files and integration status
known constraints
```

`optimization.md`:

```text
baseline
accepted/rejected hypotheses
comparison artifacts
decision reasoning
next experiments
```

## Output To User

Report in Chinese:

```text
docs root
created/updated docs
unresolved questions
recommended next stage
```

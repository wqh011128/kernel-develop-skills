---
name: implement-kernel-from-plan
description: "Implement or modify a JAX/Pallas/TPU/GPU kernel from a confirmed operator contract and trusted validation oracle. Use for bounded fixes, new kernels, ports, or research candidates after material semantic decisions are resolved. Follows the real repository contract, keeps changes attributable, and establishes correctness before performance work."
---

# Implement Kernel From Contract

1. Read applicable `AGENTS.md`, the confirmed operator/design contract, reference, tests, call sites, registry/config, similar kernels, and repository delivery rules.
2. Confirm the reference is independent and covers non-default public parameters. Stop if the candidate and oracle share an unverified semantic shortcut.
3. Make the smallest complete change that tests the design. Preserve existing APIs and ownership boundaries unless the contract explicitly changes them.
4. Validate in this order: import/compile smoke, small deterministic correctness, representative shapes, adversarial/edge shapes, then benchmark/profile.
5. After rank, mask, grouping, padding, communication, state, dtype, or accumulation changes, invalidate prior performance evidence and rerun correctness.
6. Return to `$kernel-dev-lifecycle` for the repository delivery ledger, CI audit, IR-upload tag, and commit-message draft. Do not duplicate or weaken its close-delivery gate here.

Minimum correctness evidence includes shape, dtype, seed, oracle, tolerance, finite-value checks, error metrics, non-default public modes/scales, and artifact/command. For normalized reductions or attention, validate the normalization state (`LSE`, denominator, or equivalent) when exposed or reconstructable.

For a single bounded implementation, use repository-native files and tests; do not create an experiment hierarchy. For competing/repeated candidates, use `$kernel-foundry` research state and isolate each candidate according to repository conventions. Remote scratch must remain disposable; copy durable evidence to the authorized local workspace.

If a confirmed failure can recur across kernels, hand it to `$kernel-foundry` for guardrail promotion after reproducing it against a passing control.

Report changed files, contract assumptions, correctness evidence, untested cases, and whether performance work is now permitted.

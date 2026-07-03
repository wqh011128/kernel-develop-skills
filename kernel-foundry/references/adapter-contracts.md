# Kernel Foundry Adapter and JSON Contracts

## Contents

1. Semantic fuzz adapter
2. Failure and guardrail records
3. Research result
4. Portfolio rows
5. Controlled causal observations
6. Genome specification
7. Replay results

## 1. Semantic fuzz adapter

Provide a Python file with:

```python
def generate_case(rng, index):
    """Return a JSON-serializable case."""

def evaluate(case):
    """Return {'passed': bool, 'checks': [...], 'metrics': {...}}."""

def shrink(case):  # optional
    """Yield smaller JSON-serializable candidates."""
```

`evaluate` must synchronize device work before returning. Compare against an independent oracle and include separate checks for primary output, normalization state, finite values, and mathematical invariants when applicable.
An exception raised by `evaluate` is recorded as a shrinkable failure. Returned cases, checks, and metrics must be JSON-serializable.

For a registry-backed PallasKernels example, copy `assets/adapters/pallas-matmul-fuzz-adapter.py` into the experiment workspace and set the checkout on `PYTHONPATH`. Keep project-specific adapters outside the repository under test.
For an end-to-end genome/portfolio/controlled-HLO replay, use `assets/adapters/pallas-matmul-genome-benchmark.py` with a genome proposal report. It writes raw benchmark evidence plus inputs accepted by `portfolio build` and `causal analyze`.

## 2. Failure and guardrail records

A promotable failure contains `id`, `title`, `status: confirmed`, `scope`, `root_cause`, non-empty `evidence`, `reproduction`, and `prevention.guardrail`.

Start from `assets/failure.template.json`; replace every placeholder and prove both a failing replay and passing control before promotion.

A condition has:

```json
{"path": "candidate.full_device_ms", "op": "lt", "other_path": "baseline.full_device_ms"}
```

Supported operators are `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `exists`, `truthy`, `contains`, and `in`. A rule applies when every `when` condition passes. It violates when any `assertions` condition fails.

## 3. Research result

Use:

```json
{
  "correctness": {"status": "pass", "artifact": "..."},
  "metrics": {"full_device_ms": 1.2},
  "tpu_hours": 0.25,
  "artifacts": ["..."],
  "conclusion": "accepted"
}
```

An accepted result requires correctness `pass`. The controller enforces experiment and TPU-hour budgets but does not decide whether the reference is trustworthy.
Set `estimated_tpu_hours` on costly hypotheses; `research start` refuses work whose estimate would cross the remaining budget.
Use correctness `not_run` when execution stopped before the correctness stage and `blocked` when an external prerequisite prevented it. Never encode compile, environment, or artifact failures as numerical correctness failures.
Use `effective_status` and `stop_reasons` for automation. The persisted project status remains an explicit human/agent decision; budget exhaustion is reported without silently marking the research objective complete.

## 4. Portfolio rows

Each row contains `candidate`, `dimensions`, `metrics`, and `correctness`. Add one row per independent repeat under the same measurement policy. Use `--min-repeats` and `--min-relative-margin` before emitting a dispatch choice. The builder aggregates each candidate by median, emits unresolved regions when evidence is insufficient or margins are ambiguous, and never interpolates unseen shapes.

## 5. Controlled causal observations

Each pair contains one `baseline` and one `candidate` observation with the same `pair_id`. Store `source_features`, `hlo_features`, `metrics`, and an equal `context` on both observations for shape, dtype, hardware, or other workload dimensions that may change the effect. Change one intended source variable per pair where possible.
Use distinct pair ids for repeats and contexts. Associations are grouped by source transition and context. Treat a contextual association with `consistent_direction: false` as unresolved even when its mean delta looks favorable.

## 6. Genome specification

Provide `base` gene values, `search_space` lists, optional conditional `constraints`, optional numeric `gene_rules` (`min`, `max`, `multiple_of`), and `seen` genome fingerprints. The proposer changes one gene at a time and preserves lineage.

## 7. Replay results

Each result contains `case_id`, `variant`, and metrics such as `success`, `constraint_violations`, `wrong_performance_conclusion`, `human_corrections`, `duration_minutes`, `tpu_hours`, `documentation_noise`, `artifact_completeness`, and `reproducible`.

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from foundry.causal import analyze_pairs  # noqa: E402
from foundry.common import FoundryError  # noqa: E402
from foundry.fuzzing import run_fuzzer  # noqa: E402
from foundry.genome import propose_mutations  # noqa: E402
from foundry.guardrails import check_registry, promote_failure  # noqa: E402
from foundry.portfolio import build_portfolio  # noqa: E402
from foundry.replay import score_replay  # noqa: E402
from foundry.research import (  # noqa: E402
    add_hypothesis,
    complete_experiment,
    init_state,
    start_experiment,
    status,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FoundryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_failure_promotes_to_executable_guardrail(self) -> None:
        failure = self.root / "failure.json"
        registry = self.root / "guardrails.json"
        facts = self.root / "facts.json"
        write_json(
            failure,
            {
                "id": "local-faster-full-slower",
                "title": "Reject local-only speedup",
                "status": "confirmed",
                "scope": ["performance"],
                "root_cause": "Collective time moved outside the custom call.",
                "evidence": ["results/xprof/report.json"],
                "reproduction": "replay/local_faster_full_slower.json",
                "prevention": {
                    "guardrail": {
                        "id": "full-device-must-not-regress",
                        "description": "A local win cannot justify a full-device regression.",
                        "scope": ["performance"],
                        "when": [
                            {
                                "path": "candidate.custom_call_ms",
                                "op": "lt",
                                "other_path": "baseline.custom_call_ms",
                            }
                        ],
                        "assertions": [
                            {
                                "path": "candidate.full_device_ms",
                                "op": "le",
                                "other_path": "baseline.full_device_ms",
                            }
                        ],
                        "action": "reject_performance_claim",
                    }
                },
            },
        )
        promoted = promote_failure(failure, registry)
        self.assertEqual(promoted["source_failure"], "local-faster-full-slower")
        write_json(
            facts,
            {
                "baseline": {"custom_call_ms": 1.0, "full_device_ms": 1.5},
                "candidate": {"custom_call_ms": 0.8, "full_device_ms": 1.7},
            },
        )
        report = check_registry(registry, facts, ["performance"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["results"][0]["action"], "reject_performance_claim")

    def test_research_state_enforces_correctness_and_builds_pareto(self) -> None:
        state = self.root / "research.json"
        init_state(
            state,
            "gqa",
            "research",
            "contract.md",
            [{"name": "full_device_ms", "direction": "min"}],
            5,
            2.0,
            2,
        )
        for index, latency in enumerate((1.2, 1.0)):
            hypothesis_id = f"h{index}"
            add_hypothesis(
                state,
                {
                    "id": hypothesis_id,
                    "statement": "Change one tile gene.",
                    "target_metric": "full_device_ms",
                    "expected_movement": "decrease",
                    "rejection_condition": "no reproducible improvement",
                },
            )
            experiment_id = f"e{index}"
            start_experiment(state, hypothesis_id, experiment_id)
            complete_experiment(
                state,
                experiment_id,
                {
                    "correctness": {"status": "pass", "artifact": f"correctness-{index}.json"},
                    "metrics": {"full_device_ms": latency},
                    "tpu_hours": 0.1,
                    "artifacts": [f"benchmark-{index}.json"],
                    "conclusion": "accepted",
                },
            )
        summary = status(state)
        self.assertEqual(summary["pareto_frontier"], ["e1"])
        self.assertEqual(summary["budget"]["experiments"]["used"], 2)
        self.assertEqual(summary["effective_status"], "waiting_for_hypothesis")

    def test_research_refuses_estimated_budget_overrun(self) -> None:
        state = self.root / "budget.json"
        init_state(
            state,
            "gqa",
            "research",
            "contract.md",
            [{"name": "ms", "direction": "min"}],
            2,
            0.1,
            1,
        )
        add_hypothesis(
            state,
            {
                "id": "expensive",
                "statement": "Run a large sweep.",
                "target_metric": "ms",
                "expected_movement": "decrease",
                "rejection_condition": "no win",
                "estimated_tpu_hours": 0.2,
            },
        )
        with self.assertRaisesRegex(FoundryError, "exceed the research budget"):
            start_experiment(state, "expensive", "e")

    def test_research_records_pre_correctness_block_without_faking_failure(self) -> None:
        state = self.root / "blocked.json"
        init_state(
            state,
            "fp8",
            "research",
            "contract.md",
            [{"name": "correctness_pass", "direction": "max"}],
            2,
            1.0,
            1,
        )
        add_hypothesis(
            state,
            {
                "id": "vmem",
                "statement": "The default config compiles within scoped VMEM.",
                "target_metric": "correctness_pass",
                "expected_movement": "reach correctness",
                "rejection_condition": "compiler resource exhaustion",
            },
        )
        start_experiment(state, "vmem", "vmem-run")
        result = complete_experiment(
            state,
            "vmem-run",
            {
                "correctness": {"status": "not_run", "reason": "compile_vmem_oom"},
                "metrics": {"correctness_pass": 0},
                "tpu_hours": 0.01,
                "artifacts": ["compiler.log"],
                "conclusion": "rejected",
            },
        )
        self.assertEqual(result["result"]["correctness"]["status"], "not_run")
        self.assertEqual(status(state)["budget"]["failures"]["used"], 0)

    def test_fuzzer_minimizes_counterexample(self) -> None:
        adapter = self.root / "adapter.py"
        adapter.write_text(
            """
def generate_case(rng, index):
    return {"values": [0, 0, 1, 0, 0] if index == 0 else [0]}

def evaluate(case):
    failed = 1 in case["values"]
    return {"passed": not failed, "checks": [{"name": "no_one", "passed": not failed}]}

def shrink(case):
    values = case["values"]
    for i in range(len(values)):
        yield {"values": values[:i] + values[i + 1:]}
""".lstrip(),
            encoding="utf-8",
        )
        report = run_fuzzer(adapter, iterations=2, seed=7, output_dir=self.root / "fuzz")
        self.assertFalse(report["passed"])
        failure = json.loads((self.root / "fuzz" / report["failure_files"][0]).read_text(encoding="utf-8"))
        self.assertEqual(failure["minimized_case"], {"values": [1]})

    def test_fuzzer_records_candidate_exception(self) -> None:
        adapter = self.root / "crashing_adapter.py"
        adapter.write_text(
            """
def generate_case(rng, index):
    return {"value": index}

def evaluate(case):
    raise RuntimeError("candidate crashed")
""".lstrip(),
            encoding="utf-8",
        )
        report = run_fuzzer(adapter, iterations=1, seed=0, output_dir=self.root / "crash")
        failure = json.loads((self.root / "crash" / report["failure_files"][0]).read_text(encoding="utf-8"))
        self.assertEqual(failure["result"]["exception"]["type"], "RuntimeError")

    def test_portfolio_uses_only_correct_exact_regions(self) -> None:
        source = self.root / "portfolio.json"
        output = self.root / "dispatch.json"
        write_json(
            source,
            {
                "rows": [
                    {"candidate": "a", "dimensions": {"n": 128}, "metrics": {"ms": 2.0}, "correctness": True},
                    {"candidate": "b", "dimensions": {"n": 128}, "metrics": {"ms": 1.0}, "correctness": True},
                    {"candidate": "bad", "dimensions": {"n": 256}, "metrics": {"ms": 0.1}, "correctness": False},
                ]
            },
        )
        report = build_portfolio(source, "ms", "min", output)
        self.assertEqual(report["rules"][0]["candidate"], "b")
        self.assertFalse(report["unseen_dimensions_supported"])
        self.assertEqual(report["rejected_rows"], 1)

    def test_portfolio_requires_requested_repeat_evidence(self) -> None:
        source = self.root / "repeat_portfolio.json"
        output = self.root / "repeat_dispatch.json"
        write_json(
            source,
            {
                "rows": [
                    {"candidate": "a", "dimensions": {"n": 128}, "metrics": {"ms": 1.0}, "correctness": True},
                    {"candidate": "b", "dimensions": {"n": 128}, "metrics": {"ms": 0.9}, "correctness": True},
                ]
            },
        )
        report = build_portfolio(source, "ms", "min", output, min_repeats=2)
        self.assertEqual(report["covered_regions"], 0)
        self.assertEqual(report["rules"][0]["status"], "insufficient_repeats")

    def test_causal_analysis_labels_association_not_causation(self) -> None:
        source = self.root / "pairs.json"
        output = self.root / "causal.json"
        write_json(
            source,
            {
                "observations": [
                    {"pair_id": "p", "role": "baseline", "context": {"n": 128}, "source_features": {"tile": 64}, "hlo_features": {"copies": 2}, "metrics": {"ms": 2.0}},
                    {"pair_id": "p", "role": "candidate", "context": {"n": 128}, "source_features": {"tile": 128}, "hlo_features": {"copies": 1}, "metrics": {"ms": 1.5}},
                ]
            },
        )
        report = analyze_pairs(source, "ms", output)
        self.assertEqual(report["claim_level"], "controlled_association_not_causation")
        self.assertEqual(report["associations"][0]["mean_metric_delta"], -0.5)
        self.assertEqual(report["associations"][0]["context"], {"n": 128})
        self.assertTrue(report["associations"][0]["consistent_direction"])

    def test_genome_proposes_single_gene_mutations(self) -> None:
        spec = self.root / "genome.json"
        output = self.root / "proposals.json"
        write_json(
            spec,
            {
                "base": {"tile_m": 64, "buffers": 1},
                "search_space": {"tile_m": [64, 96, 128], "buffers": [1, 2]},
                "gene_rules": {"tile_m": {"multiple_of": 64}},
                "constraints": [{"if": {"buffers": 2}, "then": {"tile_m": 64}}],
            },
        )
        report = propose_mutations(spec, output, limit=10)
        self.assertEqual(report["base"], {"tile_m": 64, "buffers": 1})
        self.assertEqual(len(report["proposals"]), 2)
        self.assertEqual(report["rejected_by_constraints"], 1)
        self.assertTrue(all(len(item["mutation"]) == 3 for item in report["proposals"]))

    def test_replay_scores_variants(self) -> None:
        suite = self.root / "suite.json"
        results = self.root / "results.json"
        output = self.root / "score.json"
        write_json(suite, {"cases": [{"id": "case-a"}], "variants": ["none", "thin"]})
        write_json(
            results,
            {
                "results": [
                    {"case_id": "case-a", "variant": "none", "metrics": {"success": False, "constraint_violations": 1}},
                    {"case_id": "case-a", "variant": "thin", "metrics": {"success": True, "constraint_violations": 0}},
                ]
            },
        )
        report = score_replay(suite, results, output)
        summaries = {item["variant"]: item for item in report["variants"]}
        self.assertEqual(summaries["thin"]["metrics"]["success"], 1.0)
        self.assertTrue(report["complete"])

    def test_schemas_are_valid_json_and_cli_help_works(self) -> None:
        for schema in (ROOT / "assets" / "schemas").glob("*.json"):
            json.loads(schema.read_text(encoding="utf-8"))
        replay_suite = json.loads((ROOT / "assets" / "replay-suite.template.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(replay_suite["cases"]), 10)
        self.assertEqual(len(replay_suite["variants"]), 3)
        compile(
            (ROOT / "assets" / "adapters" / "pallas-matmul-fuzz-adapter.py").read_text(encoding="utf-8"),
            "pallas-matmul-fuzz-adapter.py",
            "exec",
        )
        compile(
            (ROOT / "assets" / "adapters" / "pallas-matmul-genome-benchmark.py").read_text(encoding="utf-8"),
            "pallas-matmul-genome-benchmark.py",
            "exec",
        )
        completed = subprocess.run(
            [sys.executable, str(SCRIPTS / "kernel_foundry.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("guardrail", completed.stdout)


if __name__ == "__main__":
    unittest.main()

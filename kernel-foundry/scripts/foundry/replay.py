from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from . import SCHEMA_VERSION
from .common import dump_json, load_json, now_utc, require, require_fields


BOOLEAN_METRICS = {"success", "wrong_performance_conclusion", "reproducible"}
NUMERIC_METRICS = {
    "constraint_violations",
    "human_corrections",
    "duration_minutes",
    "tpu_hours",
    "documentation_noise",
    "artifact_completeness",
}


def score_replay(suite_path: Path, results_path: Path, output_path: Path) -> dict[str, Any]:
    suite = load_json(suite_path)
    results_payload = load_json(results_path)
    require(isinstance(suite, dict) and isinstance(results_payload, dict), "suite and results must be JSON objects")
    cases = suite.get("cases")
    variants = suite.get("variants")
    results = results_payload.get("results")
    require(isinstance(cases, list) and cases, "suite.cases must be non-empty")
    require(isinstance(variants, list) and variants, "suite.variants must be non-empty")
    require(isinstance(results, list), "results.results must be a list")
    case_ids = {case["id"] for case in cases}
    require(len(case_ids) == len(cases), "suite case ids must be unique")
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for result in results:
        require_fields(result, ("case_id", "variant", "metrics"), "replay result")
        require(result["case_id"] in case_ids, f"Unknown replay case: {result['case_id']}")
        require(result["variant"] in variants, f"Unknown replay variant: {result['variant']}")
        key = (result["case_id"], result["variant"])
        require(key not in seen, f"Duplicate replay result: {key}")
        seen.add(key)
        by_variant[result["variant"]].append(result)
    summaries = []
    for variant in variants:
        variant_results = by_variant.get(variant, [])
        metric_values: dict[str, list[float]] = defaultdict(list)
        for result in variant_results:
            metrics = result["metrics"]
            for name in BOOLEAN_METRICS:
                if name in metrics:
                    metric_values[name].append(float(bool(metrics[name])))
            for name in NUMERIC_METRICS:
                if name in metrics:
                    metric_values[name].append(float(metrics[name]))
        summary = {
            "variant": variant,
            "completed_cases": len(variant_results),
            "coverage": len(variant_results) / len(cases),
            "metrics": {name: mean(values) for name, values in sorted(metric_values.items()) if values},
        }
        summaries.append(summary)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_utc(),
        "case_count": len(cases),
        "variants": summaries,
        "complete": all(summary["completed_cases"] == len(cases) for summary in summaries),
        "interpretation": "Prefer higher success/reproducible/artifact_completeness and lower violations, wrong conclusions, corrections, time, TPU hours, and documentation noise.",
    }
    dump_json(output_path, report)
    return report

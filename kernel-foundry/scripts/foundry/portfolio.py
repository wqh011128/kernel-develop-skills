from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from . import SCHEMA_VERSION
from .common import dump_json, load_json, now_utc, require, require_fields, stable_id


def build_portfolio(
    input_path: Path,
    metric: str,
    direction: str,
    output_path: Path,
    min_repeats: int = 1,
    min_relative_margin: float = 0.0,
) -> dict[str, Any]:
    require(direction in {"min", "max"}, "direction must be min or max")
    require(min_repeats > 0, "min_repeats must be positive")
    require(min_relative_margin >= 0, "min_relative_margin must be non-negative")
    payload = load_json(input_path)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    require(isinstance(rows, list) and rows, "portfolio input must contain non-empty rows")
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    dimensions_by_key: dict[str, dict[str, Any]] = {}
    rejected = 0
    for row in rows:
        require_fields(row, ("candidate", "dimensions", "metrics", "correctness"), "portfolio row")
        if row["correctness"] is not True or metric not in row["metrics"]:
            rejected += 1
            continue
        key = stable_id(row["dimensions"])
        dimensions_by_key[key] = row["dimensions"]
        groups[key][row["candidate"]].append(row)
    require(groups, "No correct rows contain the requested metric")
    rules = []
    for key, candidate_rows in groups.items():
        candidates = [
            {
                "candidate": candidate,
                "metric": median(row["metrics"][metric] for row in rows),
                "repeats": len(rows),
                "values": [row["metrics"][metric] for row in rows],
            }
            for candidate, rows in candidate_rows.items()
        ]
        eligible = [candidate for candidate in candidates if candidate["repeats"] >= min_repeats]
        ordered = sorted(eligible, key=lambda item: item["metric"], reverse=direction == "max")
        status = "selected"
        winner = ordered[0] if ordered else None
        relative_margin = None
        if winner is None:
            status = "insufficient_repeats"
        elif len(ordered) > 1:
            runner_up = ordered[1]
            denominator = max(abs(runner_up["metric"]), 1e-12)
            relative_margin = abs(runner_up["metric"] - winner["metric"]) / denominator
            if relative_margin < min_relative_margin:
                status = "ambiguous_margin"
        rules.append(
            {
                "dimensions": dimensions_by_key[key],
                "status": status,
                "candidate": winner["candidate"] if status == "selected" else None,
                "metric": winner["metric"] if status == "selected" else None,
                "relative_margin": relative_margin,
                "candidate_evidence": candidates,
            }
        )
    rules.sort(key=lambda item: str(sorted(item["dimensions"].items())))
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_utc(),
        "metric": metric,
        "direction": direction,
        "min_repeats": min_repeats,
        "min_relative_margin": min_relative_margin,
        "policy": "exact_observed_dimensions_only",
        "rules": rules,
        "covered_regions": sum(rule["status"] == "selected" for rule in rules),
        "unresolved_regions": sum(rule["status"] != "selected" for rule in rules),
        "rejected_rows": rejected,
        "unseen_dimensions_supported": False,
    }
    dump_json(output_path, report)
    return report

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .common import dump_json, flatten_dict, load_json, now_utc, require, require_fields


def _changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    left = flatten_dict(before)
    right = flatten_dict(after)
    return {
        key: {"before": left.get(key), "after": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def analyze_pairs(input_path: Path, metric: str, output_path: Path) -> dict[str, Any]:
    payload = load_json(input_path)
    observations = payload.get("observations") if isinstance(payload, dict) else None
    require(isinstance(observations, list) and observations, "causal input must contain observations")
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for observation in observations:
        require_fields(
            observation,
            ("pair_id", "role", "source_features", "hlo_features", "metrics"),
            "causal observation",
        )
        require(observation["role"] in {"baseline", "candidate"}, "role must be baseline or candidate")
        require(observation["role"] not in pairs[observation["pair_id"]], "Each pair may contain one observation per role")
        pairs[observation["pair_id"]][observation["role"]] = observation
    controlled_pairs = []
    association_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    association_contexts: dict[tuple[str, str], dict[str, Any]] = {}
    skipped = []
    for pair_id, pair in pairs.items():
        if set(pair) != {"baseline", "candidate"}:
            skipped.append({"pair_id": pair_id, "reason": "incomplete_pair"})
            continue
        baseline, candidate = pair["baseline"], pair["candidate"]
        baseline_context = baseline.get("context", {})
        candidate_context = candidate.get("context", {})
        if baseline_context != candidate_context:
            skipped.append({"pair_id": pair_id, "reason": "context_mismatch"})
            continue
        if metric not in baseline["metrics"] or metric not in candidate["metrics"]:
            skipped.append({"pair_id": pair_id, "reason": "missing_metric"})
            continue
        source_changes = _changes(baseline["source_features"], candidate["source_features"])
        hlo_changes = _changes(baseline["hlo_features"], candidate["hlo_features"])
        delta = candidate["metrics"][metric] - baseline["metrics"][metric]
        controlled = len(source_changes) == 1
        record = {
            "pair_id": pair_id,
            "controlled_single_source_change": controlled,
            "source_changes": source_changes,
            "hlo_changes": hlo_changes,
            "metric_delta": delta,
            "context": baseline_context,
        }
        controlled_pairs.append(record)
        if controlled:
            key, transition = next(iter(source_changes.items()))
            association = f"{key}:{transition['before']!r}->{transition['after']!r}"
            context_key = json.dumps(baseline_context, ensure_ascii=False, sort_keys=True)
            group_key = (association, context_key)
            association_contexts[group_key] = baseline_context
            association_deltas[group_key].append(delta)
    associations = [
        {
            "source_transition": key[0],
            "context": association_contexts[key],
            "pair_count": len(values),
            "mean_metric_delta": sum(values) / len(values),
            "min_metric_delta": min(values),
            "max_metric_delta": max(values),
            "improved_count": sum(value < 0 for value in values),
            "regressed_count": sum(value > 0 for value in values),
            "unchanged_count": sum(value == 0 for value in values),
            "consistent_direction": all(value <= 0 for value in values) or all(value >= 0 for value in values),
            "metric": metric,
        }
        for key, values in sorted(association_deltas.items())
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_utc(),
        "metric": metric,
        "claim_level": "controlled_association_not_causation",
        "pairs": controlled_pairs,
        "associations": associations,
        "skipped": skipped,
    }
    dump_json(output_path, report)
    return report

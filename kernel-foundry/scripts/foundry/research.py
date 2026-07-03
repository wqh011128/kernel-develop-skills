from __future__ import annotations

from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .common import dump_json, load_json, now_utc, require, require_fields


MODES = {"quick", "standard", "research"}
CONCLUSIONS = {"accepted", "rejected", "inconclusive"}
CORRECTNESS_STATUSES = {"pass", "fail", "not_run", "blocked"}


def init_state(
    path: Path,
    project: str,
    mode: str,
    contract: str,
    objectives: list[dict[str, str]],
    max_experiments: int,
    max_tpu_hours: float,
    max_failures: int,
) -> dict[str, Any]:
    require(mode in MODES, f"mode must be one of {sorted(MODES)}")
    require(max_experiments > 0, "max_experiments must be positive")
    require(max_tpu_hours >= 0, "max_tpu_hours must be non-negative")
    require(max_failures > 0, "max_failures must be positive")
    require(objectives, "At least one objective is required")
    for objective in objectives:
        require_fields(objective, ("name", "direction"), "objective")
        require(objective["direction"] in {"min", "max"}, "objective direction must be min or max")
    state = {
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "mode": mode,
        "contract": contract,
        "status": "active",
        "created_at": now_utc(),
        "updated_at": now_utc(),
        "budgets": {
            "max_experiments": max_experiments,
            "max_tpu_hours": max_tpu_hours,
            "max_failures": max_failures,
        },
        "objectives": objectives,
        "hypotheses": [],
        "experiments": [],
        "pareto_frontier": [],
    }
    dump_json(path, state)
    return state


def _load_state(path: Path) -> dict[str, Any]:
    state = load_json(path)
    require(isinstance(state, dict), "research state must be a JSON object")
    require_fields(state, ("budgets", "objectives", "hypotheses", "experiments"), "research state")
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_utc()
    state["pareto_frontier"] = pareto_frontier(state)
    dump_json(path, state)


def add_hypothesis(path: Path, hypothesis: dict[str, Any]) -> dict[str, Any]:
    state = _load_state(path)
    require_fields(
        hypothesis,
        ("id", "statement", "target_metric", "expected_movement", "rejection_condition"),
        "hypothesis",
    )
    require(all(item["id"] != hypothesis["id"] for item in state["hypotheses"]), f"Duplicate hypothesis id: {hypothesis['id']}")
    record = {
        **hypothesis,
        "priority": int(hypothesis.get("priority", 100)),
        "estimated_tpu_hours": float(hypothesis.get("estimated_tpu_hours", 0.0)),
        "status": "queued",
        "created_at": now_utc(),
    }
    require(record["estimated_tpu_hours"] >= 0, "estimated_tpu_hours must be non-negative")
    state["hypotheses"].append(record)
    _save_state(path, state)
    return record


def _spent(state: dict[str, Any]) -> tuple[int, float, int]:
    completed = [item for item in state["experiments"] if item["status"] == "completed"]
    tpu_hours = sum(float(item.get("result", {}).get("tpu_hours", 0)) for item in completed)
    failures = sum(item.get("result", {}).get("correctness", {}).get("status") == "fail" for item in completed)
    return len(state["experiments"]), tpu_hours, failures


def budget_status(state: dict[str, Any]) -> dict[str, Any]:
    experiments, tpu_hours, failures = _spent(state)
    budgets = state["budgets"]
    return {
        "experiments": {"used": experiments, "limit": budgets["max_experiments"]},
        "tpu_hours": {"used": tpu_hours, "limit": budgets["max_tpu_hours"]},
        "failures": {"used": failures, "limit": budgets["max_failures"]},
        "exhausted": (
            experiments >= budgets["max_experiments"]
            or tpu_hours >= budgets["max_tpu_hours"]
            or failures >= budgets["max_failures"]
        ),
    }


def next_hypothesis(path: Path) -> dict[str, Any] | None:
    state = _load_state(path)
    if budget_status(state)["exhausted"]:
        return None
    queued = [item for item in state["hypotheses"] if item["status"] == "queued"]
    return min(queued, key=lambda item: (item["priority"], item["created_at"])) if queued else None


def start_experiment(path: Path, hypothesis_id: str, experiment_id: str) -> dict[str, Any]:
    state = _load_state(path)
    require(not budget_status(state)["exhausted"], "Research budget is exhausted")
    require(not any(item["status"] == "running" for item in state["experiments"]), "Another experiment is already running")
    require(all(item["id"] != experiment_id for item in state["experiments"]), f"Duplicate experiment id: {experiment_id}")
    hypothesis = next((item for item in state["hypotheses"] if item["id"] == hypothesis_id), None)
    require(hypothesis is not None, f"Unknown hypothesis: {hypothesis_id}")
    require(hypothesis["status"] == "queued", f"Hypothesis is not queued: {hypothesis_id}")
    used_tpu_hours = budget_status(state)["tpu_hours"]["used"]
    require(
        used_tpu_hours + hypothesis.get("estimated_tpu_hours", 0.0) <= state["budgets"]["max_tpu_hours"],
        "Estimated TPU cost would exceed the research budget",
    )
    hypothesis["status"] = "running"
    experiment = {
        "id": experiment_id,
        "hypothesis_id": hypothesis_id,
        "status": "running",
        "started_at": now_utc(),
    }
    state["experiments"].append(experiment)
    _save_state(path, state)
    return experiment


def complete_experiment(path: Path, experiment_id: str, result: dict[str, Any]) -> dict[str, Any]:
    state = _load_state(path)
    require_fields(result, ("correctness", "metrics", "tpu_hours", "artifacts", "conclusion"), "experiment result")
    require(result["conclusion"] in CONCLUSIONS, f"conclusion must be one of {sorted(CONCLUSIONS)}")
    require(
        result["correctness"].get("status") in CORRECTNESS_STATUSES,
        f"correctness.status must be one of {sorted(CORRECTNESS_STATUSES)}",
    )
    require(float(result["tpu_hours"]) >= 0, "tpu_hours must be non-negative")
    if result["conclusion"] == "accepted":
        require(result["correctness"]["status"] == "pass", "Cannot accept an experiment with failing correctness")
        for objective in state["objectives"]:
            require(objective["name"] in result["metrics"], f"Accepted result is missing objective metric: {objective['name']}")
    experiment = next((item for item in state["experiments"] if item["id"] == experiment_id), None)
    require(experiment is not None, f"Unknown experiment: {experiment_id}")
    require(experiment["status"] == "running", f"Experiment is not running: {experiment_id}")
    experiment.update({"status": "completed", "completed_at": now_utc(), "result": result})
    hypothesis = next(item for item in state["hypotheses"] if item["id"] == experiment["hypothesis_id"])
    hypothesis["status"] = result["conclusion"]
    _save_state(path, state)
    return experiment


def _dominates(left: dict[str, Any], right: dict[str, Any], objectives: list[dict[str, str]]) -> bool:
    no_worse = True
    strictly_better = False
    for objective in objectives:
        name = objective["name"]
        left_value = left["result"]["metrics"][name]
        right_value = right["result"]["metrics"][name]
        if objective["direction"] == "min":
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
        else:
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value
    return no_worse and strictly_better


def pareto_frontier(state: dict[str, Any]) -> list[str]:
    candidates = [
        item
        for item in state["experiments"]
        if item["status"] == "completed"
        and item.get("result", {}).get("conclusion") == "accepted"
        and item.get("result", {}).get("correctness", {}).get("status") == "pass"
        and all(objective["name"] in item.get("result", {}).get("metrics", {}) for objective in state["objectives"])
    ]
    return [
        candidate["id"]
        for candidate in candidates
        if not any(_dominates(other, candidate, state["objectives"]) for other in candidates if other is not candidate)
    ]


def status(path: Path) -> dict[str, Any]:
    state = _load_state(path)
    budget = budget_status(state)
    next_item = next_hypothesis(path)
    running = any(item["status"] == "running" for item in state["experiments"])
    stop_reasons = []
    if budget["exhausted"]:
        stop_reasons.append("budget_exhausted")
    if next_item is None and not running:
        stop_reasons.append("no_queued_hypothesis")
    if state["status"] != "active":
        effective_status = state["status"]
    elif budget["exhausted"]:
        effective_status = "budget_exhausted"
    elif running:
        effective_status = "running"
    elif next_item is None:
        effective_status = "waiting_for_hypothesis"
    else:
        effective_status = "ready"
    return {
        "project": state["project"],
        "mode": state["mode"],
        "status": state["status"],
        "effective_status": effective_status,
        "stop_reasons": stop_reasons,
        "budget": budget,
        "next_hypothesis": next_item,
        "pareto_frontier": pareto_frontier(state),
        "hypothesis_counts": {
            key: sum(item["status"] == key for item in state["hypotheses"])
            for key in ("queued", "running", "accepted", "rejected", "inconclusive")
        },
    }

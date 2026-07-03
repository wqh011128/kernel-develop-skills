from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .common import FoundryError, MISSING, dump_json, get_path, load_json, now_utc, require, require_fields


SUPPORTED_OPS = {"eq", "ne", "lt", "le", "gt", "ge", "exists", "truthy", "contains", "in"}


def _condition_value(condition: dict[str, Any], facts: dict[str, Any]) -> tuple[Any, Any]:
    actual = get_path(facts, condition["path"], MISSING)
    if "other_path" in condition:
        expected = get_path(facts, condition["other_path"], MISSING)
    else:
        expected = condition.get("value", MISSING)
    return actual, expected


def evaluate_condition(condition: dict[str, Any], facts: dict[str, Any]) -> bool:
    require_fields(condition, ("path", "op"), "condition")
    op = condition["op"]
    require(op in SUPPORTED_OPS, f"Unsupported condition operator: {op}")
    actual, expected = _condition_value(condition, facts)
    if op == "exists":
        return (actual is not MISSING) is bool(condition.get("value", True))
    if op == "truthy":
        return actual is not MISSING and bool(actual) is bool(condition.get("value", True))
    if actual is MISSING:
        return False
    require(expected is not MISSING, f"Condition {condition['path']} requires value or other_path")
    try:
        return {
            "eq": lambda: actual == expected,
            "ne": lambda: actual != expected,
            "lt": lambda: actual < expected,
            "le": lambda: actual <= expected,
            "gt": lambda: actual > expected,
            "ge": lambda: actual >= expected,
            "contains": lambda: expected in actual,
            "in": lambda: actual in expected,
        }[op]()
    except (TypeError, KeyError) as exc:
        raise FoundryError(
            f"Cannot evaluate {condition['path']} {op}: actual={actual!r}, expected={expected!r}"
        ) from exc


def validate_rule(rule: dict[str, Any]) -> None:
    require_fields(rule, ("id", "description", "scope", "when", "assertions", "action"), "guardrail")
    require(isinstance(rule["scope"], list) and rule["scope"], "guardrail.scope must be non-empty")
    require(isinstance(rule["when"], list), "guardrail.when must be a list")
    require(isinstance(rule["assertions"], list) and rule["assertions"], "guardrail.assertions must be non-empty")
    for condition in [*rule["when"], *rule["assertions"]]:
        require_fields(condition, ("path", "op"), "guardrail condition")
        require(condition["op"] in SUPPORTED_OPS, f"Unsupported condition operator: {condition['op']}")


def compile_failure(failure: dict[str, Any]) -> dict[str, Any]:
    require_fields(
        failure,
        ("id", "title", "status", "scope", "root_cause", "evidence", "reproduction", "prevention"),
        "failure",
    )
    require(failure["status"] == "confirmed", "Only confirmed failures can be promoted")
    require(isinstance(failure["scope"], list) and failure["scope"], "failure.scope must be non-empty")
    require(bool(str(failure["root_cause"]).strip()), "failure.root_cause must be stated")
    require(isinstance(failure["evidence"], list) and failure["evidence"], "failure.evidence must be non-empty")
    require(bool(failure["reproduction"]), "failure.reproduction must point to a durable replay or command")
    prevention = failure["prevention"]
    require(isinstance(prevention, dict) and "guardrail" in prevention, "failure.prevention.guardrail is required")
    rule = copy.deepcopy(prevention["guardrail"])
    rule.setdefault("id", f"guardrail-{failure['id']}")
    rule.setdefault("description", failure["title"])
    rule.setdefault("scope", failure["scope"])
    rule["source_failure"] = failure["id"]
    rule["compiled_at"] = now_utc()
    validate_rule(rule)
    return rule


def promote_failure(failure_path: Path, registry_path: Path) -> dict[str, Any]:
    failure = load_json(failure_path)
    require(isinstance(failure, dict), "failure record must be a JSON object")
    rule = compile_failure(failure)
    if registry_path.exists():
        registry = load_json(registry_path)
    else:
        registry = {"schema_version": SCHEMA_VERSION, "guardrails": []}
    require(isinstance(registry, dict), "guardrail registry must be a JSON object")
    rules = registry.setdefault("guardrails", [])
    require(isinstance(rules, list), "guardrail registry.guardrails must be a list")
    require(all(existing.get("id") != rule["id"] for existing in rules), f"Duplicate guardrail id: {rule['id']}")
    rules.append(rule)
    registry["updated_at"] = now_utc()
    dump_json(registry_path, registry)
    return rule


def evaluate_rule(rule: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    validate_rule(rule)
    applicable = all(evaluate_condition(condition, facts) for condition in rule["when"])
    if not applicable:
        return {"id": rule["id"], "status": "not_applicable", "action": None, "failed": []}
    failed = [condition for condition in rule["assertions"] if not evaluate_condition(condition, facts)]
    return {
        "id": rule["id"],
        "status": "violation" if failed else "pass",
        "action": rule["action"] if failed else None,
        "failed": failed,
    }


def check_registry(registry_path: Path, facts_path: Path, scopes: list[str]) -> dict[str, Any]:
    registry = load_json(registry_path)
    facts = load_json(facts_path)
    require(isinstance(registry, dict) and isinstance(facts, dict), "registry and facts must be JSON objects")
    selected = []
    for rule in registry.get("guardrails", []):
        validate_rule(rule)
        if scopes and not set(scopes).intersection(rule["scope"]):
            continue
        selected.append(evaluate_rule(rule, facts))
    violations = [result for result in selected if result["status"] == "violation"]
    return {
        "schema_version": SCHEMA_VERSION,
        "checked_at": now_utc(),
        "scopes": scopes,
        "results": selected,
        "violation_count": len(violations),
        "passed": not violations,
    }

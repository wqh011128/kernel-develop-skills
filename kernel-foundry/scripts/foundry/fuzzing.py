from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from . import SCHEMA_VERSION
from .common import FoundryError, dump_json, now_utc, require, stable_id


def load_adapter(path: Path) -> ModuleType:
    require(path.exists(), f"Fuzz adapter does not exist: {path}")
    spec = importlib.util.spec_from_file_location(f"kernel_fuzz_adapter_{stable_id(str(path))}", path)
    require(spec is not None and spec.loader is not None, f"Cannot load fuzz adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "generate_case", None)), "Fuzz adapter must define generate_case(rng, index)")
    require(callable(getattr(module, "evaluate", None)), "Fuzz adapter must define evaluate(case)")
    return module


def _validate_result(result: Any) -> dict[str, Any]:
    require(isinstance(result, dict), "Fuzz adapter evaluate(case) must return a dict")
    require(isinstance(result.get("passed"), bool), "Fuzz result must contain boolean passed")
    checks = result.get("checks", [])
    require(isinstance(checks, list), "Fuzz result checks must be a list")
    try:
        json.dumps(result, ensure_ascii=False)
    except TypeError as exc:
        raise FoundryError(f"Fuzz result must be JSON-serializable: {exc}") from exc
    return result


def _evaluate(module: ModuleType, case: Any) -> dict[str, Any]:
    try:
        return _validate_result(module.evaluate(case))
    except FoundryError:
        raise
    except Exception as exc:  # A candidate crash is a shrinkable semantic failure.
        return {
            "passed": False,
            "checks": [{"name": "adapter_exception", "passed": False}],
            "exception": {"type": type(exc).__name__, "message": str(exc)},
        }


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def minimize_failure(module: ModuleType, original: Any, max_rounds: int = 100) -> tuple[Any, dict[str, Any]]:
    current = original
    current_result = _evaluate(module, current)
    require(not current_result["passed"], "Cannot minimize a passing case")
    shrink = getattr(module, "shrink", None)
    if not callable(shrink):
        return current, current_result
    seen = {stable_id(current)}
    for _ in range(max_rounds):
        improved = False
        candidates: Iterable[Any] = shrink(current)
        for candidate in sorted(candidates, key=_json_size):
            fingerprint = stable_id(candidate)
            if fingerprint in seen or _json_size(candidate) >= _json_size(current):
                continue
            seen.add(fingerprint)
            result = _evaluate(module, candidate)
            if not result["passed"]:
                current, current_result, improved = candidate, result, True
                break
        if not improved:
            break
    return current, current_result


def run_fuzzer(adapter_path: Path, iterations: int, seed: int, output_dir: Path, stop_after: int = 0) -> dict[str, Any]:
    require(iterations > 0, "iterations must be positive")
    require(stop_after >= 0, "stop_after must be non-negative")
    module = load_adapter(adapter_path)
    rng = random.Random(seed)
    failures: list[dict[str, Any]] = []
    seen_failures: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    executed = 0
    for index in range(iterations):
        case = module.generate_case(rng, index)
        try:
            json.dumps(case, ensure_ascii=False)
        except TypeError as exc:
            raise FoundryError(f"Generated case {index} is not JSON-serializable: {exc}") from exc
        result = _evaluate(module, case)
        executed += 1
        if result["passed"]:
            continue
        minimized, minimized_result = minimize_failure(module, case)
        fingerprint = stable_id({"case": minimized, "checks": minimized_result.get("checks", [])}, "fuzz-")
        if fingerprint in seen_failures:
            continue
        seen_failures.add(fingerprint)
        failure = {
            "id": fingerprint,
            "seed": seed,
            "case_index": index,
            "original_case": case,
            "minimized_case": minimized,
            "result": minimized_result,
        }
        failures.append(failure)
        dump_json(output_dir / f"{fingerprint}.json", failure)
        if stop_after and len(failures) >= stop_after:
            break
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_utc(),
        "adapter": str(adapter_path.resolve()),
        "seed": seed,
        "requested_iterations": iterations,
        "executed_iterations": executed,
        "unique_failures": len(failures),
        "passed": not failures,
        "failure_files": [f"{failure['id']}.json" for failure in failures],
    }
    dump_json(output_dir / "fuzz_report.json", report)
    return report

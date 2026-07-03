#!/usr/bin/env python3
"""Batch profile registry-backed PallasKernels configs on a remote TPU.

This orchestrates the production workflow:

1. Upload the generic registry runner to /tmp on the remote host.
2. Run one-step XProf capture for each config.
3. Tar the remote profile atomically.
4. Copy the tarball locally and extract under tmp/xprof/{config}/{timestamp}.
5. Generate local XProf tool data and inspect custom-call metrics.
6. Classify profile/lowering failures and validate FLOPs trust.
7. Write machine-readable JSON and a Markdown coverage report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "pallas_xprof_registry_runner.py"
TOOLS = SCRIPT_DIR / "xprof_pallas_tools.py"
LIBTPU_FLAGS = (
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true"
)


def _ssh_base(args: argparse.Namespace) -> list[str]:
  cmd = ["ssh"]
  if args.identity_file:
    cmd += ["-i", args.identity_file]
  cmd += [
      "-o",
      "IdentitiesOnly=yes",
      "-o",
      "StrictHostKeyChecking=no",
      args.host,
  ]
  return cmd


def _scp_base(args: argparse.Namespace) -> list[str]:
  cmd = ["scp"]
  if args.identity_file:
    cmd += ["-i", args.identity_file]
  cmd += [
      "-o",
      "IdentitiesOnly=yes",
      "-o",
      "StrictHostKeyChecking=no",
  ]
  return cmd


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
  proc = subprocess.run(
      cmd,
      cwd=str(cwd) if cwd else None,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      timeout=timeout,
      check=False,
  )
  if check and proc.returncode != 0:
    raise RuntimeError(
        "command failed\n"
        f"cmd={cmd}\n"
        f"returncode={proc.returncode}\n"
        f"stdout={proc.stdout}\n"
        f"stderr={proc.stderr}"
    )
  return proc


def _classify_failure(error: str) -> dict[str, str]:
  text = error or ""
  if "aic.block_matmul" in text and "unregistered dialect" in text:
    return {
        "failure_class": "kernel_lowering_unregistered_aic_block_matmul",
        "failure_owner": "kernel_or_environment",
        "failure_note": (
            "Kernel lowering failed before profiling because aic.block_matmul "
            "was emitted with an unregistered MLIR dialect."
        ),
    }
  if "No TPU" in text or "no TPU" in text or "Could not find registered platform" in text:
    return {
        "failure_class": "environment_no_tpu_backend",
        "failure_owner": "environment",
        "failure_note": "JAX could not see a TPU backend; profile capture did not start.",
    }
  if "LIBTPU_INIT_ARGS is missing" in text:
    return {
        "failure_class": "environment_missing_libtpu_trace_flags",
        "failure_owner": "profile_workflow",
        "failure_note": "Required custom-call trace flags were not set before importing JAX.",
    }
  if "No .xplane.pb found" in text or "xplane may be corrupt" in text:
    return {
        "failure_class": "profile_artifact_missing_or_corrupt",
        "failure_owner": "profile_workflow",
        "failure_note": "Profile capture/download did not produce a valid xplane artifact.",
    }
  if "timed out" in text.lower() or "TimeoutExpired" in text:
    return {
        "failure_class": "timeout",
        "failure_owner": "unknown",
        "failure_note": "Command timed out; rerun with a larger timeout or isolate the kernel.",
    }
  return {
      "failure_class": "unknown",
      "failure_owner": "unknown",
      "failure_note": "Failure did not match a known profile/lowering/environment pattern.",
  }


def _remote(args: argparse.Namespace, command: str, *, timeout: int | None = None):
  return _run(_ssh_base(args) + [command], timeout=timeout)


def _preflight_remote(args: argparse.Namespace) -> dict[str, Any]:
  script = r'''
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys

required_flags = [
    "--xla_enable_custom_call_region_trace=true",
    "--xla_xprof_register_llo_debug_info=true",
]
out = {
    "python": sys.executable,
    "cwd": os.getcwd(),
    "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    "packages": {},
    "imports": {},
    "git": {},
    "configs": {},
    "checks": {},
}

for package in ("jax", "jaxlib", "libtpu", "xprof", "PyYAML"):
  try:
    out["packages"][package] = md.version(package)
  except Exception as exc:
    out["packages"][package] = f"missing: {exc.__class__.__name__}"

try:
  import yaml  # noqa: F401
  out["imports"]["yaml"] = True
except Exception as exc:
  out["imports"]["yaml"] = repr(exc)

try:
  from pallas_kernels.kernels.registry.lookup import get_kernel  # noqa: F401
  out["imports"]["pallas_registry"] = True
except Exception as exc:
  out["imports"]["pallas_registry"] = repr(exc)

try:
  import jax
  devices = jax.devices()
  out["jax_version"] = jax.__version__
  out["devices"] = [str(device) for device in devices]
  out["tpu_device_count"] = sum(1 for device in devices if getattr(device, "platform", None) == "tpu")
except Exception as exc:
  out["jax_error"] = repr(exc)
  out["devices"] = []
  out["tpu_device_count"] = 0

for name, cmd in {
    "branch": ["git", "branch", "--show-current"],
    "commit": ["git", "rev-parse", "HEAD"],
    "status_short": ["git", "status", "--short"],
}.items():
  try:
    out["git"][name] = subprocess.check_output(cmd, text=True).strip()
  except Exception as exc:
    out["git"][name] = repr(exc)

config_dir = Path("pallas_kernels/configs")
try:
  configs = sorted(path.stem for path in config_dir.glob("*.yaml"))
except Exception:
  configs = []
out["configs"]["count"] = len(configs)
out["configs"]["sample"] = configs[:10]

flags = out["libtpu_init_args"]
out["checks"]["required_libtpu_flags"] = all(flag in flags for flag in required_flags)
out["checks"]["has_tpu"] = out["tpu_device_count"] > 0
out["checks"]["yaml_import"] = out["imports"].get("yaml") is True
out["checks"]["registry_import"] = out["imports"].get("pallas_registry") is True
out["checks"]["has_configs"] = out["configs"]["count"] > 0
out["ok"] = all(out["checks"].values())
print(json.dumps(out, sort_keys=True))
'''
  cmd = (
      f"cd {shlex.quote(args.remote_repo)} && "
      f"LIBTPU_INIT_ARGS={shlex.quote(LIBTPU_FLAGS)} "
      f"{shlex.quote(args.remote_python)} - <<'PY'\n{script}\nPY"
  )
  proc = _run(_ssh_base(args) + [cmd], timeout=180, check=False)
  try:
    payload = _extract_json(proc.stdout)
  except Exception as exc:
    payload = {
        "ok": False,
        "parse_error": repr(exc),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
  payload["ssh_returncode"] = proc.returncode
  payload["ssh_stderr_tail"] = proc.stderr[-4000:]
  return payload


def _write_preflight(args: argparse.Namespace, payload: dict[str, Any]) -> Path:
  args.local_root.mkdir(parents=True, exist_ok=True)
  path = args.local_root / f"{getattr(args, 'report_stem', 'batch_profile_report')}_preflight.json"
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
  print(f"wrote {path}")
  return path


def _upload_runner(args: argparse.Namespace) -> None:
  if not RUNNER.exists():
    raise SystemExit(f"missing runner script: {RUNNER}")
  _run(_scp_base(args) + [str(RUNNER), f"{args.host}:/tmp/pallas_xprof_registry_runner.py"])
  _remote(args, "chmod +x /tmp/pallas_xprof_registry_runner.py")


def _enumerate_configs(args: argparse.Namespace) -> list[str]:
  cmd = (
      f"cd {shlex.quote(args.remote_repo)} && "
      "find pallas_kernels/configs -maxdepth 1 -type f -name '*.yaml' "
      "-printf '%f\\n' | sed 's/[.]yaml$//' | sort"
  )
  proc = _remote(args, cmd, timeout=60)
  configs = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
  if args.config:
    wanted = set(args.config)
    configs = [cfg for cfg in configs if cfg in wanted]
    missing = sorted(wanted - set(configs))
    if missing:
      raise SystemExit(f"requested configs not found remotely: {missing}")
  if args.limit:
    configs = configs[: args.limit]
  return configs


def _extract_json(stdout: str) -> dict[str, Any]:
  decoder = json.JSONDecoder()
  for match in reversed(list(re.finditer(r"\{", stdout))):
    tail = stdout[match.start():]
    try:
      payload, end = decoder.raw_decode(tail)
    except json.JSONDecodeError:
      continue
    if tail[end:].strip():
      continue
    if not isinstance(payload, dict):
      continue
    return payload
  raise RuntimeError(f"could not find JSON object at end of stdout:\n{stdout}")


def _profile_one(args: argparse.Namespace, config: str, mode: str = "public") -> dict[str, Any]:
  remote_cmd = (
      f"cd {shlex.quote(args.remote_repo)} && "
      f"LIBTPU_INIT_ARGS={shlex.quote(LIBTPU_FLAGS)} "
      f"{shlex.quote(args.remote_python)} /tmp/pallas_xprof_registry_runner.py "
      f"--repo {shlex.quote(args.remote_repo)} "
      f"--config {shlex.quote(config)} "
      f"--out-root {shlex.quote(args.remote_out_root)} "
      f"--warmup {int(args.warmup)} "
      f"--mode {shlex.quote(mode)}"
  )
  proc = _remote(args, remote_cmd, timeout=args.remote_timeout)
  manifest = _extract_json(proc.stdout)
  manifest["remote_stdout_tail"] = proc.stdout[-4000:]
  manifest["remote_stderr_tail"] = proc.stderr[-4000:]
  return manifest


def _safe_extract(tar_path: Path, dest: Path) -> None:
  dest = dest.resolve()
  with tarfile.open(tar_path, "r:gz") as tf:
    for member in tf.getmembers():
      target = (dest / member.name).resolve()
      if not str(target).startswith(str(dest)):
        raise RuntimeError(f"unsafe tar member path: {member.name}")
    try:
      tf.extractall(dest, filter="data")
    except TypeError:
      tf.extractall(dest)


def _download_one(args: argparse.Namespace, manifest: dict[str, Any]) -> Path:
  config = manifest["config"]
  trace_root = manifest["trace_root"]
  trace_root_posix = PurePosixPath(trace_root)
  timestamp = trace_root_posix.name
  remote_config_dir = str(trace_root_posix.parent)
  tar_name = f"{config}_{timestamp}.tgz"
  remote_tar = f"/tmp/{tar_name}"
  local_config_dir = args.local_root / config
  local_config_dir.mkdir(parents=True, exist_ok=True)
  local_tar = local_config_dir / tar_name

  tar_cmd = (
      f"cd {shlex.quote(remote_config_dir)} && "
      f"tar -czf {shlex.quote(remote_tar)} {shlex.quote(timestamp)} && "
      f"test -s {shlex.quote(remote_tar)}"
  )
  _remote(args, tar_cmd, timeout=120)
  _run(_scp_base(args) + [f"{args.host}:{remote_tar}", str(local_tar)], timeout=300)
  _safe_extract(local_tar, local_config_dir)
  return local_config_dir / timestamp


def _inspect_one(args: argparse.Namespace, local_profile: Path, config: str) -> dict[str, Any]:
  gen = _run(
      [
          args.local_python,
          str(TOOLS),
          "generate-cache",
          "--profile-dir",
          str(local_profile),
      ],
      timeout=args.local_timeout,
      check=False,
  )
  ins = _run(
      [
          args.local_python,
          str(TOOLS),
          "inspect",
          "--profile-dir",
          str(local_profile),
          "--json",
      ],
      timeout=args.local_timeout,
      check=False,
  )
  payload: dict[str, Any] | None = None
  if ins.returncode == 0:
    try:
      # xprof may emit logs before JSON; parse the final object.
      payload = json.loads(_extract_json(ins.stdout) if not ins.stdout.lstrip().startswith("{") else ins.stdout)
    except Exception:
      match = re.search(r"(\{\s*\"inspection_source\".*\})\s*$", ins.stdout, re.DOTALL)
      if match:
        payload = json.loads(match.group(1))
  return {
      "generate_cache_returncode": gen.returncode,
      "generate_cache_stdout_tail": gen.stdout[-4000:],
      "generate_cache_stderr_tail": gen.stderr[-4000:],
      "inspect_returncode": ins.returncode,
      "inspect_stdout_tail": ins.stdout[-4000:],
      "inspect_stderr_tail": ins.stderr[-4000:],
      "inspect": payload,
  }


def _custom_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return [
      row for row in rows
      if str(row.get("category")) == "custom-call" or "custom-call" in str(row.get("name"))
  ]


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  seen = set()
  out = []
  for row in rows:
    key = (
        row.get("source"),
        row.get("name"),
        row.get("raw_time_ps"),
        row.get("avg_time_ps"),
        row.get("avg_time_us"),
        row.get("raw_flops"),
        row.get("model_flops"),
        row.get("model_flops_v2"),
    )
    if key in seen:
      continue
    seen.add(key)
    out.append(row)
  return out


def _row_time_us(row: dict[str, Any]) -> float | None:
  if row.get("avg_time_us") is not None:
    return float(row["avg_time_us"])
  if row.get("avg_time_ps") is not None:
    return float(row["avg_time_ps"]) / 1e6
  if row.get("raw_time_ps") is not None:
    return float(row["raw_time_ps"]) / 1e6
  return None


def _row_flops(row: dict[str, Any]) -> int | None:
  for key in ("raw_flops", "model_flops_v2", "model_flops", "bf16_flops"):
    value = row.get(key)
    if value is not None:
      return int(float(value))
  return None


def _summarize_rows(rows: list[dict[str, Any]]) -> tuple[int, float | None, int | None]:
  custom_rows = _dedupe_rows(_custom_rows(rows))
  if not custom_rows:
    return 0, None, None
  op_profile_rows = [row for row in custom_rows if row.get("source") == "op_profile"]
  detailed_op_profile_rows = [
      row for row in op_profile_rows if str(row.get("name")) != "custom-call"
  ]
  source_rows = detailed_op_profile_rows or op_profile_rows or custom_rows
  durations = [value for row in source_rows if (value := _row_time_us(row)) is not None]
  flops = [_row_flops(row) for row in source_rows]
  flops = [value for value in flops if value is not None]
  duration_us = max(durations) if durations else None
  raw_flops = sum(flops) if flops else None
  return len(custom_rows), duration_us, raw_flops


def _positive_flops(value: Any) -> float | None:
  try:
    flops = float(value)
  except (TypeError, ValueError):
    return None
  if flops <= 0:
    return None
  return flops


def _expected_flops_from_manifest(manifest: dict[str, Any]) -> float | None:
  cost_analysis = manifest.get("cost_analysis")
  if isinstance(cost_analysis, dict):
    return _positive_flops(cost_analysis.get("flops"))
  if isinstance(cost_analysis, list):
    total = 0.0
    for item in cost_analysis:
      if not isinstance(item, dict):
        continue
      value = item.get("cost_analysis")
      if not isinstance(value, dict):
        continue
      flops = _positive_flops(value.get("flops"))
      if flops is not None:
        total += flops
    return total or None
  return None


def _expected_from_manual_model(
    result: dict[str, Any],
    expected_models: dict[str, Any] | None,
) -> tuple[float | None, dict[str, Any] | None]:
  if not expected_models:
    return None, None
  config = str(result.get("config", ""))
  manifest = result.get("manifest") or {}
  candidates = [
      config,
      str(manifest.get("kernel", "")),
      str(manifest.get("runner", "")),
  ]
  for key in candidates:
    if not key or key not in expected_models:
      continue
    model = expected_models[key]
    if isinstance(model, (int, float)):
      return _positive_flops(model), {"flops": model, "source": "manual"}
    if isinstance(model, dict):
      flops = _positive_flops(model.get("flops"))
      if flops is not None:
        return flops, model
  return None, None


def _validate_flops(
    result: dict[str, Any],
    expected_models: dict[str, Any] | None = None,
) -> dict[str, Any]:
  inspect = ((result.get("inspection") or {}).get("inspect") or {})
  rows = inspect.get("rows") or []
  _, _, xprof_flops = _summarize_rows(rows)
  manual_flops, manual_model = _expected_from_manual_model(result, expected_models)
  cost_flops = _expected_flops_from_manifest(result.get("manifest") or {})
  expected_flops = manual_flops if manual_flops is not None else cost_flops
  tolerance = 0.10
  if isinstance(manual_model, dict) and manual_model.get("tolerance") is not None:
    tolerance = float(manual_model["tolerance"])
  expected_source = "manual_model" if manual_flops is not None else (
      "cost_analysis" if cost_flops is not None else None
  )
  validation: dict[str, Any] = {
      "xprof_custom_call_flops": xprof_flops,
      "expected_flops": expected_flops,
      "expected_flops_source": expected_source,
      "expected_cost_analysis_flops": cost_flops,
      "expected_manual_flops": manual_flops,
      "expected_model": manual_model,
      "status": "no_expected_flops_model",
      "ratio_xprof_to_expected": None,
      "tolerance": tolerance,
  }
  if not xprof_flops:
    validation["status"] = "xprof_flops_missing"
    return validation
  if expected_flops is None:
    return validation
  ratio = float(xprof_flops) / float(expected_flops)
  validation["ratio_xprof_to_expected"] = ratio
  validation["status"] = (
      "xprof_flops_trusted"
      if (1.0 - tolerance) <= ratio <= (1.0 + tolerance)
      else "xprof_flops_mismatch"
  )
  return validation


def _trust_note(validation: dict[str, Any], mode: str | None) -> str:
  status = validation.get("status")
  ratio = validation.get("ratio_xprof_to_expected")
  source = validation.get("expected_flops_source")
  prefix = "trace-contract; " if mode == "trace-contract" else ""
  if status == "xprof_flops_trusted":
    return f"{prefix}XProf FLOPs match {source}"
  if status == "xprof_flops_mismatch":
    return (
        f"{prefix}XProf FLOPs mismatch {source}"
        + (f" (ratio {ratio:.3f})" if ratio is not None else "")
    )
  if status == "xprof_flops_missing":
    return f"{prefix}timing captured; XProf reported no custom-call FLOPs"
  return f"{prefix}timing captured; no independent expected FLOPs model"


def _enrich_result(
    result: dict[str, Any],
    expected_models: dict[str, Any] | None = None,
) -> None:
  status = result.get("status")
  if status in ("profiled", "profiled_inspect_failed"):
    result["flops_validation"] = _validate_flops(result, expected_models)
    result["trust_note"] = _trust_note(
        result["flops_validation"], (result.get("manifest") or {}).get("mode")
    )
    return
  if status == "failed":
    result.update(_classify_failure(result.get("error", "")))


def _audit_summary(
    results: list[dict[str, Any]],
    expected_models: dict[str, Any] | None = None,
) -> dict[str, Any]:
  status_counts: dict[str, int] = {}
  validation_counts: dict[str, int] = {}
  failure_counts: dict[str, int] = {}
  configs_by_status: dict[str, list[str]] = {}
  configs_by_validation: dict[str, list[str]] = {}
  configs_by_failure: dict[str, list[str]] = {}
  configs_by_expected_source: dict[str, list[str]] = {}
  for result in results:
    _enrich_result(result, expected_models)
    config = str(result.get("config", ""))
    status = str(result.get("status", "unknown"))
    status_counts[status] = status_counts.get(status, 0) + 1
    configs_by_status.setdefault(status, []).append(config)
    validation_status = (result.get("flops_validation") or {}).get("status")
    expected_source = (result.get("flops_validation") or {}).get("expected_flops_source")
    if validation_status:
      validation_counts[validation_status] = validation_counts.get(validation_status, 0) + 1
      configs_by_validation.setdefault(validation_status, []).append(config)
    if expected_source:
      configs_by_expected_source.setdefault(str(expected_source), []).append(config)
    failure_class = result.get("failure_class")
    if failure_class:
      failure_counts[failure_class] = failure_counts.get(failure_class, 0) + 1
      configs_by_failure.setdefault(failure_class, []).append(config)
  return {
      "total": len(results),
      "status_counts": status_counts,
      "flops_validation_counts": validation_counts,
      "failure_counts": failure_counts,
      "configs_by_status": configs_by_status,
      "configs_by_flops_validation": configs_by_validation,
      "configs_by_failure_class": configs_by_failure,
      "configs_by_expected_flops_source": configs_by_expected_source,
  }


def _gate_violations(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    expected_models: dict[str, Any] | None = None,
) -> list[str]:
  for result in results:
    _enrich_result(result, expected_models)
  allowed_failure_classes = set(getattr(args, "allow_failure_class", None) or [])
  violations = []
  if getattr(args, "require_all_profiled", False):
    bad = [
        result for result in results
        if result.get("status") not in ("profiled", "profiled_inspect_failed")
        and result.get("failure_class") not in allowed_failure_classes
    ]
    if bad:
      violations.append(
          "non-profiled configs not allowed: "
          + ", ".join(str(result.get("config")) for result in bad)
      )
  if getattr(args, "require_flops_trusted", False):
    bad = [
        result for result in results
        if result.get("status") in ("profiled", "profiled_inspect_failed")
        and (result.get("flops_validation") or {}).get("status") != "xprof_flops_trusted"
    ]
    if bad:
      violations.append(
          "profiled configs without trusted FLOPs: "
          + ", ".join(
              f"{result.get('config')}="
              f"{(result.get('flops_validation') or {}).get('status')}"
              for result in bad
          )
      )
  return violations


def _write_reports(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
  args.local_root.mkdir(parents=True, exist_ok=True)
  report_stem = getattr(args, "report_stem", None) or "batch_profile_report"
  expected_models = getattr(args, "expected_models_data", None)
  json_path = args.local_root / f"{report_stem}.json"
  md_path = args.local_root / f"{report_stem}.md"
  for result in results:
    _enrich_result(result, expected_models)
  json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

  summary = _audit_summary(results, expected_models)
  summary_path = args.local_root / f"{report_stem}_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

  lines = [
      "# Pallas XProf Batch Profile Report",
      "",
      f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
      f"Remote: `{getattr(args, 'host', None) or ''}`",
      f"Remote repo: `{getattr(args, 'remote_repo', None) or ''}`",
      f"Local root: `{args.local_root}`",
      "",
      f"Status counts: `{summary['status_counts']}`",
      f"FLOPs validation counts: `{summary['flops_validation_counts']}`",
      f"Failure counts: `{summary['failure_counts']}`",
      "",
      "| config | status | mode | local profile | custom rows | avg time us | XProf FLOPs | expected FLOPs | expected source | xprof/expected | validation | failure class | owner | trust/failure note |",
      "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |",
  ]
  for result in results:
    status = result.get("status", "unknown")
    config = result.get("config", "")
    local_profile = result.get("local_profile", "")
    mode = (result.get("manifest") or {}).get("mode", "public")
    inspect = ((result.get("inspection") or {}).get("inspect") or {})
    rows = inspect.get("rows") or []
    row_count, duration_us, raw_flops = _summarize_rows(rows)
    validation = result.get("flops_validation") or {}
    expected_flops = validation.get("expected_flops")
    expected_source = validation.get("expected_flops_source") or ""
    ratio = validation.get("ratio_xprof_to_expected")
    validation_status = validation.get("status", "")
    trust = result.get("trust_note") or "timing captured; FLOPs require CostEstimate/manual model validation"
    failure_class = result.get("failure_class", "")
    failure_owner = result.get("failure_owner", "")
    note = result.get("failure_note") or trust
    lines.append(
        "| "
        + " | ".join([
            config,
            status,
            mode,
            f"`{local_profile}`" if local_profile else "",
            str(row_count),
            "" if duration_us is None else f"{duration_us:.3f}",
            "" if raw_flops is None else str(raw_flops),
            "" if expected_flops is None else str(int(expected_flops)),
            expected_source,
            "" if ratio is None else f"{ratio:.3f}",
            validation_status,
            failure_class,
            failure_owner,
            note,
        ])
        + " |"
    )
  md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote {json_path}")
  print(f"wrote {md_path}")
  print(f"wrote {summary_path}")
  template_path = getattr(args, "write_expected_model_template", None)
  if template_path:
    _write_expected_model_template(template_path, results)


def _load_expected_models(path: Path | None) -> dict[str, Any] | None:
  if path is None:
    return None
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise SystemExit(f"{path} must contain a JSON object")
  if "kernels" in payload:
    kernels = payload["kernels"]
    if not isinstance(kernels, dict):
      raise SystemExit(f"{path}: 'kernels' must be a JSON object")
    return kernels
  return payload


def _write_expected_model_template(path: Path, results: list[dict[str, Any]]) -> None:
  """Write an editable expected-model file, preserving known manual models."""
  kernels: dict[str, Any] = {}
  for result in results:
    if result.get("status") not in ("profiled", "profiled_inspect_failed"):
      continue
    validation = result.get("flops_validation") or {}
    config = str(result.get("config", ""))
    if not config:
      continue
    if validation.get("expected_flops_source") == "manual_model":
      model = validation.get("expected_model")
      if isinstance(model, dict):
        entry = dict(model)
      else:
        entry = {
            "flops": validation.get("expected_manual_flops"),
            "source": "manual/analyze-kernel",
            "tolerance": validation.get("tolerance", 0.1),
        }
    else:
      entry = {
          "flops": None,
          "bytes": None,
          "source": "manual/analyze-kernel",
          "note": (
              "TODO: inspect kernel body and fill per custom-call occurrence "
              "useful FLOPs. Do not copy XProf or cost_analysis blindly."
          ),
          "tolerance": 0.1,
      }
    entry.update({
        "observed_validation_status": validation.get("status"),
        "observed_xprof_custom_call_flops": validation.get("xprof_custom_call_flops"),
        "observed_cost_analysis_flops": validation.get("expected_cost_analysis_flops"),
        "observed_expected_source": validation.get("expected_flops_source"),
        "local_profile": result.get("local_profile"),
        "mode": (result.get("manifest") or {}).get("mode"),
    })
    kernels[config] = entry
  payload = {
      "kernels": kernels,
      "notes": [
          "`flops` must be a manual/analyze-kernel value per custom-call occurrence.",
          "Existing manual models are preserved; TODO entries use null flops.",
          "Entries with null flops are ignored by pallas_xprof_batch.py validation.",
          "Use observed XProf/cost_analysis fields only as debugging clues.",
      ],
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
  print(f"wrote {path}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", help="SSH target, e.g. gcpuser@host")
  parser.add_argument("--identity-file")
  parser.add_argument("--remote-repo")
  parser.add_argument("--remote-python", default=".venv/bin/python")
  parser.add_argument("--remote-out-root", default="/tmp/xprof")
  parser.add_argument("--local-root", required=True, type=Path)
  parser.add_argument("--local-python", default=sys.executable)
  parser.add_argument(
      "--input-report",
      type=Path,
      help="Existing batch/consolidated JSON report to revalidate without remote execution.",
  )
  parser.add_argument(
      "--expected-models",
      type=Path,
      help=(
          "JSON object mapping config/kernel names to manual expected FLOPs. "
          "Each value may be a number or an object with flops, bytes, source, note, tolerance."
      ),
  )
  parser.add_argument(
      "--report-stem",
      default="batch_profile_report",
      help="Output report filename stem under --local-root.",
  )
  parser.add_argument(
      "--write-expected-model-template",
      type=Path,
      help=(
          "Write a JSON skeleton listing profiled configs that still need "
          "manual/analyze-kernel FLOPs models."
      ),
  )
  parser.add_argument(
      "--preflight-only",
      action="store_true",
      help="Run remote environment checks, write <report-stem>_preflight.json, and exit.",
  )
  parser.add_argument(
      "--skip-preflight",
      action="store_true",
      help="Skip the default remote environment preflight before online profiling.",
  )
  parser.add_argument(
      "--require-all-profiled",
      action="store_true",
      help="Exit nonzero if any config failed, except allowed failure classes.",
  )
  parser.add_argument(
      "--require-flops-trusted",
      action="store_true",
      help="Exit nonzero if any profiled config lacks xprof_flops_trusted.",
  )
  parser.add_argument(
      "--allow-failure-class",
      action="append",
      help="Failure class allowed by --require-all-profiled; repeatable.",
  )
  parser.add_argument("--config", action="append", help="Config to run; repeatable. Default: all.")
  parser.add_argument("--limit", type=int)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--remote-timeout", type=int, default=1800)
  parser.add_argument("--local-timeout", type=int, default=600)
  parser.add_argument("--skip-existing", action="store_true")
  parser.add_argument(
      "--retry-trace-contract",
      action="store_true",
      help="Retry failed public runner profiles using pallas_trace_args().",
  )
  args = parser.parse_args()

  args.local_root = args.local_root.resolve()
  args.local_root.mkdir(parents=True, exist_ok=True)
  args.expected_models_data = _load_expected_models(args.expected_models)

  if args.input_report:
    results = json.loads(args.input_report.read_text(encoding="utf-8"))
    if not isinstance(results, list):
      raise SystemExit(f"{args.input_report} must contain a JSON list")
    _write_reports(args, results)
    violations = _gate_violations(args, results, args.expected_models_data)
    if violations:
      raise SystemExit("audit gate failed:\n" + "\n".join(violations))
    return

  if not args.host or not args.remote_repo:
    raise SystemExit("--host and --remote-repo are required unless --input-report is used")

  if args.preflight_only or not args.skip_preflight:
    preflight = _preflight_remote(args)
    _write_preflight(args, preflight)
    if not preflight.get("ok"):
      raise SystemExit("remote preflight failed; inspect the written preflight JSON")
    if args.preflight_only:
      return

  _upload_runner(args)
  configs = _enumerate_configs(args)
  print(f"configs: {configs}")

  results = []
  for config in configs:
    result: dict[str, Any] = {"config": config, "status": "started"}
    try:
      print(f"profiling {config}", flush=True)
      try:
        manifest = _profile_one(args, config, "public")
      except Exception as public_exc:
        if not args.retry_trace_contract:
          raise
        print(f"public runner failed for {config}; retrying trace-contract", flush=True)
        manifest = _profile_one(args, config, "trace-contract")
        manifest["public_runner_error"] = repr(public_exc)
      result["manifest"] = manifest
      local_profile = _download_one(args, manifest)
      result["local_profile"] = str(local_profile)
      result["inspection"] = _inspect_one(args, local_profile, config)
      result["flops_validation"] = _validate_flops(result, args.expected_models_data)
      if result["inspection"]["inspect_returncode"] == 0:
        result["status"] = "profiled"
      else:
        result["status"] = "profiled_inspect_failed"
      result["trust_note"] = _trust_note(result["flops_validation"], manifest.get("mode"))
    except Exception as exc:  # Keep batch moving.
      result["status"] = "failed"
      result["error"] = repr(exc)
      result.update(_classify_failure(result["error"]))
    results.append(result)
    _write_reports(args, results)
  violations = _gate_violations(args, results, args.expected_models_data)
  if violations:
    raise SystemExit("audit gate failed:\n" + "\n".join(violations))


if __name__ == "__main__":
  main()

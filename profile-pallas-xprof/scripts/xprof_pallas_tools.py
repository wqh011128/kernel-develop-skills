#!/usr/bin/env python3
"""Utilities for validating and opening local XProf profiles for Pallas kernels.

Run with the same Python environment that has `xprof` installed.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request


def _import_xprof():
  try:
    from xprof.convert import raw_to_tool_data  # pylint: disable=import-outside-toplevel
    from xprof.protobuf import op_stats_pb2  # pylint: disable=import-outside-toplevel
  except ModuleNotFoundError as exc:
    raise SystemExit(
        "xprof is not importable. Re-run with the Python environment that owns "
        "xprof, for example .venv/Scripts/python.exe on Windows."
    ) from exc
  return raw_to_tool_data, op_stats_pb2


def _run_dir(profile_dir: Path) -> Path:
  profile_dir = profile_dir.resolve()
  if list(profile_dir.glob("*.xplane.pb")):
    return profile_dir
  plugin = profile_dir / "plugins" / "profile"
  if plugin.exists():
    runs = sorted([p for p in plugin.iterdir() if p.is_dir()])
    if len(runs) == 1:
      return runs[0]
    if runs:
      return runs[-1]
  matches = sorted(profile_dir.glob("**/*.xplane.pb"))
  if not matches:
    raise SystemExit(f"No .xplane.pb found under {profile_dir}")
  return matches[-1].parent


def _xplane(run_dir: Path) -> Path:
  matches = sorted(run_dir.glob("*.xplane.pb"))
  if not matches:
    raise SystemExit(f"No .xplane.pb found in {run_dir}")
  return matches[0]


def _op_stats_path(run_dir: Path) -> Path:
  return run_dir / "ALL_HOSTS.op_stats_v2.pb"


def _cache_version() -> str:
  try:
    return importlib.metadata.version("xprof")
  except importlib.metadata.PackageNotFoundError:
    return "unknown"


def write_cache_version(run_dir: Path) -> None:
  (run_dir / "cache_version.txt").write_text(_cache_version(), encoding="utf-8")


def generate_cache(profile_dir: Path, tools: list[str]) -> Path:
  raw_to_tool_data, _ = _import_xprof()
  run_dir = _run_dir(profile_dir)
  xplane = str(_xplane(run_dir))
  for tool in tools:
    options = {"use_saved_result": False}
    if tool == "op_profile":
      options["group_by"] = "program"
    raw_to_tool_data.xspace_to_tool_data([xplane], tool, options)
  write_cache_version(run_dir)
  if not _op_stats_path(run_dir).exists():
    print(
        "warning: XProf returned tool data but did not leave "
        f"{_op_stats_path(run_dir).name}; inspect will fall back to tool JSON."
    )
  return run_dir


def _load_op_stats(run_dir: Path):
  _, op_stats_pb2 = _import_xprof()
  path = _op_stats_path(run_dir)
  if not path.exists():
    raise SystemExit(f"Missing {path}; run generate-cache first.")
  stats = op_stats_pb2.OpStats()
  stats.ParseFromString(path.read_bytes())
  return path, stats


def _iter_metrics(stats):
  dbs = [
      ("device_op_metrics_db", stats.device_op_metrics_db),
      ("hlo_metrics_db_complete_steps_only", stats.hlo_metrics_db_complete_steps_only),
      ("host_op_metrics_db", stats.host_op_metrics_db),
  ]
  for db_name, db in dbs:
    for metric in db.metrics_db:
      yield db_name, metric


def _row_from_roofline(cols: list[str], row: dict) -> dict:
  values = [cell.get("v") if cell else None for cell in row.get("c", [])]
  return dict(zip(cols, values))


def _walk_op_profile(node: dict):
  yield node
  for child in node.get("children", []) or []:
    yield from _walk_op_profile(child)


def _inspect_tool_data(run_dir: Path, op_regex: str | None) -> dict:
  raw_to_tool_data, _ = _import_xprof()
  xplane = str(_xplane(run_dir))
  pattern = re.compile(op_regex) if op_regex else None
  rows = []

  roofline_raw, _ = raw_to_tool_data.xspace_to_tool_data(
      [xplane], "roofline_model", {"use_saved_result": False}
  )
  if roofline_raw is None:
    raise SystemExit(
        "XProf roofline_model conversion returned no data. The xplane may be "
        "corrupt or partially copied; re-download the profile atomically."
    )
  if isinstance(roofline_raw, bytes):
    roofline_raw = roofline_raw.decode("utf-8")
  roofline = json.loads(roofline_raw)
  if roofline:
    cols = [c["id"] for c in roofline[0].get("cols", [])]
    for row in roofline[0].get("rows", []):
      d = _row_from_roofline(cols, row)
      op_name = str(d.get("operation", ""))
      if pattern and not pattern.search(op_name):
        continue
      rows.append({
          "source": "roofline_model",
          "name": op_name,
          "category": d.get("category"),
          "occurrences": d.get("occurrences"),
          "avg_time_us": d.get("avg_time"),
          "model_flop_rate_gflops": d.get("model_flop_rate"),
          "measured_flop_rate_gflops": d.get("measured_flop_rate"),
          "compute_efficiency": d.get("compute_efficiency"),
          "operational_intensity": d.get("operational_intensity"),
          "bound_by": d.get("bound_by"),
      })

  op_profile_raw, _ = raw_to_tool_data.xspace_to_tool_data(
      [xplane],
      "op_profile",
      {"use_saved_result": False, "group_by": "program"},
  )
  if op_profile_raw is None:
    raise SystemExit(
        "XProf op_profile conversion returned no data. The xplane may be "
        "corrupt or partially copied; re-download the profile atomically."
    )
  if isinstance(op_profile_raw, bytes):
    op_profile_raw = op_profile_raw.decode("utf-8")
  op_profile = json.loads(op_profile_raw)
  roots = [
      op_profile.get("byProgram"),
      op_profile.get("byProgramExcludeIdle"),
  ]
  for root in roots:
    if not root:
      continue
    for node in _walk_op_profile(root):
      name = str(node.get("name", ""))
      if pattern and not pattern.search(name):
        continue
      metrics = node.get("metrics", {})
      rows.append({
          "source": "op_profile",
          "name": name,
          "category": (node.get("xla") or {}).get("category"),
          "occurrences": metrics.get("occurrences"),
          "raw_time_ps": metrics.get("rawTime"),
          "avg_time_ps": metrics.get("avgTimePs"),
          "raw_flops": metrics.get("rawFlops"),
          "bf16_flops": metrics.get("bf16Flops"),
          "flops_utilization": metrics.get("flops"),
      })
  return {
      "run_dir": str(run_dir),
      "op_stats": None,
      "inspection_source": "tool_data_json",
      "rows": rows,
  }


def inspect(profile_dir: Path, op_regex: str | None, output_json: bool) -> None:
  run_dir = _run_dir(profile_dir)
  op_stats_file = _op_stats_path(run_dir)
  if not op_stats_file.exists():
    payload = _inspect_tool_data(run_dir, op_regex)
    if output_json:
      print(json.dumps(payload, indent=2, sort_keys=True))
      return
    print(f"run_dir: {run_dir}")
    print("op_stats: missing; using XProf tool JSON fallback")
    for row in payload["rows"]:
      print(json.dumps(row, sort_keys=True))
    return

  path, stats = _load_op_stats(run_dir)
  pattern = re.compile(op_regex) if op_regex else None
  rows = []
  for db_name, metric in _iter_metrics(stats):
    if pattern and not pattern.search(metric.name):
      continue
    if not pattern and not metric.name:
      continue
    rows.append({
        "db": db_name,
        "name": metric.name,
        "category": metric.category,
        "occurrences": metric.occurrences,
        "time_ps": metric.time_ps,
        "avg_time_us": metric.time_ps / max(metric.occurrences, 1) / 1e6,
        "flops": metric.flops,
        "model_flops": metric.model_flops,
        "flops_v2": metric.flops_v2,
        "model_flops_v2": metric.model_flops_v2,
        "bytes_accessed": metric.bytes_accessed,
    })
  payload = {
      "run_dir": str(run_dir),
      "op_stats": str(path),
      "inspection_source": "op_stats_v2_pb",
      "rows": rows,
  }
  if output_json:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return
  print(f"run_dir: {run_dir}")
  print(f"op_stats: {path}")
  for row in rows:
    print(
        f"{row['db']} | {row['name']} | occ={row['occurrences']} | "
        f"avg_us={row['avg_time_us']:.3f} | model_flops={row['model_flops']} | "
        f"flops_v2={row['flops_v2']:.0f} | bytes={row['bytes_accessed']}"
    )


def patch_op_stats(
    profile_dir: Path,
    op_regex: str,
    flops_per_occurrence: int,
    bytes_per_occurrence: int | None,
) -> Path:
  run_dir = _run_dir(profile_dir)
  path, stats = _load_op_stats(run_dir)
  backup = path.with_suffix(path.suffix + ".orig")
  if not backup.exists():
    shutil.copy2(path, backup)
  pattern = re.compile(op_regex)
  patched = 0
  for _, metric in _iter_metrics(stats):
    if not pattern.search(metric.name):
      continue
    occurrences = max(metric.occurrences, 1)
    total_flops = int(flops_per_occurrence) * occurrences
    metric.flops = total_flops
    metric.model_flops = total_flops
    metric.flops_v2 = float(total_flops)
    metric.model_flops_v2 = float(total_flops)
    if bytes_per_occurrence is not None:
      metric.bytes_accessed = int(bytes_per_occurrence) * occurrences
    patched += 1
  if patched == 0:
    raise SystemExit(f"No op matched regex {op_regex!r} in {path}")
  path.write_bytes(stats.SerializeToString())
  write_cache_version(run_dir)
  print(f"patched {patched} metric entries in {path}")
  return run_dir


def patch_trace_json(
    profile_dir: Path,
    op_regex: str,
    flops_per_occurrence: int,
    bytes_per_occurrence: int | None,
) -> None:
  run_dir = _run_dir(profile_dir)
  pattern = re.compile(op_regex)
  trace_files = sorted(run_dir.glob("*.trace.json.gz"))
  if not trace_files:
    print(f"no *.trace.json.gz found in {run_dir}; skipping trace patch")
    return
  for path in trace_files:
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
      shutil.copy2(path, backup)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
      data = json.load(f)
    patched = 0
    for event in data.get("traceEvents", []):
      if not pattern.search(event.get("name", "")):
        continue
      args = event.setdefault("args", {})
      args["model_flops"] = str(flops_per_occurrence)
      if bytes_per_occurrence is not None:
        args["bytes_accessed"] = str(bytes_per_occurrence)
        args["raw_bytes_accessed"] = str(bytes_per_occurrence)
      patched += 1
    if patched:
      with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
      print(f"patched {patched} trace events in {path}")


def patch_all(
    profile_dir: Path,
    op_regex: str,
    flops_per_occurrence: int,
    bytes_per_occurrence: int | None,
) -> None:
  patch_op_stats(profile_dir, op_regex, flops_per_occurrence, bytes_per_occurrence)
  patch_trace_json(profile_dir, op_regex, flops_per_occurrence, bytes_per_occurrence)


def start_xprof(profile_dir: Path, port: int, xprof_exe: str | None) -> None:
  profile_dir = profile_dir.resolve()
  exe = xprof_exe or shutil.which("xprof")
  if exe is None:
    raise SystemExit("xprof executable not found; pass --xprof-exe")
  log = profile_dir.parent / f"xprof_server_{port}.log"
  err = profile_dir.parent / f"xprof_server_{port}.err.log"
  cmd = [exe, "--logdir", str(profile_dir), "--port", str(port)]
  with open(log, "ab", buffering=0) as stdout, open(err, "ab", buffering=0) as stderr:
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
  print(f"started xprof pid={proc.pid} url=http://127.0.0.1:{port}/")
  print(f"stdout={log}")
  print(f"stderr={err}")


def api_check(port: int, run: str | None, host: str | None) -> None:
  base = f"http://127.0.0.1:{port}"
  with urllib.request.urlopen(f"{base}/data/plugin/profile/runs", timeout=10) as r:
    runs = json.loads(r.read().decode("utf-8"))
  print(f"runs: {runs}")
  if run is None:
    run = runs[-1] if runs else None
  if not run or not host:
    return
  url = (
      f"{base}/data/plugin/profile/data?run={run}&tag=roofline_model"
      f"&host={host}&hosts={host}"
  )
  with urllib.request.urlopen(url, timeout=20) as r:
    data = json.loads(r.read().decode("utf-8"))
  cols = [c["id"] for c in data[0]["cols"]]
  for row in data[0]["rows"]:
    values = [cell.get("v") if cell else None for cell in row["c"]]
    d = dict(zip(cols, values))
    if d.get("category") == "custom-call":
      print(
          f"{d.get('operation')}: occ={d.get('occurrences')} "
          f"avg_us={d.get('avg_time')} "
          f"gflops={d.get('model_flop_rate')} "
          f"peak_frac={d.get('compute_efficiency')}"
      )


def _api_runs(port: int) -> list[str]:
  base = f"http://127.0.0.1:{port}"
  with urllib.request.urlopen(f"{base}/data/plugin/profile/runs", timeout=10) as r:
    data = r.read()
  if data.startswith(b"\x1f\x8b"):
    data = gzip.decompress(data)
  runs = json.loads(data.decode("utf-8"))
  if not isinstance(runs, list):
    raise SystemExit(f"Unexpected runs payload from XProf: {runs!r}")
  return [str(run) for run in runs]


def _profile_run_names(profile_root: Path) -> list[str]:
  profile_root = profile_root.resolve()
  names = []
  for xplane in sorted(profile_root.glob("**/*.xplane.pb")):
    try:
      rel = xplane.parent.relative_to(profile_root)
    except ValueError:
      continue
    names.append(str(rel).replace(os.sep, "/"))
  return names


def _load_summary(path: Path | None) -> dict | None:
  if path is None:
    return None
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise SystemExit(f"{path} must contain a JSON object")
  return payload


def readiness(
    profile_root: Path,
    port: int,
    summary_path: Path | None,
    output_json: bool,
) -> None:
  profile_root = profile_root.resolve()
  run_names = _profile_run_names(profile_root)
  try:
    api_runs = _api_runs(port)
    server_ok = True
    server_error = None
  except Exception as exc:  # Report actionable readiness instead of traceback.
    api_runs = []
    server_ok = False
    server_error = repr(exc)

  normalized_api = {run.replace("\\", "/") for run in api_runs}
  visible_runs = [run for run in run_names if run in normalized_api]
  summary = _load_summary(summary_path)

  payload = {
      "profile_root": str(profile_root),
      "port": port,
      "server_ok": server_ok,
      "server_error": server_error,
      "xplane_run_count": len(run_names),
      "xprof_api_run_count": len(api_runs),
      "visible_profile_run_count": len(visible_runs),
      "all_profile_runs_visible": bool(run_names) and len(visible_runs) == len(run_names),
      "missing_profile_runs": [run for run in run_names if run not in normalized_api],
      "sample_api_runs": api_runs[:10],
      "sample_profile_runs": run_names[:10],
      "summary": summary,
  }
  if summary:
    validation_counts = summary.get("flops_validation_counts") or {}
    status_counts = summary.get("status_counts") or {}
    payload["profiled_count"] = status_counts.get("profiled", 0)
    payload["failed_count"] = status_counts.get("failed", 0)
    payload["trusted_flops_count"] = validation_counts.get("xprof_flops_trusted", 0)
    payload["untrusted_profiled_count"] = (
        payload["profiled_count"] - payload["trusted_flops_count"]
    )
    payload["production_ready"] = (
        payload["all_profile_runs_visible"]
        and payload["failed_count"] == 0
        and payload["untrusted_profiled_count"] == 0
    )
  else:
    payload["production_ready"] = payload["all_profile_runs_visible"]

  if output_json:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return
  print(f"profile_root: {profile_root}")
  print(f"xprof server ok: {server_ok}")
  if server_error:
    print(f"xprof server error: {server_error}")
  print(f"xplane runs: {len(run_names)}")
  print(f"xprof api runs: {len(api_runs)}")
  print(f"visible profile runs: {len(visible_runs)}")
  print(f"all profile runs visible: {payload['all_profile_runs_visible']}")
  print(f"production_ready: {payload['production_ready']}")
  if payload["missing_profile_runs"]:
    print("missing profile runs:")
    for run in payload["missing_profile_runs"]:
      print(f"  {run}")


def main() -> None:
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest="cmd", required=True)

  gen = sub.add_parser("generate-cache")
  gen.add_argument("--profile-dir", required=True, type=Path)
  gen.add_argument(
      "--tools",
      default="roofline_model,op_profile,overview_page,trace_viewer",
      help="Comma-separated XProf tools to precompute.",
  )

  ins = sub.add_parser("inspect")
  ins.add_argument("--profile-dir", required=True, type=Path)
  ins.add_argument("--op-regex")
  ins.add_argument("--json", action="store_true")

  pat = sub.add_parser("patch")
  pat.add_argument("--profile-dir", required=True, type=Path)
  pat.add_argument("--op-regex", required=True)
  pat.add_argument("--flops-per-occurrence", required=True, type=int)
  pat.add_argument("--bytes-per-occurrence", type=int)

  start = sub.add_parser("start-xprof")
  start.add_argument("--profile-dir", required=True, type=Path)
  start.add_argument("--port", type=int, default=6006)
  start.add_argument("--xprof-exe")

  check = sub.add_parser("api-check")
  check.add_argument("--port", type=int, default=6006)
  check.add_argument("--run")
  check.add_argument("--host")

  ready = sub.add_parser("readiness")
  ready.add_argument("--profile-root", required=True, type=Path)
  ready.add_argument("--port", type=int, default=6006)
  ready.add_argument("--summary", type=Path)
  ready.add_argument("--json", action="store_true")

  args = parser.parse_args()
  if args.cmd == "generate-cache":
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    run_dir = generate_cache(args.profile_dir, tools)
    print(run_dir)
  elif args.cmd == "inspect":
    inspect(args.profile_dir, args.op_regex, args.json)
  elif args.cmd == "patch":
    patch_all(
        args.profile_dir,
        args.op_regex,
        args.flops_per_occurrence,
        args.bytes_per_occurrence,
    )
  elif args.cmd == "start-xprof":
    start_xprof(args.profile_dir, args.port, args.xprof_exe)
  elif args.cmd == "api-check":
    api_check(args.port, args.run, args.host)
  elif args.cmd == "readiness":
    readiness(args.profile_root, args.port, args.summary, args.json)


if __name__ == "__main__":
  main()

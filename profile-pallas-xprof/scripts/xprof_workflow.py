#!/usr/bin/env python3
"""Run the standard remote-capture to local-XProf workflow.

This wrapper intentionally delegates kernel execution to the registry batch
runner and local profile handling to xprof_pallas_tools.py. It owns artifact
placement, status reporting, and UI readiness so agents do not re-create the
fragile SSH/tar/scp/startup sequence for every kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
BATCH = SCRIPT_DIR / "pallas_xprof_batch.py"
TOOLS = SCRIPT_DIR / "xprof_pallas_tools.py"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
  print("+", subprocess.list2cmdline(cmd), flush=True)
  return subprocess.run(cmd, text=True, capture_output=True, check=check)


def _discover_xprof(explicit: str | None, local_python: Path) -> str | None:
  candidates = [
    explicit,
    shutil.which("xprof"),
    shutil.which("xprof.exe"),
    str(local_python.parent / "xprof.exe"),
    str(local_python.parent / "xprof"),
  ]
  for candidate in candidates:
    if candidate and Path(candidate).is_file():
      return str(Path(candidate).resolve())
  return None


def _write_status(root: Path, payload: dict) -> None:
  (root / "xprof_workflow_status.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  lines = [
    "# XProf 工作流状态",
    "",
    f"- 状态：`{payload['status']}`",
    f"- 本地目录：`{root}`",
    f"- XProf URL：`{payload.get('url') or '未启动'}`",
    f"- XPlane 数量：`{payload.get('xplane_count', 0)}`",
    f"- Trace 数量：`{payload.get('trace_count', 0)}`",
  ]
  if payload.get("error"):
    lines.extend(["", "## 错误", "", "```text", payload["error"], "```"])
  (root / "xprof_ui_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", required=True, help="SSH alias or user@host")
  parser.add_argument("--identity-file")
  parser.add_argument("--remote-repo", required=True)
  parser.add_argument("--remote-python", default=".venv/bin/python")
  parser.add_argument("--workspace-root", required=True, type=Path)
  parser.add_argument("--method", required=True)
  parser.add_argument("--config", action="append", required=True)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--port", type=int, default=6010)
  parser.add_argument("--local-python", type=Path, default=Path(sys.executable))
  parser.add_argument("--xprof-exe")
  parser.add_argument("--preflight-only", action="store_true")
  parser.add_argument("--skip-ui", action="store_true")
  parser.add_argument("--require-all-profiled", action="store_true")
  args = parser.parse_args()

  workspace_root = args.workspace_root.resolve()
  local_root = (
    workspace_root / "experiments" / args.method / "results" / "xprof"
  )
  local_root.mkdir(parents=True, exist_ok=True)
  if not (workspace_root / "docs").is_dir():
    raise SystemExit(f"Missing standard docs directory: {workspace_root / 'docs'}")

  batch_cmd = [
    str(args.local_python),
    str(BATCH),
    "--host",
    args.host,
    "--remote-repo",
    args.remote_repo,
    "--remote-python",
    args.remote_python,
    "--local-root",
    str(local_root),
    "--local-python",
    str(args.local_python),
    "--warmup",
    str(args.warmup),
  ]
  if args.identity_file:
    batch_cmd.extend(["--identity-file", args.identity_file])
  for config in args.config:
    batch_cmd.extend(["--config", config])
  if args.preflight_only:
    batch_cmd.append("--preflight-only")
  if args.require_all_profiled:
    batch_cmd.append("--require-all-profiled")

  payload = {
    "status": "started",
    "started_at": int(time.time()),
    "host": args.host,
    "remote_repo": args.remote_repo,
    "configs": args.config,
    "local_root": str(local_root),
    "port": args.port,
  }
  try:
    proc = _run(batch_cmd, check=False)
    payload["batch_returncode"] = proc.returncode
    payload["batch_stdout_tail"] = proc.stdout[-4000:]
    payload["batch_stderr_tail"] = proc.stderr[-4000:]
    if proc.returncode:
      raise RuntimeError(proc.stderr[-4000:] or proc.stdout[-4000:])
    if args.preflight_only:
      payload["status"] = "preflight_passed"
      return

    xplanes = sorted(local_root.glob("**/*.xplane.pb"))
    traces = sorted(local_root.glob("**/*.trace.json.gz"))
    payload["xplane_count"] = len(xplanes)
    payload["trace_count"] = len(traces)
    if not xplanes:
      raise RuntimeError("Profile workflow completed without a local *.xplane.pb")

    cache = _run(
      [
        str(args.local_python),
        str(TOOLS),
        "generate-cache",
        "--profile-dir",
        str(local_root),
      ],
      check=False,
    )
    payload["cache_returncode"] = cache.returncode
    payload["cache_stderr_tail"] = cache.stderr[-4000:]

    if args.skip_ui:
      payload["status"] = "profile_downloaded_ui_skipped"
      return

    xprof_exe = _discover_xprof(args.xprof_exe, args.local_python)
    payload["xprof_exe"] = xprof_exe
    if xprof_exe is None:
      raise RuntimeError("No local xprof executable found")
    started = _run(
      [
        str(args.local_python),
        str(TOOLS),
        "start-xprof",
        "--profile-dir",
        str(local_root),
        "--port",
        str(args.port),
        "--xprof-exe",
        xprof_exe,
      ]
    )
    payload["start_output"] = started.stdout.strip()
    payload["url"] = f"http://127.0.0.1:{args.port}/"
    time.sleep(2)
    ready = _run(
      [
        str(args.local_python),
        str(TOOLS),
        "readiness",
        "--profile-root",
        str(local_root),
        "--port",
        str(args.port),
        "--json",
      ],
      check=False,
    )
    payload["readiness_returncode"] = ready.returncode
    payload["readiness"] = ready.stdout[-8000:]
    try:
      readiness = json.loads(ready.stdout)
    except json.JSONDecodeError:
      readiness = {}
    payload["readiness_details"] = readiness
    ui_ready = (
      ready.returncode == 0
      and readiness.get("server_ok") is True
      and readiness.get("all_profile_runs_visible") is True
    )
    payload["status"] = "profile_opened" if ui_ready else "ui_not_ready"
  except Exception as exc:
    payload["status"] = "failed"
    payload["error"] = repr(exc)
    raise
  finally:
    payload["finished_at"] = int(time.time())
    _write_status(local_root, payload)


if __name__ == "__main__":
  main()

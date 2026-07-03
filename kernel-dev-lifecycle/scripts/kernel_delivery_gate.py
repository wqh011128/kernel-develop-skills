#!/usr/bin/env python3
"""Audit repository contracts and run repeatable kernel delivery checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


GENERATED_PATTERNS = ("*.xplane.pb", "*.trace.json.gz", "*_before_opt.hlo", "*.tgz")


def _run(cmd: list[str], cwd: Path) -> dict:
  proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
  return {
    "command": subprocess.list2cmdline(cmd),
    "returncode": proc.returncode,
    "stdout_tail": proc.stdout[-4000:],
    "stderr_tail": proc.stderr[-4000:],
  }


def _git(repo: Path, *args: str) -> str:
  return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _find_executable(repo: Path, names: tuple[str, ...]) -> str | None:
  candidates = []
  for name in names:
    candidates.extend(
      [repo / ".venv" / "bin" / name, repo / ".venv" / "Scripts" / f"{name}.exe"]
    )
    found = shutil.which(name)
    if found:
      candidates.append(Path(found))
  for candidate in candidates:
    if candidate.is_file():
      return str(candidate)
  return None


def _applicable_agents(repo: Path, changed: list[str]) -> list[str]:
  agents = sorted(repo.rglob("AGENTS.md"))
  applicable = []
  changed_paths = [(repo / path).resolve() for path in changed]
  for path in agents:
    parent = path.parent.resolve()
    if parent == repo or any(parent in item.parents or parent == item.parent for item in changed_paths):
      applicable.append(str(path.relative_to(repo)))
  return applicable


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--kernel", required=True)
  parser.add_argument("--config")
  parser.add_argument("--test")
  parser.add_argument("--package", default="kernels")
  parser.add_argument("--device-num", type=int, default=1)
  parser.add_argument("--snapshot-root", type=Path)
  parser.add_argument("--pr-text", type=Path, help="PR body or commit message to audit")
  parser.add_argument("--allow-missing-expected", action="store_true")
  parser.add_argument("--run", action="store_true")
  parser.add_argument("--json-out", type=Path)
  args = parser.parse_args()

  repo = args.repo.resolve()
  if not (repo / ".git").exists():
    raise SystemExit(f"Not a git repository: {repo}")
  config = args.config or args.kernel
  test = args.test or f"test_{args.kernel}_correctness"
  changed = [line for line in _git(repo, "status", "--porcelain").splitlines() if line]
  changed_files = [line[3:] for line in changed]
  branch = _git(repo, "branch", "--show-current")

  expected = [
    f"pallas_kernels/{args.package}/{args.kernel}.py",
    f"pallas_kernels/configs/{config}.yaml",
    f"tests/{args.package}/test_{args.kernel}.py",
    f"jax_ops/{args.kernel}.py",
    f"docs/{args.package}/{args.kernel}.md",
  ]
  expected_status = {path: (repo / path).is_file() for path in expected}
  tracked = set(_git(repo, "ls-files").splitlines())
  generated = []
  for pattern in GENERATED_PATTERNS:
    generated.extend(
      str(path.relative_to(repo))
      for path in repo.glob(f"**/{pattern}")
      if str(path.relative_to(repo)).replace("\\", "/") not in tracked
    )

  ir_tag = (
    f"[ir-upload package={args.package} kernel={args.kernel} config={config} "
    f"test={test} device_num={args.device_num}]"
  )
  payload = {
    "repo": str(repo),
    "branch": branch,
    "git_status": changed,
    "changed_files": changed_files,
    "agents": _applicable_agents(repo, changed_files),
    "expected_files": expected_status,
    "generated_artifacts_in_repo": sorted(set(generated)),
    "ir_upload_tag": ir_tag,
    "checks": [],
  }
  if args.pr_text:
    pr_text = args.pr_text.read_text(encoding="utf-8")
    payload["ir_upload_tag_present"] = ir_tag in pr_text

  if args.snapshot_root:
    root = args.snapshot_root.resolve()
    payload["snapshot"] = {
      "root": str(root),
      "before_opt_count": len(list(root.glob("**/*_before_opt.hlo"))),
      "error_count": len(list(root.glob("**/*_error.log"))),
    }

  if args.run:
    payload["checks"].append(_run(["git", "diff", "--check"], repo))
    python_files = [path for path in changed_files if path.endswith(".py")]
    pre_commit = _find_executable(repo, ("pre-commit",))
    if pre_commit and changed_files:
      payload["checks"].append(
        _run([pre_commit, "run", "--files", *changed_files], repo)
      )
    ruff = _find_executable(repo, ("ruff",))
    if ruff and python_files:
      cmd = [ruff, "check"]
      ruff_config = repo / ".github" / "workflows" / "ruff.toml"
      if ruff_config.is_file():
        cmd.extend(["--config", str(ruff_config)])
      payload["checks"].append(_run([*cmd, *python_files], repo))
    python = _find_executable(repo, ("python", "python3"))
    typing_helper = repo / ".ci-shared" / "scripts" / "typing_helper.py"
    mypy = _find_executable(repo, ("mypy",))
    if python_files and python and typing_helper.is_file():
      payload["checks"].append(
        _run(
          [python, str(typing_helper), "--changed-files", ",".join(python_files)],
          repo,
        )
      )
    elif python_files and mypy:
      payload["checks"].append(
        _run([mypy, "--ignore-missing-imports", *python_files], repo)
      )
    validator = repo / "tools" / "config_validator.py"
    if python and validator.is_file():
      payload["checks"].append(_run([python, str(validator)], repo))

  failures = [item for item in payload["checks"] if item["returncode"]]
  snapshot = payload.get("snapshot")
  blockers = []
  warnings = []
  if not payload["agents"]:
    warnings.append("No applicable AGENTS.md was found; use repository conventions")
  missing_expected = [path for path, exists in expected_status.items() if not exists]
  if missing_expected and not args.allow_missing_expected:
    blockers.append(f"Missing expected kernel files: {missing_expected}")
  if payload["generated_artifacts_in_repo"]:
    blockers.append("Generated profile/IR artifacts exist inside the git repository")
  if snapshot and (snapshot["before_opt_count"] == 0 or snapshot["error_count"]):
    blockers.append("Snapshot artifacts are missing or contain error logs")
  if failures:
    blockers.append("One or more delivery commands failed")
  if args.pr_text and not payload["ir_upload_tag_present"]:
    blockers.append("Required IR-upload tag is missing from PR/commit text")
  payload["warnings"] = warnings
  payload["blockers"] = blockers
  payload["status"] = "pass" if not blockers else "fail"

  rendered = json.dumps(payload, indent=2, ensure_ascii=False)
  print(rendered)
  if args.json_out:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(rendered + "\n", encoding="utf-8")
  raise SystemExit(0 if not blockers else 1)


if __name__ == "__main__":
  main()

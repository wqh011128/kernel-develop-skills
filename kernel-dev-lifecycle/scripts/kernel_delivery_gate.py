#!/usr/bin/env python3
"""Audit repository contracts and run repeatable kernel delivery checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess


GENERATED_PATTERNS = ("*.xplane.pb", "*.trace.json.gz", "*_before_opt.hlo", "*.tgz")
COMMIT_TEMPLATE = Path(__file__).resolve().parents[1] / "assets" / "commit_message_template.txt"
COMMIT_TITLE_RE = re.compile(
  r"^(feat|fix|perf|refactor|test|docs|build|ci|chore)"
  r"\[[A-Z][A-Z0-9_-]*\]: [^\s].+$"
)
JIRA_RE = re.compile(r"^[A-Z][A-Z0-9]+-(?:\d+|XXXX)$")
IR_UPLOAD_TAG_SCOPE = (
  "One tag covers one runnable upload matrix item "
  "(package, registered kernel, config, test module, device count). "
  "Internal Pallas/custom-call/HLO phases covered by that item do not need separate tags."
)
STRUCTURAL_HLO_OPS = frozenset(
  {
    "after-all",
    "get-tuple-element",
    "parameter",
    "tuple",
  }
)
HLO_OPCODE_RE = re.compile(r"\b([a-z][a-z0-9-]*)\(")
CUSTOM_CALL_TARGET_RE = re.compile(r'custom_call_target="([^"]+)"')


def _run(cmd: list[str], cwd: Path, *, name: str | None = None) -> dict:
  proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
  return {
    "name": name or Path(cmd[0]).name,
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


def _tool_command(
  repo: Path,
  executable: str,
  module: str,
  python: str | None,
) -> list[str] | None:
  found = _find_executable(repo, (executable,))
  if found:
    return [found]
  if python:
    return [python, "-m", module]
  return None


def _missing_check(name: str, reason: str) -> dict:
  return {
    "name": name,
    "command": None,
    "returncode": 127,
    "stdout_tail": "",
    "stderr_tail": reason,
  }


def _audit_hlo_file(path: Path, *, allowed_extra_ops: set[str]) -> dict:
  text = path.read_text(encoding="utf-8", errors="replace")
  opcode_counts: dict[str, int] = {}
  for line in text.splitlines():
    if " = " not in line:
      continue
    rhs = line.split(" = ", 1)[1]
    match = HLO_OPCODE_RE.search(rhs)
    if not match:
      continue
    opcode = match.group(1)
    opcode_counts[opcode] = opcode_counts.get(opcode, 0) + 1

  custom_call_count = opcode_counts.pop("custom-call", 0)
  targets = sorted(CUSTOM_CALL_TARGET_RE.findall(text))
  unexpected = {
    opcode: count
    for opcode, count in sorted(opcode_counts.items())
    if opcode not in STRUCTURAL_HLO_OPS and opcode not in allowed_extra_ops
  }
  return {
    "path": str(path.resolve()),
    "custom_call_count": custom_call_count,
    "custom_call_targets": targets,
    "non_custom_opcode_counts": dict(sorted(opcode_counts.items())),
    "unexpected_non_custom_opcode_counts": unexpected,
  }


def _audit_hlo_root(
  root: Path | None, *, allowed_extra_ops: set[str]
) -> dict | None:
  if root is None:
    return None
  resolved = root.resolve()
  files = sorted(resolved.glob("**/*_before_opt.hlo")) if resolved.exists() else []
  return {
    "root": str(resolved),
    "files": [
      _audit_hlo_file(path, allowed_extra_ops=allowed_extra_ops) for path in files
    ],
  }


def _compare_hlo_audits(tpu: dict | None, cpu: dict | None) -> list[dict]:
  if not tpu or not cpu:
    return []
  tpu_by_name = {Path(item["path"]).name: item for item in tpu["files"]}
  cpu_by_name = {Path(item["path"]).name: item for item in cpu["files"]}
  comparisons = []
  for name in sorted(set(tpu_by_name) & set(cpu_by_name)):
    tpu_item = tpu_by_name[name]
    cpu_item = cpu_by_name[name]
    count_match = tpu_item["custom_call_count"] == cpu_item["custom_call_count"]
    targets_match = tpu_item["custom_call_targets"] == cpu_item["custom_call_targets"]
    comparisons.append(
      {
        "filename": name,
        "custom_call_count_match": count_match,
        "custom_call_targets_match": targets_match,
        "status": "pass" if count_match and targets_match else "fail",
      }
    )
  return comparisons


def _workflow_run_commands(path: str, text: str) -> list[dict[str, str]]:
  commands = []
  lines = text.splitlines()
  index = 0
  while index < len(lines):
    line = lines[index]
    match = re.match(r"^(?P<indent>\s*)(?:-\s*)?run:\s*(?P<value>.*)$", line)
    if not match:
      index += 1
      continue
    value = match.group("value").strip()
    if value.startswith(("|", ">")):
      base_indent = len(match.group("indent"))
      block = []
      index += 1
      while index < len(lines):
        candidate = lines[index]
        if not candidate.strip():
          block.append("")
          index += 1
          continue
        indent = len(candidate) - len(candidate.lstrip())
        if indent <= base_indent:
          break
        block.append(candidate.strip())
        index += 1
      command = "\n".join(block).strip()
    else:
      command = value
      index += 1
    commands.append({"workflow": path, "command": command})
  return commands


def _discover_ci(repo: Path) -> dict:
  workflow_root = repo / ".github" / "workflows"
  workflow_files = sorted(
    [*workflow_root.glob("*.yml"), *workflow_root.glob("*.yaml")]
  ) if workflow_root.is_dir() else []
  sources = list(workflow_files)
  for path in (repo / ".pre-commit-config.yaml", repo / "pyproject.toml"):
    if path.is_file():
      sources.append(path)
  workflow_text = {
    path.relative_to(repo).as_posix(): path.read_text(
      encoding="utf-8", errors="replace"
    )
    for path in workflow_files
  }
  source_text = "\n".join(
    path.read_text(encoding="utf-8", errors="replace").lower() for path in sources
  )
  run_commands = []
  for path, text in workflow_text.items():
    run_commands.extend(_workflow_run_commands(path, text))
  surfaces = {
    "pre_commit": (repo / ".pre-commit-config.yaml").is_file()
    or "pre-commit" in source_text,
    "ruff_check": "ruff" in source_text,
    "ruff_format": "ruff format" in source_text or "ruff-format" in source_text,
    "typing": any(token in source_text for token in ("mypy", "typing_helper", "pyright")),
    "unit_tests": "pytest" in source_text,
    "config_validator": (repo / "tools" / "config_validator.py").is_file(),
  }
  return {
    "workflow_files": [path.relative_to(repo).as_posix() for path in workflow_files],
    "workflow_run_commands": run_commands,
    "surfaces": surfaces,
    "note": (
      "Inventoried every workflow file and run: entry, then detected common local CI "
      "surfaces. AGENTS.md and workflow-specific commands remain authoritative; each "
      "inventory item must be reconciled in the delivery ledger."
    ),
  }


def _validate_commit_message(text: str) -> dict:
  normalized = text.replace("\r\n", "\n").strip()
  lines = normalized.splitlines()
  errors = []
  warnings = []
  if not lines or not COMMIT_TITLE_RE.fullmatch(lines[0]):
    errors.append(
      "Title must match type[SCOPE]: summary with a supported lowercase type "
      "and uppercase scope"
    )

  positions = {}
  for label in ("Task:", "Solution:", "Test:"):
    matches = [index for index, line in enumerate(lines) if line == label]
    if len(matches) != 1:
      errors.append(f"Expected exactly one {label} section")
    elif matches:
      positions[label] = matches[0]
  jira_lines = [
    (index, line.removeprefix("JIRA:").strip())
    for index, line in enumerate(lines)
    if line.startswith("JIRA:")
  ]
  if len(jira_lines) != 1:
    errors.append("Expected exactly one JIRA: <PROJECT-NUMBER> line")
  else:
    _, jira = jira_lines[0]
    if not JIRA_RE.fullmatch(jira):
      errors.append("JIRA must look like COMPIL-123 or the draft placeholder COMPIL-XXXX")
    elif jira.endswith("-XXXX"):
      warnings.append("Replace the virtual JIRA placeholder COMPIL-XXXX before commit")

  if len(positions) == 3:
    ordered = [positions[label] for label in ("Task:", "Solution:", "Test:")]
    if ordered != sorted(ordered):
      errors.append("Sections must be ordered Task, Solution, Test")
    jira_index = jira_lines[0][0] if len(jira_lines) == 1 else len(lines)
    bounds = [ordered[1], ordered[2], jira_index]
    for label, start, end in zip(("Task:", "Solution:", "Test:"), ordered, bounds):
      bullets = [line for line in lines[start + 1:end] if line.startswith("- ")]
      if not bullets or any(len(line[2:].strip()) == 0 for line in bullets):
        errors.append(f"{label} must contain at least one non-empty bullet")

  body_without_jira = "\n".join(
    line for line in lines if not line.startswith("JIRA:")
  )
  if re.search(r"<[^>]+>", body_without_jira):
    errors.append("Commit message still contains an unresolved template placeholder")
  return {
    "status": "pass" if not errors else "fail",
    "errors": errors,
    "warnings": warnings,
  }


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
  parser.add_argument(
    "--tpu-test-file",
    help="Focused TPU test file used by scripts/test_all.py (required with --run).",
  )
  parser.add_argument(
    "--tpu-test-case",
    default="correctness",
    help="Test selector passed to scripts/test_all.py (default: correctness).",
  )
  parser.add_argument(
    "--cpu-dump-out",
    type=Path,
    help="Output directory for dump_golden_and_hlo_cpu.py (required with --run).",
  )
  parser.add_argument(
    "--ir-upload-tag",
    help="Override the generated IR-upload tag for the CPU dump command.",
  )
  parser.add_argument(
    "--allow-extra-hlo-op",
    action="append",
    default=[],
    help=(
      "Acknowledge one intentional non-structural outer HLO opcode. Repeat for "
      "multiple opcodes; unacknowledged opcodes block delivery."
    ),
  )
  parser.add_argument("--pr-text", type=Path, help="PR body or commit message to audit")
  parser.add_argument("--commit-message", type=Path, help="Draft commit message to validate")
  parser.add_argument(
    "--write-commit-template",
    type=Path,
    help="Copy the bundled commit-message template to this path.",
  )
  parser.add_argument(
    "--allow-missing-commit-message",
    action="store_true",
    help="Do not block --run when changes exist but no draft was supplied.",
  )
  parser.add_argument("--allow-missing-expected", action="store_true")
  parser.add_argument("--run", action="store_true")
  parser.add_argument("--json-out", type=Path)
  args = parser.parse_args()

  repo = args.repo.resolve()
  if not (repo / ".git").exists():
    raise SystemExit(f"Not a git repository: {repo}")
  config = args.config or args.kernel
  test = args.test or f"test_{args.kernel}"
  changed = [line for line in _git(repo, "status", "--porcelain").splitlines() if line]
  changed_files = [line[3:] for line in changed]
  branch = _git(repo, "branch", "--show-current")
  ci_discovery = _discover_ci(repo)

  if args.write_commit_template:
    if not COMMIT_TEMPLATE.is_file():
      raise SystemExit(f"Missing bundled commit-message template: {COMMIT_TEMPLATE}")
    args.write_commit_template.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(COMMIT_TEMPLATE, args.write_commit_template)

  expected = [
    f"pallas_kernels/{args.package}/{args.kernel}.py",
    f"pallas_kernels/configs/{config}.yaml",
    f"tests/{args.package}/test_{args.kernel}.py",
    f"jax_ops/{args.kernel}.py",
  ]
  expected_status = {path: (repo / path).is_file() for path in expected}
  docs_candidates = [
    repo / "docs" / args.package / f"{args.kernel}.md",
    repo / "docs" / f"{args.kernel}.md",
  ]
  expected_status["docs/<package-or-root>/" + f"{args.kernel}.md"] = any(
    path.is_file() for path in docs_candidates
  )
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
    "ir_upload_tag_scope": IR_UPLOAD_TAG_SCOPE,
    "ci_discovery": ci_discovery,
    "commit_message_template": str(COMMIT_TEMPLATE),
    "checks": [],
  }
  if args.write_commit_template:
    payload["commit_message_template_copy"] = str(
      args.write_commit_template.resolve()
    )
  if args.commit_message:
    commit_text = args.commit_message.read_text(encoding="utf-8")
    payload["commit_message"] = {
      "path": str(args.commit_message.resolve()),
      **_validate_commit_message(commit_text),
      "ir_upload_tag_present": ir_tag in commit_text,
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
    required_inputs = {
      "tpu_test_file": args.tpu_test_file,
      "snapshot_root": str(args.snapshot_root) if args.snapshot_root else None,
      "cpu_dump_out": str(args.cpu_dump_out) if args.cpu_dump_out else None,
    }
    payload["required_acceptance_inputs"] = required_inputs
    missing_inputs = [name for name, value in required_inputs.items() if not value]
    if missing_inputs:
      payload.setdefault("input_errors", []).append(
        "--run requires: " + ", ".join(missing_inputs)
      )
    payload["checks"].append(
      _run(["git", "diff", "--check"], repo, name="git_diff_check")
    )
    python_files = [path for path in changed_files if path.endswith(".py")]
    python = _find_executable(repo, ("python", "python3"))
    surfaces = ci_discovery["surfaces"]

    pre_commit = _tool_command(repo, "pre-commit", "pre_commit", python)
    if pre_commit:
      payload["checks"].append(
        _run([*pre_commit, "run", "--all-files"], repo, name="pre_commit_all")
      )
    else:
      payload["checks"].append(
        _missing_check("pre_commit_all", "pre-commit is a mandatory delivery gate")
      )

    ruff = _tool_command(repo, "ruff", "ruff", python)
    if ruff:
      ruff_config = repo / ".github" / "workflows" / "ruff.toml"
      cmd = [*ruff, "check"]
      if ruff_config.is_file():
        cmd.extend(["--config", str(ruff_config)])
      payload["checks"].append(_run([*cmd, "."], repo, name="ruff_check"))
      format_cmd = [*ruff, "format", "--check"]
      if ruff_config.is_file():
        format_cmd.extend(["--config", str(ruff_config)])
      payload["checks"].append(
        _run([*format_cmd, "."], repo, name="ruff_format_check")
      )
    else:
      payload["checks"].extend(
        [
          _missing_check("ruff_check", "Ruff is a mandatory delivery gate"),
          _missing_check("ruff_format_check", "Ruff format is a mandatory delivery gate"),
        ]
      )

    if python and args.tpu_test_file and args.snapshot_root:
      payload["checks"].append(
        _run(
          [
            python,
            "scripts/test_all.py",
            "-i",
            args.tpu_test_file,
            "-o",
            str(args.snapshot_root),
            "--snapshot",
            "-c",
            args.tpu_test_case,
          ],
          repo,
          name="tpu_correctness_snapshot",
        )
      )
    elif not python:
      payload["checks"].append(
        _missing_check("tpu_correctness_snapshot", "Python is required for TPU validation")
      )

    if python and args.cpu_dump_out:
      cpu_dump_tag = args.ir_upload_tag or ir_tag
      payload["cpu_dump"] = {
        "commit_msg": cpu_dump_tag,
        "commit": _git(repo, "rev-parse", "HEAD"),
        "out_dir": str(args.cpu_dump_out.resolve()),
      }
      payload["checks"].append(
        _run(
          [
            python,
            "tools/dump_golden_and_hlo_cpu.py",
            "--commit-msg",
            cpu_dump_tag,
            "--commit",
            payload["cpu_dump"]["commit"],
            "--out-dir",
            str(args.cpu_dump_out),
            "--strict",
          ],
          repo,
          name="cpu_golden_hlo_dump_strict",
        )
      )
    elif not python:
      payload["checks"].append(
        _missing_check("cpu_golden_hlo_dump_strict", "Python is required for CPU IR validation")
      )

    typing_helper = repo / ".ci-shared" / "scripts" / "typing_helper.py"
    if surfaces["typing"] and python_files:
      if python and typing_helper.is_file():
        payload["checks"].append(
          _run(
            [python, str(typing_helper), "--changed-files", ",".join(python_files)],
            repo,
            name="typing_helper",
          )
        )
      else:
        mypy = _tool_command(repo, "mypy", "mypy", python)
        if mypy:
          payload["checks"].append(
            _run([*mypy, *python_files], repo, name="mypy")
          )
        else:
          payload["checks"].append(
            _missing_check("typing", "Typing is required by repository CI")
          )

    if surfaces["unit_tests"]:
      unit_root = repo / "tests" / "unit"
      if python and unit_root.is_dir():
        payload["checks"].append(
          _run(
            [python, "-m", "pytest", "tests/unit", "-q"],
            repo,
            name="unit_tests",
          )
        )
      elif not python:
        payload["checks"].append(
          _missing_check("unit_tests", "Python is required for repository unit tests")
        )
      else:
        payload["checks"].append(
          _missing_check(
            "unit_tests",
            "Pytest is required by a workflow, but tests/unit was not found; "
            "run and record the exact workflow-specific pytest target",
          )
        )

    validator = repo / "tools" / "config_validator.py"
    if surfaces["config_validator"]:
      if python:
        payload["checks"].append(
          _run([python, str(validator)], repo, name="config_validator")
        )
      else:
        payload["checks"].append(
          _missing_check("config_validator", "Python is required for config validation")
        )

    # Re-scan artifacts after the mandatory commands have run. The initial inventory
    # above is useful for read-only audits, but must not make a fresh --run look empty.
    if args.snapshot_root:
      root = args.snapshot_root.resolve()
      payload["snapshot"] = {
        "root": str(root),
        "before_opt_count": len(list(root.glob("**/*_before_opt.hlo"))),
        "error_count": len(list(root.glob("**/*_error.log"))),
      }

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
  if args.run:
    blockers.extend(payload.get("input_errors", []))
    if args.snapshot_root:
      snapshot_files = list(args.snapshot_root.resolve().glob("**/*"))
      payload.setdefault("artifacts", {})["tpu_snapshot_files"] = len(
        [path for path in snapshot_files if path.is_file()]
      )
      if not payload["artifacts"]["tpu_snapshot_files"]:
        blockers.append("TPU snapshot directory is empty or was not inspectable")
    if args.cpu_dump_out:
      cpu_files = list(args.cpu_dump_out.resolve().glob("**/*"))
      payload.setdefault("artifacts", {})["cpu_dump_files"] = len(
        [path for path in cpu_files if path.is_file()]
      )
      if not payload["artifacts"]["cpu_dump_files"]:
        blockers.append("CPU golden/HLO dump directory is empty or was not inspectable")
  allowed_extra_ops = set(args.allow_extra_hlo_op)
  tpu_hlo_audit = _audit_hlo_root(
    args.snapshot_root, allowed_extra_ops=allowed_extra_ops
  )
  cpu_hlo_audit = _audit_hlo_root(
    args.cpu_dump_out, allowed_extra_ops=allowed_extra_ops
  )
  hlo_comparisons = _compare_hlo_audits(tpu_hlo_audit, cpu_hlo_audit)
  payload["hlo_audit"] = {
    "structural_opcodes": sorted(STRUCTURAL_HLO_OPS),
    "acknowledged_extra_opcodes": sorted(allowed_extra_ops),
    "tpu_snapshot": tpu_hlo_audit,
    "cpu_dump": cpu_hlo_audit,
    "same_name_comparisons": hlo_comparisons,
  }
  for source_name, audit in (
    ("TPU snapshot", tpu_hlo_audit),
    ("CPU dump", cpu_hlo_audit),
  ):
    if audit is None:
      continue
    if not audit["files"]:
      blockers.append(f"{source_name} contains no *_before_opt.hlo files to audit")
    for item in audit["files"]:
      if item["custom_call_count"] == 0:
        blockers.append(f"{item['path']} contains zero custom calls")
      if item["unexpected_non_custom_opcode_counts"]:
        blockers.append(
          f"{item['path']} contains unacknowledged non-custom HLO ops: "
          f"{item['unexpected_non_custom_opcode_counts']}"
        )
  for comparison in hlo_comparisons:
    if comparison["status"] != "pass":
      blockers.append(
        "TPU/CPU HLO custom-call mismatch for " + comparison["filename"]
      )
  if failures:
    blockers.append("One or more delivery commands failed")
  commit_message = payload.get("commit_message")
  if commit_message:
    warnings.extend(commit_message["warnings"])
    if commit_message["status"] != "pass":
      blockers.append(f"Commit-message draft is invalid: {commit_message['errors']}")
  elif args.run and changed_files and not args.allow_missing_commit_message:
    blockers.append(
      "Changed files require a validated commit-message draft; pass --commit-message "
      "or explicitly use --allow-missing-commit-message"
    )
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

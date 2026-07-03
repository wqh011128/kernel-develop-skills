#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "command": subprocess.list2cmdline(command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> int:
    results: dict[str, object] = {
        "syntax": [],
        "skills": [],
        "resources": [],
        "tests": [],
    }
    failed = False
    for path in sorted(ROOT.glob("**/*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            results["syntax"].append({"path": str(path.relative_to(ROOT)), "status": "pass"})
        except (SyntaxError, UnicodeError) as exc:
            failed = True
            results["syntax"].append({"path": str(path.relative_to(ROOT)), "status": "fail", "error": str(exc)})

    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    quick_validate = codex_home / "skills" / ".system" / "skill-creator" / "scripts" / "quick_validate.py"
    for skill_file in sorted(ROOT.glob("*/SKILL.md")):
        lines = skill_file.read_text(encoding="utf-8").splitlines()
        record: dict[str, object] = {"skill": skill_file.parent.name, "lines": len(lines)}
        if len(lines) > 500:
            record.update({"status": "fail", "error": "SKILL.md exceeds 500 lines"})
            failed = True
        elif quick_validate.is_file():
            validation = run([sys.executable, str(quick_validate), str(skill_file.parent)])
            record["validation"] = validation
            record["status"] = "pass" if validation["returncode"] == 0 else "fail"
            failed |= validation["returncode"] != 0
        else:
            record["status"] = "pass_with_quick_validate_unavailable"
        results["skills"].append(record)

    required_resources = {
        "kernel-design-docs/references/RFC_template.md": tuple(
            f"## {index}." for index in range(1, 15)
        ),
        "kernel-dev-lifecycle/assets/commit_message_template.txt": (
            "Task:",
            "Solution:",
            "Test:",
            "JIRA: COMPIL-XXXX",
        ),
    }
    for relative, markers in required_resources.items():
        path = ROOT / relative
        missing = []
        if not path.is_file():
            missing.append("file")
        else:
            text = path.read_text(encoding="utf-8")
            missing.extend(marker for marker in markers if marker not in text)
        record = {
            "path": relative,
            "status": "pass" if not missing else "fail",
            "missing": missing,
        }
        results["resources"].append(record)
        failed |= bool(missing)

    for suite in sorted(ROOT.glob("*/scripts/tests")):
        validation = run([sys.executable, "-m", "unittest", "discover", "-s", str(suite), "-v"])
        results["tests"].append({"suite": str(suite.relative_to(ROOT)), **validation})
        failed |= validation["returncode"] != 0

    results["passed"] = not failed
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

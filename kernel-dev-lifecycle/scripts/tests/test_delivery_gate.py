from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).resolve().parents[1] / "kernel_delivery_gate.py"
SPEC = importlib.util.spec_from_file_location("kernel_delivery_gate", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class DeliveryGateTest(unittest.TestCase):
    def test_bundled_commit_template_uses_required_shape(self) -> None:
        template = GATE.COMMIT_TEMPLATE.read_text(encoding="utf-8")
        self.assertTrue(template.startswith("feat[TOOL]: <imperative summary>"))
        for section in ("Task:\n- ", "Solution:\n- ", "Test:\n- ", "JIRA: COMPIL-XXXX"):
            self.assertIn(section, template)

    def test_commit_message_contract_accepts_real_jira_and_warns_for_placeholder(self) -> None:
        valid = """feat[TOOL]: add IR simulator support

Task:
- Implement a simulation pass.

Solution:
- Use a visitor pattern to traverse the IR.

Test:
- Unit tests for the relu op.

JIRA: COMPIL-123
"""
        result = GATE._validate_commit_message(valid)
        self.assertEqual(result, {"status": "pass", "errors": [], "warnings": []})

        placeholder = valid.replace("COMPIL-123", "COMPIL-XXXX")
        result = GATE._validate_commit_message(placeholder)
        self.assertEqual(result["status"], "pass")
        self.assertIn("Replace the virtual JIRA placeholder", result["warnings"][0])

    def test_commit_message_contract_rejects_wrong_shape(self) -> None:
        result = GATE._validate_commit_message("feat: vague\n\nJIRA: none")
        self.assertEqual(result["status"], "fail")
        self.assertGreaterEqual(len(result["errors"]), 4)

    def test_ci_discovery_inventories_workflows_and_common_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            workflows = repo / ".github" / "workflows"
            workflows.mkdir(parents=True)
            (repo / "tests" / "unit").mkdir(parents=True)
            (repo / ".pre-commit-config.yaml").write_text("repos: []\n", encoding="utf-8")
            (workflows / "ci.yml").write_text(
                """jobs:
  checks:
    steps:
      - run: pre-commit run --all-files
      - run: ruff check .
      - run: |
          ruff format --check .
          echo formatting-complete
      - run: mypy src
      - run: pytest tests/unit
""",
                encoding="utf-8",
            )
            result = GATE._discover_ci(repo)
            self.assertEqual(result["workflow_files"], [".github/workflows/ci.yml"])
            self.assertEqual(len(result["workflow_run_commands"]), 5)
            self.assertEqual(
                result["workflow_run_commands"][2]["command"],
                "ruff format --check .\necho formatting-complete",
            )
            self.assertTrue(all(result["surfaces"][name] for name in (
                "pre_commit", "ruff_check", "ruff_format", "typing", "unit_tests"
            )))

    def test_write_commit_template_copies_bundled_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            destination = Path(temp) / "draft.txt"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--repo",
                    str(repo),
                    "--kernel",
                    "sample",
                    "--write-commit-template",
                    str(destination),
                    "--allow-missing-expected",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Task:\n", destination.read_text(encoding="utf-8"))

    def test_cli_accepts_valid_commit_draft_without_ir_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            draft = Path(temp) / "draft.txt"
            draft.write_text(
                """feat[TOOL]: add sample kernel

Task:
- Add the requested kernel.

Solution:
- Implement the Pallas path.

Test:
- Unit tests pass.

JIRA: COMPIL-123
""",
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(SCRIPT),
                "--repo",
                str(repo),
                "--kernel",
                "sample",
                "--commit-message",
                str(draft),
                "--allow-missing-expected",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["commit_message"]["status"], "pass")

    def test_discovers_root_agents_and_emits_ir_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "AGENTS.md").write_text("# Contract\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--repo",
                    str(repo),
                    "--kernel",
                    "sample",
                    "--allow-missing-expected",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["agents"], ["AGENTS.md"])
            self.assertEqual(
                payload["ir_upload_tag"],
                "[ir-upload package=kernels kernel=sample config=sample "
                "test=test_sample device_num=1]",
            )
            self.assertIn("one runnable upload matrix item", payload["ir_upload_tag_scope"])
            self.assertEqual(payload["status"], "pass")

    def test_run_requires_all_acceptance_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--repo",
                    str(repo),
                    "--kernel",
                    "sample",
                    "--run",
                    "--allow-missing-expected",
                    "--allow-missing-commit-message",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            payload = json.loads(completed.stdout)
            self.assertTrue(any("tpu_test_file" in item for item in payload["blockers"]))
            self.assertTrue(any("cpu_dump_out" in item for item in payload["blockers"]))

    def test_multiple_internal_hlo_phases_still_emit_one_matrix_item_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "AGENTS.md").write_text("# Contract\n", encoding="utf-8")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            for phase in ("permute", "routes_to_token_major", "reduce"):
                (snapshot / f"{phase}_before_opt.hlo").write_text(
                    f"""HloModule {phase}

ENTRY main {{
  p0 = bf16[8] parameter(0)
  ROOT result = bf16[8] custom-call(p0), custom_call_target="tpu_custom_call"
}}
""",
                    encoding="utf-8",
                )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--repo",
                    str(repo),
                    "--kernel",
                    "moe_permute_combine",
                    "--config",
                    "moe_permute_combine",
                    "--test",
                    "test_moe_permute_combine",
                    "--snapshot-root",
                    str(snapshot),
                    "--allow-missing-expected",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["snapshot"]["before_opt_count"], 3)
            self.assertEqual(
                payload["ir_upload_tag"],
                "[ir-upload package=kernels kernel=moe_permute_combine "
                "config=moe_permute_combine test=test_moe_permute_combine device_num=1]",
            )

    def test_hlo_audit_reports_custom_calls_and_outer_ops(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            hlo = Path(temp) / "sample_before_opt.hlo"
            hlo.write_text(
                """HloModule sample

ENTRY main {
  p0 = bf16[8]{0} parameter(0)
  reshaped = bf16[8]{0} reshape(p0)
  ROOT result = bf16[8]{0} custom-call(reshaped), custom_call_target="tpu_custom_call"
}
""",
                encoding="utf-8",
            )

            audit = GATE._audit_hlo_file(hlo, allowed_extra_ops=set())

            self.assertEqual(audit["custom_call_count"], 1)
            self.assertEqual(audit["custom_call_targets"], ["tpu_custom_call"])
            self.assertEqual(audit["non_custom_opcode_counts"], {"parameter": 1, "reshape": 1})
            self.assertEqual(audit["unexpected_non_custom_opcode_counts"], {"reshape": 1})

            acknowledged = GATE._audit_hlo_file(
                hlo, allowed_extra_ops={"reshape"}
            )
            self.assertEqual(acknowledged["unexpected_non_custom_opcode_counts"], {})

    def test_hlo_audit_compares_same_named_tpu_and_cpu_dumps(self) -> None:
        tpu = {
            "files": [
                {
                    "path": "/tpu/sample_before_opt.hlo",
                    "custom_call_count": 2,
                    "custom_call_targets": ["a", "b"],
                }
            ]
        }
        cpu = {
            "files": [
                {
                    "path": "/cpu/sample_before_opt.hlo",
                    "custom_call_count": 1,
                    "custom_call_targets": ["a"],
                }
            ]
        }

        comparisons = GATE._compare_hlo_audits(tpu, cpu)

        self.assertEqual(comparisons[0]["status"], "fail")
        self.assertFalse(comparisons[0]["custom_call_count_match"])
        self.assertFalse(comparisons[0]["custom_call_targets_match"])


if __name__ == "__main__":
    unittest.main()

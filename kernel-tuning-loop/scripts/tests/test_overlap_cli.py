from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "overlap_feasibility.py"


class OverlapCliTest(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False)

    def test_rejects_expression_order_equivalent_to_serial(self) -> None:
        completed = self.run_cli(
            "--compute-ms", "2", "--comm-ms", "1", "--serial-ms", "3", "--candidate-ms", "2.9"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["decision"], "reject_expression_order_tuning")

    def test_rejects_negative_measurements(self) -> None:
        completed = self.run_cli(
            "--compute-ms", "-1", "--comm-ms", "1", "--serial-ms", "1", "--candidate-ms", "1"
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("non-negative", completed.stderr)


if __name__ == "__main__":
    unittest.main()

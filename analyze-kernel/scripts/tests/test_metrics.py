from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import benchmark_utils  # noqa: E402
import correctness  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_identical_zero_arrays_pass(self) -> None:
        result = correctness.compare(np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertEqual(result["cosine"], 1.0)
        self.assertEqual(result["status"], "PASS")

    def test_shape_broadcasting_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            correctness.max_diff(np.zeros((2, 1)), np.zeros((2,)))

    def test_non_finite_values_fail(self) -> None:
        result = correctness.compare(np.array([np.nan]), np.array([np.nan]))
        self.assertFalse(result["finite"])
        self.assertEqual(result["status"], "FAIL")

    def test_benchmark_stats_validate_samples(self) -> None:
        stats = benchmark_utils.compute_stats([1.0, 2.0, 3.0])
        self.assertEqual(stats["median"], 2.0)
        with self.assertRaises(ValueError):
            benchmark_utils.compute_stats([])
        with self.assertRaises(ValueError):
            benchmark_utils.compute_stats([float("nan")])


if __name__ == "__main__":
    unittest.main()

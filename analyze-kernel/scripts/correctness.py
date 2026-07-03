"""Kernel-agnostic correctness comparison utilities."""

from __future__ import annotations

import numpy as np


def _pair(a, b):
    left = np.asarray(a)
    right = np.asarray(b)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    return left.astype(np.float32), right.astype(np.float32)


def cos_sim(a, b):
    a, b = _pair(a, b)
    a = a.flatten()
    b = b.flatten()
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 and b_norm == 0:
        return 1.0
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def max_diff(a, b):
    a, b = _pair(a, b)
    return float(np.max(np.abs(a - b)))


def mean_diff(a, b):
    a, b = _pair(a, b)
    return float(np.mean(np.abs(a - b)))


def check_pass(cos, threshold=0.9999):
    if cos >= threshold:
        return "PASS"
    elif cos >= 0.99:
        return "MARGINAL"
    else:
        return "FAIL"


def compare(a, b, *, atol=1e-5, rtol=1e-5, cosine_threshold=0.9999):
    """Return standard numerical evidence without hiding shape or finite failures."""
    left, right = _pair(a, b)
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    abs_error = np.abs(left - right)
    denominator = np.maximum(np.abs(right), np.finfo(np.float32).tiny)
    cosine = cos_sim(left, right)
    allclose = bool(np.allclose(left, right, atol=atol, rtol=rtol)) if finite else False
    passed = finite and allclose and cosine >= cosine_threshold
    return {
        "shape": list(left.shape),
        "finite": finite,
        "atol": float(atol),
        "rtol": float(rtol),
        "cosine_threshold": float(cosine_threshold),
        "max_abs_diff": float(abs_error.max()) if abs_error.size else 0.0,
        "mean_abs_diff": float(abs_error.mean()) if abs_error.size else 0.0,
        "max_relative_diff": float((abs_error / denominator).max()) if abs_error.size else 0.0,
        "cosine": cosine,
        "allclose": allclose,
        "status": "PASS" if passed else "FAIL",
    }

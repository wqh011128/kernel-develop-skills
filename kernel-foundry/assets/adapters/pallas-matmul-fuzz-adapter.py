"""Example semantic-fuzz adapter for a registry-backed PallasKernels matmul.

Set PALLAS_KERNELS_TEST_CONFIG and put the PallasKernels checkout on
PYTHONPATH before running kernel-foundry fuzz.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from jax_ops import run_reference_op
from jax_ops.matmul import matmul_reference
from pallas_kernels.kernels import get_kernel
from pallas_kernels.kernels.config_loader import load_kernel_config


_CONFIG_NAME = os.environ.get("PALLAS_KERNELS_TEST_CONFIG", "matmul")
_CONFIG = load_kernel_config(_CONFIG_NAME)
_DTYPE = jnp.dtype(_CONFIG.dtype)
_KERNEL = get_kernel("matmul", _CONFIG)
_MODES = ("zeros", "ones", "random")


def generate_case(rng, index):
  return {"mode": _MODES[index % len(_MODES)], "seed": rng.randrange(0, 2**31)}


def _inputs(case):
  shape = _CONFIG.shape
  x_shape = (shape["m"], shape["k"])
  y_shape = (shape["k"], shape["n"])
  if case["mode"] == "zeros":
    return jnp.zeros(x_shape, _DTYPE), jnp.zeros(y_shape, _DTYPE)
  if case["mode"] == "ones":
    return jnp.ones(x_shape, _DTYPE), jnp.ones(y_shape, _DTYPE)
  key = jax.random.PRNGKey(case["seed"])
  x_key, y_key = jax.random.split(key)
  scale = shape["k"] ** -0.5
  return (
    jax.random.normal(x_key, x_shape, _DTYPE) * scale,
    jax.random.normal(y_key, y_shape, _DTYPE),
  )


def evaluate(case):
  x, y = _inputs(case)
  reference = run_reference_op("matmul", matmul_reference, x, y, out_dtype=_DTYPE)
  candidate = _KERNEL(x, y, out_dtype=_DTYPE)
  reference_np = np.asarray(reference, dtype=np.float32)
  candidate_np = np.asarray(candidate, dtype=np.float32)
  tolerance = _CONFIG.get_tolerance(_DTYPE.name)
  finite = bool(np.isfinite(candidate_np).all())
  allclose = bool(np.allclose(reference_np, candidate_np, **tolerance))
  max_abs_diff = float(np.max(np.abs(reference_np - candidate_np)))
  return {
    "passed": finite and allclose,
    "checks": [
      {"name": "finite", "passed": finite},
      {"name": "independent_reference_allclose", "passed": allclose},
    ],
    "metrics": {"max_abs_diff": max_abs_diff, **tolerance},
  }

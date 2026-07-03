#!/usr/bin/env python3
"""Blind reproduction challenge for three registry-backed Pallas kernels.

The candidate implementations below depend only on the public mathematical
contract, shapes, dtypes, and Pallas API. They intentionally do not import
production kernel modules or registry runner implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _block(tree: Any) -> Any:
  return jax.tree_util.tree_map(
    lambda value: value.block_until_ready()
    if hasattr(value, "block_until_ready")
    else value,
    tree,
  )


def _compiler_params(rank: int) -> pltpu.CompilerParams:
  return pltpu.CompilerParams(dimension_semantics=("parallel",) * rank)


def candidate_matmul(x: jax.Array, y: jax.Array) -> jax.Array:
  if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[0]:
    raise ValueError(f"invalid matmul shapes: {x.shape}, {y.shape}")
  m, k = x.shape
  _, n = y.shape
  block_m = 128
  block_n = 128
  if m % block_m or n % block_n:
    raise ValueError("candidate matmul requires m and n divisible by 128")

  def kernel(x_ref, y_ref, out_ref):
    acc = jnp.dot(
      x_ref[...],
      y_ref[...],
      preferred_element_type=jnp.float32,
    )
    out_ref[...] = acc.astype(out_ref.dtype)

  return pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
    grid=(m // block_m, n // block_n),
    in_specs=(
      pl.BlockSpec((block_m, k), lambda i, j: (i, 0)),
      pl.BlockSpec((k, block_n), lambda i, j: (0, j)),
    ),
    out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
    compiler_params=_compiler_params(2),
    name="foundry_blind_matmul",
  )(x, y)


def candidate_moe_router(
  logits: jax.Array,
  *,
  top_k: int,
) -> tuple[jax.Array, jax.Array]:
  if logits.ndim != 2 or top_k <= 0 or top_k > logits.shape[1]:
    raise ValueError(f"invalid MoE router contract: {logits.shape}, top_k={top_k}")
  tokens, experts = logits.shape
  block_tokens = 1024
  if tokens % block_tokens:
    raise ValueError("candidate MoE router requires tokens divisible by 1024")

  def kernel(logits_ref, weights_ref, indices_ref):
    logits_f32 = logits_ref[...].astype(jnp.float32)
    row_max = jnp.max(logits_f32, axis=1, keepdims=True)
    normalizer = jnp.sum(jnp.exp(logits_f32 - row_max), axis=1)
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    scores = logits_f32
    for index in range(top_k):
      top_logits = jnp.max(scores, axis=1)
      top_indices = jnp.argmax(scores, axis=1).astype(jnp.int32)
      weights_ref[:, index] = (
        jnp.exp(top_logits - row_max[:, 0]) / normalizer
      ).astype(weights_ref.dtype)
      indices_ref[:, index] = top_indices
      scores = jnp.where(expert_ids == top_indices[:, None], -jnp.inf, scores)

  input_spec = pl.BlockSpec((block_tokens, experts), lambda i: (i, 0))
  output_spec = pl.BlockSpec((block_tokens, top_k), lambda i: (i, 0))
  return pl.pallas_call(
    kernel,
    out_shape=(
      jax.ShapeDtypeStruct((tokens, top_k), logits.dtype),
      jax.ShapeDtypeStruct((tokens, top_k), jnp.int32),
    ),
    grid=(tokens // block_tokens,),
    in_specs=(input_spec,),
    out_specs=(output_spec, output_spec),
    compiler_params=_compiler_params(1),
    name="foundry_blind_moe_router",
  )(logits)


def candidate_gqa_decode(
  q: jax.Array,
  k_cache: jax.Array,
  v_cache: jax.Array,
  cache_lengths: jax.Array,
) -> jax.Array:
  batch, num_q_heads, q_len, head_dim = q.shape
  batch_k, num_kv_heads, seq_len, k_dim = k_cache.shape
  if (
    q_len != 1
    or batch_k != batch
    or v_cache.shape != k_cache.shape
    or k_dim != head_dim
    or cache_lengths.shape != (batch,)
    or num_q_heads % num_kv_heads
  ):
    raise ValueError("invalid GQA decode contract")
  group_size = num_q_heads // num_kv_heads
  sm_scale = head_dim**-0.5

  def kernel(q_ref, k_ref, v_ref, length_ref, out_ref):
    query = q_ref[0, :, 0, :]
    keys = k_ref[0, 0, :, :]
    values = v_ref[0, 0, :, :]
    scores = jnp.dot(
      query,
      jnp.swapaxes(keys, 0, 1),
      preferred_element_type=jnp.float32,
    )
    scores = scores * sm_scale
    scores = jnp.where(
      jnp.arange(seq_len, dtype=jnp.int32)[None, :]
      < length_ref[pl.program_id(0), 0],
      scores,
      jnp.asarray(-1e30, jnp.float32),
    )
    shifted = scores - jnp.max(scores, axis=1, keepdims=True)
    numerator = jnp.exp(shifted)
    probs = numerator / jnp.sum(numerator, axis=1, keepdims=True)
    output = jnp.dot(
      probs.astype(values.dtype),
      values,
      preferred_element_type=jnp.float32,
    )
    out_ref[0, :, 0, :] = output.astype(out_ref.dtype)

  q_spec = pl.BlockSpec(
    (1, group_size, 1, head_dim), lambda b, h: (b, h, 0, 0)
  )
  kv_spec = pl.BlockSpec(
    (1, 1, seq_len, head_dim),
    lambda b, h: (b, h, 0, 0),
  )
  length_spec = pl.BlockSpec((batch, 128), lambda b, h: (0, 0))
  aligned_lengths = jnp.broadcast_to(cache_lengths[:, None], (batch, 128))
  return pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
    grid=(batch, num_kv_heads),
    in_specs=(q_spec, kv_spec, kv_spec, length_spec),
    out_specs=q_spec,
    compiler_params=_compiler_params(2),
    name="foundry_blind_gqa_decode",
  )(q, k_cache, v_cache, aligned_lengths)


def _numpy_softmax(x: np.ndarray) -> np.ndarray:
  shifted = x - np.max(x, axis=-1, keepdims=True)
  exp = np.exp(shifted)
  return exp / np.sum(exp, axis=-1, keepdims=True)


def _numpy_reference(name: str, inputs: tuple[jax.Array, ...], eps: float) -> Any:
  arrays = [np.asarray(value).astype(np.float32) for value in inputs]
  if name == "matmul":
    return arrays[0] @ arrays[1]
  if name == "moe_router":
    logits = arrays[0]
    top_k = int(eps)
    indices = np.argsort(-logits, axis=1, kind="stable")[:, :top_k].astype(np.int32)
    weights = _numpy_softmax(logits)
    return np.take_along_axis(weights, indices, axis=1), indices
  if name == "gqa_decode":
    q, k_cache, v_cache = arrays[:3]
    cache_lengths = np.asarray(inputs[3]).astype(np.int32)
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    kv_indices = np.arange(num_q_heads) // group_size
    keys = k_cache[:, kv_indices]
    values = v_cache[:, kv_indices]
    scores = np.einsum("bhsd,bhtd->bhst", q, keys) * q.shape[-1] ** -0.5
    positions = np.arange(k_cache.shape[2])[None, None, None, :]
    scores = np.where(positions < cache_lengths[:, None, None, None], scores, -1e30)
    probs = _numpy_softmax(scores)
    return np.einsum("bhst,bhtd->bhsd", probs, values)
  raise ValueError(name)


def _repo_call(name: str, runner: Any, eps: float) -> Callable[..., Any]:
  if name == "matmul":
    return lambda x, y: runner(x, y, out_dtype=x.dtype)
  if name == "moe_router":
    return lambda logits: runner(logits, out_dtype=logits.dtype)
  if name == "gqa_decode":
    return lambda q, k, v, lengths: runner(q, k, v, lengths, sm_scale=None)
  raise ValueError(name)


def _candidate_call(name: str, eps: float) -> Callable[..., Any]:
  if name == "matmul":
    return candidate_matmul
  if name == "moe_router":
    return lambda logits: candidate_moe_router(logits, top_k=int(eps))
  if name == "gqa_decode":
    return candidate_gqa_decode
  raise ValueError(name)


def _make_inputs(name: str, shape: dict[str, int], case: str, seed: int) -> tuple[jax.Array, ...]:
  rng = np.random.default_rng(seed)
  if name == "matmul":
    m, k, n = shape["m"], shape["k"], shape["n"]
    if case == "zeros":
      x = np.zeros((m, k), np.float32)
      y = np.zeros((k, n), np.float32)
    elif case == "structured":
      x = np.full((m, k), k**-0.5, np.float32)
      y = np.where(np.arange(k * n).reshape(k, n) % 2, -1.0, 1.0).astype(np.float32)
    else:
      x = rng.normal(size=(m, k)).astype(np.float32) * k**-0.5
      y = rng.normal(size=(k, n)).astype(np.float32)
    return jnp.asarray(x, jnp.bfloat16), jnp.asarray(y, jnp.bfloat16)

  if name == "moe_router":
    tokens, experts = shape["tokens"], shape["experts"]
    if case == "zeros":
      logits = np.zeros((tokens, experts), np.float32)
    elif case == "structured":
      logits = np.broadcast_to(
        np.linspace(-2.0, 2.0, experts, dtype=np.float32), (tokens, experts)
      ).copy()
    else:
      logits = rng.normal(size=(tokens, experts)).astype(np.float32) * 0.5
      logits += np.arange(experts, dtype=np.float32)[None, :] * 0.01
    return (jnp.asarray(logits, jnp.bfloat16),)
  if name == "gqa_decode":
    batch = shape["batch"]
    q_heads = shape["num_q_heads"]
    kv_heads = shape["num_kv_heads"]
    seq_len = shape["seq_len"]
    head_dim = shape["head_dim"]
    if case == "zeros":
      q = np.zeros((batch, q_heads, 1, head_dim), np.float32)
      k = np.zeros((batch, kv_heads, seq_len, head_dim), np.float32)
      v = np.zeros((batch, kv_heads, seq_len, head_dim), np.float32)
    else:
      q = rng.normal(size=(batch, q_heads, 1, head_dim)).astype(np.float32) * 0.02
      k = rng.normal(size=(batch, kv_heads, seq_len, head_dim)).astype(np.float32) * 0.02
      v = rng.normal(size=(batch, kv_heads, seq_len, head_dim)).astype(np.float32) * 0.02
      if case == "structured":
        q.fill(0.02)
        k[..., ::2, :].fill(0.03)
    lengths = np.maximum(1, seq_len - np.arange(batch, dtype=np.int32) * 7)
    return (
      jnp.asarray(q, jnp.bfloat16),
      jnp.asarray(k, jnp.bfloat16),
      jnp.asarray(v, jnp.bfloat16),
      jnp.asarray(lengths, jnp.int32),
    )
  raise ValueError(name)


def _leaves(value: Any) -> list[np.ndarray]:
  return [np.asarray(leaf).astype(np.float32) for leaf in jax.tree_util.tree_leaves(value)]


def _correctness(name: str, implementation: str, output: Any, reference: Any, atol: float, rtol: float) -> dict[str, Any]:
  got_leaves = _leaves(output)
  ref_leaves = [reference] if isinstance(reference, np.ndarray) else list(reference)
  if len(got_leaves) != len(ref_leaves):
    return {"passed": False, "error": "output leaf count mismatch"}
  if name == "moe_router":
    got_weights = got_leaves[0]
    got_indices = got_leaves[1].astype(np.int32)
    ref_weights = np.asarray(ref_leaves[0], dtype=np.float32)
    finite = bool(np.all(np.isfinite(got_weights)))
    shape_ok = got_weights.shape == ref_weights.shape == got_indices.shape
    in_range = bool(np.all((got_indices >= 0) & (got_indices < 256)))
    unique = bool(
      np.all(
        np.apply_along_axis(lambda row: len(np.unique(row)), 1, got_indices)
        == got_indices.shape[1]
      )
    )
    # bfloat16 creates exact ties. Any unique expert set with the same selected
    # score/weight multiset is semantically valid; exact index order is not.
    sorted_got = np.sort(got_weights, axis=1)
    sorted_ref = np.sort(ref_weights, axis=1)
    max_abs = float(np.max(np.abs(sorted_got - sorted_ref)))
    weights_close = bool(np.allclose(sorted_got, sorted_ref, atol=atol, rtol=rtol))
    return {
      "implementation": implementation,
      "passed": finite and shape_ok and in_range and unique and weights_close,
      "tie_policy": "unique indices and selected-weight multiset equivalence",
      "finite": finite,
      "shape_ok": shape_ok,
      "indices_in_range": in_range,
      "indices_unique": unique,
      "selected_weights_allclose": weights_close,
      "max_abs_error": max_abs,
    }
  metrics = []
  passed = True
  for got, expected in zip(got_leaves, ref_leaves):
    expected = np.asarray(expected, dtype=np.float32)
    finite = bool(np.all(np.isfinite(got)))
    abs_error = np.abs(got - expected)
    max_abs = float(np.max(abs_error))
    denom = np.maximum(np.abs(expected), 1e-7)
    max_rel = float(np.max(abs_error / denom))
    close = bool(np.allclose(got, expected, atol=atol, rtol=rtol))
    passed = passed and finite and close
    metrics.append({
      "shape": list(got.shape),
      "finite": finite,
      "max_abs_error": max_abs,
      "max_rel_error": max_rel,
      "allclose": close,
    })
  return {"implementation": implementation, "passed": passed, "leaves": metrics}


def _benchmark(fn: Callable[..., Any], args: tuple[jax.Array, ...], warmup: int, iterations: int, rounds: int) -> dict[str, Any]:
  jitted = jax.jit(fn)
  for _ in range(warmup):
    _block(jitted(*args))
  round_medians = []
  samples = []
  for _ in range(rounds):
    current = []
    for _ in range(iterations):
      start = time.perf_counter_ns()
      _block(jitted(*args))
      elapsed_ms = (time.perf_counter_ns() - start) / 1e6
      current.append(elapsed_ms)
    samples.append(current)
    round_medians.append(statistics.median(current))
  return {
    "round_medians_ms": round_medians,
    "median_of_round_medians_ms": statistics.median(round_medians),
    "min_ms": min(min(values) for values in samples),
    "max_ms": max(max(values) for values in samples),
    "warmup": warmup,
    "iterations": iterations,
    "rounds": rounds,
  }


def _hlo_facts(fn: Callable[..., Any], args: tuple[jax.Array, ...]) -> dict[str, Any]:
  text = str(jax.jit(fn).lower(*args).compiler_ir(dialect="stablehlo"))
  return {
    "stablehlo_bytes": len(text.encode("utf-8")),
    "stablehlo_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    "custom_call_count": text.count("stablehlo.custom_call"),
  }


def _geomean(values: list[float]) -> float:
  return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", required=True)
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--iterations", type=int, default=30)
  parser.add_argument("--rounds", type=int, default=3)
  parser.add_argument(
    "--workload",
    action="append",
    choices=("matmul", "moe_router", "gqa_decode"),
    help="Run only selected workloads; repeat for multiple workloads.",
  )
  args = parser.parse_args()

  repo = Path(args.repo).resolve()
  sys.path.insert(0, str(repo))
  from pallas_kernels.kernels import get_kernel  # pylint: disable=import-outside-toplevel
  from pallas_kernels.kernels.config_loader import load_kernel_config  # pylint: disable=import-outside-toplevel

  workloads = [
    {"name": "matmul", "config": "matmul", "case_id": "matmul_default", "benchmark": True},
    {
      "name": "matmul",
      "config": "matmul",
      "case_id": "matmul_holdout_rect",
      "overrides": {"shape": {"m": 384, "k": 512, "n": 256}},
      "benchmark": False,
    },
    {"name": "moe_router", "config": "moe_router", "case_id": "moe_default", "benchmark": True},
    {
      "name": "moe_router",
      "config": "moe_router",
      "case_id": "moe_holdout_tokens2048",
      "overrides": {"shape": {"tokens": 2048}},
      "benchmark": False,
    },
    {"name": "gqa_decode", "config": "gqa_decode", "case_id": "gqa_default", "benchmark": True},
    {
      "name": "gqa_decode",
      "config": "gqa_decode",
      "case_id": "gqa_holdout_heads_seq",
      "overrides": {
        "shape": {"num_q_heads": 16, "num_kv_heads": 4, "seq_len": 512},
        "tiling": {"block_kv": 512},
      },
      "benchmark": False,
    },
  ]
  if args.workload:
    selected = set(args.workload)
    workloads = [item for item in workloads if item["name"] in selected]
  report: dict[str, Any] = {
    "schema_version": 1,
    "protocol": {
      "candidate_source_excludes": [
        "pallas_kernels/kernels/<target>.py",
        "pallas_kernels/kernels/registry/<target>.py",
        "docs/kernels/<target>.md",
      ],
      "dominant_threshold": "all correctness; >=10% geomean win; no workload >5% regression",
      "competitive_threshold": "all correctness; geomean within 10% of repository",
    },
    "environment": {
      "jax": jax.__version__,
      "backend": jax.default_backend(),
      "devices": [str(device) for device in jax.devices()],
    },
    "workloads": [],
  }
  ratios = []
  correctness_counts = {
    "candidate": {"passed": 0, "total": 0},
    "repository": {"passed": 0, "total": 0},
  }
  no_large_regression = True

  for workload_spec in workloads:
    name = workload_spec["name"]
    config_name = workload_spec["config"]
    cfg = load_kernel_config(config_name, overrides=workload_spec.get("overrides"))
    eps = float(cfg.shape["top_k"]) if name == "moe_router" else 1e-6
    runner = get_kernel(name, cfg)
    repo_fn = _repo_call(name, runner, eps)
    candidate_fn = _candidate_call(name, eps)
    workload: dict[str, Any] = {
      "case_id": workload_spec["case_id"],
      "name": name,
      "config": config_name,
      "shape": cfg.shape,
      "dtype": cfg.dtype,
      "cases": [],
    }
    benchmark_args = None
    for index, case in enumerate(("random", "zeros", "structured")):
      inputs = _make_inputs(name, cfg.shape, case, seed=20260713 + index)
      reference = _numpy_reference(name, inputs, eps)
      case_result: dict[str, Any] = {"case": case}
      for implementation, fn in (("repository", repo_fn), ("candidate", candidate_fn)):
        try:
          output = _block(jax.jit(fn)(*inputs))
          result = _correctness(
            name,
            implementation,
            output,
            reference,
            float(cfg.get_tolerance(cfg.dtype)["atol"]),
            float(cfg.get_tolerance(cfg.dtype)["rtol"]),
          )
        except Exception as exc:  # Candidate failures must not hide baseline evidence.
          result = {"implementation": implementation, "passed": False, "error": repr(exc)}
        case_result[implementation] = result
        correctness_counts[implementation]["total"] += 1
        correctness_counts[implementation]["passed"] += int(bool(result.get("passed")))
      workload["cases"].append(case_result)
      if case == "random":
        benchmark_args = inputs

    if workload_spec["benchmark"] and benchmark_args is not None and all(
      item[impl].get("passed", False)
      for item in workload["cases"]
      for impl in ("repository", "candidate")
    ):
      repo_bench = _benchmark(repo_fn, benchmark_args, args.warmup, args.iterations, args.rounds)
      candidate_bench = _benchmark(
        candidate_fn, benchmark_args, args.warmup, args.iterations, args.rounds
      )
      repo_ms = repo_bench["median_of_round_medians_ms"]
      candidate_ms = candidate_bench["median_of_round_medians_ms"]
      ratio = candidate_ms / repo_ms
      ratios.append(ratio)
      no_large_regression = no_large_regression and ratio <= 1.05
      workload["benchmark"] = {
        "repository": repo_bench,
        "candidate": candidate_bench,
        "candidate_over_repository": ratio,
      }
      workload["hlo"] = {
        "repository": _hlo_facts(repo_fn, benchmark_args),
        "candidate": _hlo_facts(candidate_fn, benchmark_args),
      }
    report["workloads"].append(workload)

  if ratios:
    geomean_ratio = _geomean(ratios)
  else:
    geomean_ratio = None
  candidate_all_correct = (
    correctness_counts["candidate"]["passed"]
    == correctness_counts["candidate"]["total"]
  )
  coverage_not_lower = (
    correctness_counts["candidate"]["passed"]
    >= correctness_counts["repository"]["passed"]
  )
  if candidate_all_correct and coverage_not_lower and geomean_ratio is not None and geomean_ratio <= 0.9 and no_large_regression:
    verdict = "dominant"
  elif candidate_all_correct and coverage_not_lower and geomean_ratio is not None and geomean_ratio <= 1.1:
    verdict = "competitive"
  elif candidate_all_correct and coverage_not_lower:
    verdict = "inferior_performance"
  else:
    verdict = "not_proven_correct"
  report["summary"] = {
    "candidate_all_correct": candidate_all_correct,
    "correctness_coverage": correctness_counts,
    "candidate_coverage_not_lower": coverage_not_lower,
    "benchmarked_workloads": len(ratios),
    "geomean_candidate_over_repository": geomean_ratio,
    "no_workload_over_5pct_regression": no_large_regression,
    "verdict": verdict,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
  main()

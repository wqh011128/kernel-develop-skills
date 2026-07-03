"""Benchmark PallasKernels matmul genome proposals without editing its checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_ops import run_reference_op
from jax_ops.matmul import matmul_reference
from pallas_kernels.kernels import get_kernel
from pallas_kernels.kernels.config_loader import load_kernel_config


def _stats(values):
  data = np.asarray(values, dtype=np.float64)
  return {
    "median_ms": float(np.median(data)),
    "mean_ms": float(np.mean(data)),
    "p5_ms": float(np.percentile(data, 5)),
    "p95_ms": float(np.percentile(data, 95)),
    "iterations": len(values),
  }


def _inputs(shape, dtype, seed):
  key = jax.random.PRNGKey(seed)
  x_key, y_key = jax.random.split(key)
  scale = shape["k"] ** -0.5
  return (
    jax.random.normal(x_key, (shape["m"], shape["k"]), dtype) * scale,
    jax.random.normal(y_key, (shape["k"], shape["n"]), dtype),
  )


def _hlo_features(runner):
  _, function, avals, static_kwargs = runner.pallas_trace_args()[0]
  hlo = str(function.lower(*avals, **static_kwargs).compiler_ir(dialect="stablehlo"))
  return {
    "stablehlo_bytes": len(hlo.encode("utf-8")),
    "custom_call_count": hlo.count("stablehlo.custom_call"),
    "hlo_sha256": hashlib.sha256(hlo.encode("utf-8")).hexdigest(),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--genomes", required=True, type=Path)
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--config", default="matmul")
  parser.add_argument("--shape", action="append", required=True, help="m,k,n")
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--iterations", type=int, default=10)
  args = parser.parse_args()

  genome_report = json.loads(args.genomes.read_text(encoding="utf-8"))
  base = genome_report["base"]
  candidates = [{"id": "base", "genome": base, "mutation": None}]
  candidates.extend(genome_report["proposals"])
  shapes = []
  for raw in args.shape:
    m, k, n = (int(value) for value in raw.split(","))
    shapes.append({"m": m, "k": k, "n": n})
  if args.warmup < 0 or args.iterations <= 0:
    parser.error("warmup must be non-negative and iterations must be positive")

  args.out.mkdir(parents=True, exist_ok=True)
  raw_results = []
  portfolio_rows = []
  causal_observations = []
  for shape_index, shape in enumerate(shapes):
    dtype = jnp.dtype(load_kernel_config(args.config).dtype)
    x, y = _inputs(shape, dtype, seed=shape_index)
    reference = np.asarray(
      run_reference_op("matmul", matmul_reference, x, y, out_dtype=dtype),
      dtype=np.float32,
    )
    shape_records = []
    for candidate in candidates:
      tiling = candidate["genome"]
      cfg = load_kernel_config(args.config, overrides={"shape": shape, "tiling": tiling})
      runner = get_kernel("matmul", cfg)
      failure = None
      times = []
      hlo_features = {}
      finite = False
      correct = False
      max_abs_diff = None
      try:
        output = runner(x, y, out_dtype=dtype)
        output_np = np.asarray(output, dtype=np.float32)
        tolerance = cfg.get_tolerance(dtype.name)
        finite = bool(np.isfinite(output_np).all())
        correct = finite and bool(np.allclose(reference, output_np, **tolerance))
        max_abs_diff = float(np.max(np.abs(reference - output_np)))
        if correct:
          for _ in range(args.warmup):
            runner(x, y, out_dtype=dtype).block_until_ready()
          for _ in range(args.iterations):
            start = time.perf_counter()
            runner(x, y, out_dtype=dtype).block_until_ready()
            times.append((time.perf_counter() - start) * 1000)
        try:
          hlo_features = _hlo_features(runner)
        except Exception as exc:  # Preserve benchmark evidence if only HLO extraction fails.
          failure = {"stage": "hlo", "type": type(exc).__name__, "message": str(exc)}
      except Exception as exc:  # One invalid genome must not abort the candidate sweep.
        failure = {"stage": "compile_or_execute", "type": type(exc).__name__, "message": str(exc)}
      metrics = _stats(times) if times else {}
      record = {
        "candidate": candidate["id"],
        "mutation": candidate.get("mutation"),
        "shape": shape,
        "tiling": tiling,
        "correctness": correct,
        "finite": finite,
        "max_abs_diff": max_abs_diff,
        "metrics": metrics,
        "hlo_features": hlo_features,
        "failure": failure,
      }
      raw_results.append(record)
      shape_records.append(record)
      portfolio_rows.append(
        {
          "candidate": candidate["id"],
          "dimensions": shape,
          "metrics": metrics,
          "correctness": correct,
        }
      )
    baseline = next(record for record in shape_records if record["candidate"] == "base")
    for record in shape_records:
      if record["candidate"] == "base":
        continue
      pair_id = f"m{shape['m']}_k{shape['k']}_n{shape['n']}:{record['candidate']}"
      causal_observations.extend(
        [
          {
            "pair_id": pair_id,
            "role": "baseline",
            "context": shape,
            "source_features": baseline["tiling"],
            "hlo_features": baseline["hlo_features"],
            "metrics": baseline["metrics"],
          },
          {
            "pair_id": pair_id,
            "role": "candidate",
            "context": shape,
            "source_features": record["tiling"],
            "hlo_features": record["hlo_features"],
            "metrics": record["metrics"],
          },
        ]
      )

  (args.out / "benchmark_results.json").write_text(json.dumps({"results": raw_results}, indent=2) + "\n")
  (args.out / "portfolio_input.json").write_text(json.dumps({"rows": portfolio_rows}, indent=2) + "\n")
  (args.out / "causal_input.json").write_text(json.dumps({"observations": causal_observations}, indent=2) + "\n")
  print(json.dumps({"candidate_runs": len(raw_results), "all_correct": all(item["correctness"] for item in raw_results)}, indent=2))


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""Generic remote XProf runner for registry-backed PallasKernels repos.

Copy this file to a TPU host and run it from a shell that sets:

  LIBTPU_INIT_ARGS="--xla_xprof_register_llo_debug_info=true"

The script intentionally imports JAX only after argument parsing and cwd setup.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REQUIRED_LIBTPU_FLAGS = ("--xla_xprof_register_llo_debug_info=true",)


def _block(tree: Any) -> Any:
  import jax  # pylint: disable=import-outside-toplevel

  return jax.tree_util.tree_map(
      lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
      tree,
  )


def _shape_dtype(tree: Any) -> list[dict[str, Any]]:
  import jax  # pylint: disable=import-outside-toplevel

  leaves = jax.tree_util.tree_leaves(tree)
  out = []
  for value in leaves:
    out.append({
        "shape": list(getattr(value, "shape", ())),
        "dtype": str(getattr(value, "dtype", type(value).__name__)),
    })
  return out


def _is_shape_dtype_struct(value: Any) -> bool:
  return (
      hasattr(value, "shape")
      and hasattr(value, "dtype")
      and type(value).__name__ == "ShapeDtypeStruct"
  )


def _materialize_aval(aval: Any, key: Any) -> Any:
  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

  if aval is None:
    return None
  if not _is_shape_dtype_struct(aval):
    return aval
  shape = tuple(aval.shape)
  dtype = jnp.dtype(aval.dtype)
  if jnp.issubdtype(dtype, jnp.complexfloating):
    real_key, imag_key = jax.random.split(key)
    real = jax.random.normal(real_key, shape, dtype=jnp.float32)
    imag = jax.random.normal(imag_key, shape, dtype=jnp.float32)
    return (real + 1j * imag).astype(dtype)
  if jnp.issubdtype(dtype, jnp.integer):
    return jax.random.randint(key, shape, minval=0, maxval=7, dtype=dtype)
  if jnp.issubdtype(dtype, jnp.bool_):
    return jax.random.randint(key, shape, minval=0, maxval=2).astype(dtype)
  return jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype) * jnp.asarray(
      0.02, dtype
  )


def _materialize_kwargs(kwargs: dict[str, Any], key: Any) -> dict[str, Any]:
  import jax  # pylint: disable=import-outside-toplevel

  out = {}
  keys = iter(jax.random.split(key, max(len(kwargs), 1)))
  for name, value in kwargs.items():
    if _is_shape_dtype_struct(value):
      out[name] = _materialize_aval(value, next(keys))
    else:
      out[name] = value
  return out


def _git_value(args: list[str]) -> str | None:
  try:
    return subprocess.check_output(args, text=True).strip()
  except Exception:
    return None


def _split_call_inputs(runner: Any, flat_inputs: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Map make_inputs() tuple onto runner.__call__ positional/keyword-only args.

  Some registry runners include keyword-only arrays (for example rotary tables
  or positions) in their test input tuple. Passing the whole tuple positionally
  fails even though the input contract is otherwise valid.
  """
  sig = inspect.signature(runner.__call__)
  positional_params = []
  keyword_only_names = []
  for param in sig.parameters.values():
    if param.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
      positional_params.append(param)
    elif param.kind == inspect.Parameter.KEYWORD_ONLY:
      keyword_only_names.append(param.name)

  required_pos_count = sum(
      1 for param in positional_params
      if param.default is inspect.Parameter.empty
  )
  if len(flat_inputs) < required_pos_count:
    raise RuntimeError(
        f"{type(runner).__name__}.make_inputs returned {len(flat_inputs)} "
        f"values but __call__ requires {required_pos_count} positional values"
    )
  pos_count = min(len(flat_inputs), len(positional_params))
  args = tuple(flat_inputs[:pos_count])
  extra = list(flat_inputs[pos_count:])
  kwargs = {}
  if extra:
    required_kwonly = [
        name
        for name in keyword_only_names
        if sig.parameters[name].default is inspect.Parameter.empty
    ]
    for name, value in zip(required_kwonly, extra):
      kwargs[name] = value
    extra = extra[len(required_kwonly):]

  if extra:
    # Some make_inputs() contracts append optional array kwargs used by tests or
    # references. Map the common public-runner names before deciding an input is
    # genuinely unusable.
    preferred_optional = [
        "initial_state",
        "freqs_cis",
        "positions",
        "sm_scale",
        "scale",
    ]
    for name in preferred_optional:
      if not extra:
        break
      if name not in keyword_only_names or name in kwargs:
        continue
      value = extra[0]
      if value is None:
        continue
      if name in ("sm_scale", "scale") and getattr(value, "shape", ()) != ():
        continue
      kwargs[name] = value
      extra = extra[1:]

  if extra:
    # Some reference-only residuals, such as gqa_bwd's optional precomputed di,
    # are not accepted by the public runner. Ignore them rather than failing the
    # profile if the public call has all required runtime inputs.
    if not all(value is None or hasattr(value, "shape") for value in extra):
      raise RuntimeError(
          f"{type(runner).__name__}.make_inputs has unsupported extra values: "
          f"{[type(value).__name__ for value in extra]}"
      )
  return args, kwargs


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", required=True, help="Remote repo root.")
  parser.add_argument("--config", required=True, help="Kernel config name.")
  parser.add_argument("--out-root", default="/tmp/xprof")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--step-name", default="train")
  parser.add_argument(
      "--mode",
      choices=("public", "trace-contract"),
      default="public",
      help="Profile public runner call or direct pallas_trace_args contract.",
  )
  args = parser.parse_args()

  repo = Path(args.repo).resolve()
  os.chdir(repo)
  sys.path.insert(0, str(repo))

  libtpu_init_args = os.environ.get("LIBTPU_INIT_ARGS", "")
  missing_flags = [
      flag for flag in REQUIRED_LIBTPU_FLAGS if flag not in libtpu_init_args
  ]
  if missing_flags:
    raise SystemExit(
        "LIBTPU_INIT_ARGS is missing required XProf flags: "
        + ", ".join(missing_flags)
    )

  # LIBTPU_INIT_ARGS is parsed when JAX initializes libtpu. Validate the
  # environment before importing JAX so an actionable gate failure is not
  # replaced by a backend initialization error.
  import jax  # pylint: disable=import-outside-toplevel
  from pallas_kernels.kernels.config_loader import load_kernel_config  # pylint: disable=import-outside-toplevel
  from pallas_kernels.kernels.registry.lookup import get_kernel  # pylint: disable=import-outside-toplevel

  cfg = load_kernel_config(args.config)
  runner = get_kernel(args.config, cfg)
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  trace_root = Path(args.out_root) / args.config / timestamp
  trace_root.mkdir(parents=True, exist_ok=True)
  trace_dir = trace_root / "profile"

  if args.mode == "trace-contract":
    trace_specs = runner.pallas_trace_args()
    key = jax.random.PRNGKey(args.seed)
    split_keys = iter(jax.random.split(key, 2 * max(len(trace_specs), 1) + 8))
    calls = []
    for trace_name, fn, avals, static_kwargs in trace_specs:
      input_keys = iter(jax.random.split(next(split_keys), max(len(avals), 1)))
      call_inputs = tuple(_materialize_aval(aval, next(input_keys)) for aval in avals)
      call_kwargs = _materialize_kwargs(static_kwargs, next(split_keys))

      def call(*xs, _fn=fn, _kwargs=call_kwargs):
        return _fn(*xs, **_kwargs)

      jitted = jax.jit(call)
      for _ in range(args.warmup):
        _block(jitted(*call_inputs))
      calls.append((trace_name, jitted, call_inputs, call_kwargs))

    start = time.perf_counter()
    outputs = []
    jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=True)
    try:
      with jax.profiler.StepTraceAnnotation(args.step_name, step_num=0):
        for trace_name, jitted, call_inputs, _ in calls:
          with jax.profiler.TraceAnnotation(trace_name):
            output = jitted(*call_inputs)
            _block(output)
            outputs.append(output)
    finally:
      jax.profiler.stop_trace()
    elapsed_s = time.perf_counter() - start

    cost_analysis = []
    for trace_name, jitted, call_inputs, _ in calls:
      try:
        cost_analysis.append({
            "trace_name": trace_name,
            "cost_analysis": jitted.lower(*call_inputs).compile().cost_analysis(),
        })
      except Exception as exc:
        cost_analysis.append({"trace_name": trace_name, "error": repr(exc)})

    manifest = {
        "config": args.config,
        "kernel": cfg.kernel or args.config,
        "runner": type(runner).__name__,
        "mode": args.mode,
        "repo": str(repo),
        "branch": _git_value(["git", "branch", "--show-current"]),
        "commit": _git_value(["git", "rev-parse", "HEAD"]),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS"),
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "trace_root": str(trace_root),
        "trace_dir": str(trace_dir),
        "elapsed_s_including_profiler": elapsed_s,
        "input_leaves": _shape_dtype(
            [(name, inputs, kwargs) for name, _, inputs, kwargs in calls]
        ),
        "output_leaves": _shape_dtype(outputs),
        "trace_contract_count": len(trace_specs),
        "cost_analysis": cost_analysis,
    }
    (trace_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return

  inputs_by_ref = runner.make_inputs(jax.random.PRNGKey(args.seed))
  if not inputs_by_ref:
    raise RuntimeError(f"{args.config}: runner.make_inputs returned no inputs")

  # The first input tuple is the public runner call. Some kernels expose
  # additional tuples for individual inner Pallas calls/reference snapshots.
  call_args, call_kwargs = _split_call_inputs(runner, tuple(inputs_by_ref[0]))

  def call(*xs):
    return runner(*xs, **call_kwargs)

  jitted = jax.jit(call)
  for _ in range(args.warmup):
    _block(jitted(*call_args))

  start = time.perf_counter()
  jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=True)
  try:
    with jax.profiler.StepTraceAnnotation(args.step_name, step_num=0):
      with jax.profiler.TraceAnnotation(args.config):
        output = jitted(*call_args)
        _block(output)
  finally:
    jax.profiler.stop_trace()
  elapsed_s = time.perf_counter() - start

  try:
    cost_analysis = jitted.lower(*call_args).compile().cost_analysis()
  except Exception as exc:  # Cost analysis is advisory for Pallas.
    cost_analysis = {"error": repr(exc)}

  manifest = {
      "config": args.config,
      "kernel": cfg.kernel or args.config,
      "runner": type(runner).__name__,
      "repo": str(repo),
      "branch": _git_value(["git", "branch", "--show-current"]),
      "commit": _git_value(["git", "rev-parse", "HEAD"]),
      "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS"),
      "jax_version": jax.__version__,
      "default_backend": jax.default_backend(),
      "devices": [str(device) for device in jax.devices()],
      "trace_root": str(trace_root),
      "trace_dir": str(trace_dir),
      "elapsed_s_including_profiler": elapsed_s,
      "input_leaves": _shape_dtype((call_args, call_kwargs)),
      "output_leaves": _shape_dtype(output),
      "trace_contract_count": len(runner.pallas_trace_args()),
      "cost_analysis": cost_analysis,
  }
  (trace_root / "manifest.json").write_text(
      json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
  )
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()

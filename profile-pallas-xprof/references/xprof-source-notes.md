# XProf Source Notes for Pallas Custom Calls

Use this reference when validating whether XProf is displaying correct FLOPs for a Pallas custom-call under the target repository's documented runtime. These notes are derived from `openxla/xprof` source, especially:

- `xprof/convert/xplane_to_tools_data.cc`
- `xprof/convert/base_op_stats_processor.cc`
- `xprof/convert/xplane_to_op_stats.cc`
- `xprof/convert/xplane_to_op_stats_test.cc`
- `xprof/utils/diagnostics.cc`

## Custom-call visibility flags

Use the compatible default below for current JAX/libtpu environments:

```shell
LIBTPU_INIT_ARGS="--xla_xprof_register_llo_debug_info=true"
```

This is the only profiling flag used by the pinned runtime. Do not add legacy or
version-dependent flags to the batch path.

## Programmatic JAX capture

`docs/jax_profiling.md` documents `jax.profiler.start_trace` / `stop_trace` and emphasizes `block_until_ready()`. The TPU JAX integration test uses:

```python
with jax.profiler.trace(logdir, profiler_options=options):
  for step in range(5):
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      matmul(x, y).block_until_ready()
```

For kernel profiling, prefer one traced step unless the user explicitly wants multiple repeated blocks.

## Cache behavior

`xprof/convert/multi_xplanes_to_op_stats.cc` implements:

```text
ConvertMultiXSpaceToCombinedOpStatsWithCache
  if ALL_HOSTS.op_stats_v2.pb exists:
    cache hit, read binary proto
  else:
    cache miss, convert xplane, write cache file
```

`plugin/xprof/profile_plugin.py` also uses `cache_version.txt`. If the version file is missing or stale, the server can invalidate/recompute caches and overwrite a patched `ALL_HOSTS.op_stats_v2.pb`. After patching a profile cache, write `cache_version.txt` matching the installed `xprof` package version before opening the UI.

The Python helper `xprof.convert.raw_to_tool_data.xspace_to_tool_data` returns
tool JSON/bytes to the caller. Depending on version and invocation path, it may
log that the C++ converter wrote a cache while no `ALL_HOSTS.op_stats_v2.pb`
is left in the run directory. Treat protobuf cache presence as an observation,
not an assumption. If the protobuf is absent, validate Roofline and Op Profile
from returned tool JSON, but do not attempt protobuf patching.

## FLOPs fields used by XProf UI

`xprof/utils/op_utils.cc` populates `OpMetrics` from event data or performance info:

- `flops` and `flops_v2` receive device FLOPs.
- `model_flops` and `model_flops_v2` receive explicit model FLOPs when nonzero.
- If `model_flops == 0`, XProf falls back to device FLOPs.
- `bytes_accessed` is accumulated per occurrence.

`xprof/convert/op_profile_builder.cc` sets UI raw FLOPs from:

```text
op_metrics.model_flops_v2()
```

Therefore, for custom-call correction, patch all of these fields consistently:

- `flops`
- `model_flops`
- `flops_v2`
- `model_flops_v2`
- `bytes_accessed`, if the kernel bytes model is known

Patch values in `ALL_HOSTS.op_stats_v2.pb` as totals across occurrences. If the theoretical value is per custom-call occurrence, multiply by `metric.occurrences`.

## Roofline and Op Profile

`xprof/convert/xplane_to_tools_data.cc` routes `trace_viewer` directly through trace-event conversion, while `overview_page`, `op_profile`, and `roofline_model` go through combined `OpStats`.

`xprof/convert/base_op_stats_processor.cc` preprocesses XSpace with step grouping and derived timeline, converts it to `OpStats`, then combines host-level `OpStats` into `ALL_HOSTS.op_stats_v2.pb`.

`xprof/convert/op_stats_to_roofline_model.cc` aggregates `flops_v2`, `model_flops`, `model_flops_v2`, and `bytes_accessed` from `OpMetrics`. `op_profile_builder.cc` uses `model_flops_v2` as raw FLOPs. If these protobuf fields are wrong, both Roofline and Op Profile are wrong even when the trace has the right device duration.

`xprof/convert/xplane_to_op_stats_test.cc` explicitly treats a custom-call with `flops = 0`, or without a FLOPs stat, as off-duty for custom-call duty-cycle tracking. Therefore a visible active custom-call lane in Trace Viewer is not enough to prove accurate FLOPs/MFU.

## Trace Viewer tooltip

Trace Viewer reads generated `*.trace.json.gz` data for event details. If the custom-call event args contain stale `model_flops`, patch the matching event args:

- `model_flops`
- `bytes_accessed`
- `raw_bytes_accessed`

This patch affects tooltip/details; `ALL_HOSTS.op_stats_v2.pb` affects Roofline and Op Profile.

## Overview page warning

`xprof/convert/xplane_to_op_stats.cc` sets `run_environment.is_training` when XProf's model tracker recognizes a training model. `overview_page_processor.cc` chooses training-vs-inference overview behavior based on `run_environment.is_training`.

For JAX microbenchmarks, `StepTraceAnnotation("train", step_num=...)` may appear in Trace Viewer while Overview still reports:

```text
No step marker observed and hence the step time is unknown.
```

Do not treat Overview Performance Summary as authoritative for custom-call FLOPs if Roofline/Op Profile/Trace disagree. Validate the lower-level `op_stats_v2.pb` and trace event args.

## Correctness standard for new Pallas kernels

For an unknown Pallas kernel, do not trust the UI by default. A profile is considered validated only when:

1. The traced custom-call duration comes from xplane/Trace Viewer and is isolated to the target op.
2. The theoretical per-occurrence FLOPs model is explicit and documented.
3. `ALL_HOSTS.op_stats_v2.pb` contains the same total FLOPs for the target op.
4. Trace event args contain the same per-occurrence `model_flops`.
5. Roofline or Op Profile API reports `rawFlops` / `model_flop_rate` consistent with `FLOPs / duration`.

If condition 2 cannot be met, report timing and XProf raw fields, but do not claim accurate MFU.

For Pallas custom-calls without `pl.CostEstimate`, XProf's inferred FLOPs are
not a source of truth. Add a cost estimate or compute a manual model before
claiming FLOPs/MFU.

Registry-backed profiling can use either the public runner or the lower-level
`pallas_trace_args()` contract. The latter is useful when public `make_inputs()`
contains padded/residual tensors for artifact dumping that the public wrapper
does not accept. This fixes capture mechanics only; FLOPs still require the
same `model_flops_v2`/manual-model validation.

`scripts/pallas_xprof_batch.py` applies this rule automatically:

- Run a remote preflight before online profiling unless `--skip-preflight` is
  passed. The preflight checks SSH command execution, repo path, required
  configured XProf flags, JAX TPU visibility, package versions, registry
  imports, config discovery, and git branch/status. It writes
  `<report-stem>_preflight.json`.
- Prefer detailed `op_profile` custom-call rows over aggregate `name=custom-call` rows.
- Prefer `--expected-models` manual/analyze-kernel FLOPs when available.
- Extract expected FLOPs from runner `cost_analysis["flops"]` only when no manual model is available.
- Mark `xprof_flops_trusted` only when XProf custom-call FLOPs match the chosen expected source within tolerance.
- Mark `xprof_flops_mismatch` when a profile is captured but XProf FLOPs disagree with the runner model.
- Mark `no_expected_flops_model` when timing is captured but there is no independent FLOPs model.
- Mark `xprof_flops_missing` when timing is captured but XProf reports no custom-call FLOPs.
- Write `<report-stem>_summary.json` with status, FLOPs validation, and failure-class buckets.
- Support `--write-expected-model-template` to generate an editable expected
  model JSON file. Existing manual models are preserved; configs that still
  lack a manual/analyze-kernel model use `"flops": null`.
- Support CI gates: `--require-all-profiled`, `--require-flops-trusted`, and
  repeatable `--allow-failure-class`. An allowed failure class means the
  failure is classified and expected in the current environment; it does not
  mean a TPU profile exists for that config.
- Use `xprof_pallas_tools.py readiness` after starting XProf to verify that
  local xplane runs are visible through the XProf `/runs` API and to combine UI
  visibility with batch summary gates. This prevents confusing "profile
  downloaded" with "profile is openable and FLOPs are trusted".

Treat `cost_analysis["flops"]` as a cross-check, not an absolute oracle for
Pallas. For standard JAX ops it is usually a good FLOPs source. For Pallas and
composite wrappers, it can be missing, partial, or inconsistent with useful
algorithmic FLOPs. When XProf and cost analysis disagree, escalate to
`analyze-kernel`: inspect the kernel body, enumerate matmuls/reductions, compute
manual FLOPs, then combine that manual FLOPs value with xplane device duration.

Manual expected-model files should use this schema:

```json
{
  "kernels": {
    "config_or_kernel_name": {
      "flops": 690114396160,
      "bytes": 807411776,
      "source": "manual/analyze-kernel",
      "note": "per custom-call occurrence; include shape convention",
      "tolerance": 0.1
    }
  }
}
```

The batch script matches expected models by config name, manifest kernel name,
then runner name. Manual models override `cost_analysis()` in validation because
they are the only source that can encode the intended useful-FLOPs convention.
TODO template entries intentionally use `"flops": null`; validation ignores
them until `analyze-kernel` replaces them with a manual number.

If a kernel fails before profiling with:

```text
'aic.block_matmul' op created with unregistered dialect
```

classify it as `kernel_lowering_unregistered_aic_block_matmul`. This is not an
XProf capture/download problem; no valid TPU profile can be produced until the
JAX/libtpu lowering path accepts that dialect or the kernel is rewritten to an
executable primitive for the current environment.

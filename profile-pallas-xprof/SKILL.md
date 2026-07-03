---
name: profile-pallas-xprof
description: "Capture a remote TPU XProf profile for a JAX/Pallas kernel, download it locally, generate/open the local XProf UI, and place artifacts under tmp/{kernel_name}_{date}/experiments/{method_name}/results/xprof. Use when asked to run xprof, capture a TPU profile, open XProf locally, inspect trace timing, or fetch remote profile dumps."
---

# Profile Pallas XProf

Use this skill for the lightweight XProf workflow: capture remote TPU profile, download locally, prepare/open local UI, and report a usable URL plus artifact paths.

## Standard Executable Workflow

For registry/config-driven kernels, use the bundled orchestrator instead of reconstructing SSH, tar, scp, cache, and UI commands:

```shell
python scripts/xprof_workflow.py \
  --host <ssh-host> \
  --remote-repo <remote-repo> \
  --remote-python <remote-python> \
  --workspace-root tmp/<kernel>_<date> \
  --method <experiment-method> \
  --config <kernel-config> \
  --local-python <python-with-xprof> \
  --port <port> \
  --require-all-profiled
```

The script runs remote preflight/profile capture through `pallas_xprof_batch.py`, downloads and extracts artifacts, generates cache, starts XProf, checks readiness, and writes `xprof_workflow_status.json` plus `xprof_ui_status.md` under the standard experiment path.

Use a custom profile runner only when the repository has no registry/config contract. It must accept an explicit output directory, print shape/dtype/kernel/profile path, warm up before tracing, and terminate after a deterministic number of iterations.

## Required Output Location

Write XProf artifacts under the method being profiled:

```text
tmp/{kernel_name}_{YYYYMMDD}/experiments/{method_name}/results/xprof/
```

Keep raw profile directories, `.tgz` archives, generated cache files, server logs, and XProf-derived JSON under that `xprof/` directory. Do not create new top-level `results/xprof/` artifacts for standard kernel projects.

## Workflow

1. Locate SSH host, remote repo, remote Python environment, kernel runner, local project root, and output root. Recheck remote branch/status and runtime versions on every new instance.
2. Reuse an existing benchmark/profile runner when available.
3. If creating a temporary runner, set profiling flags before importing JAX.
4. Run warmup, then trace a small deterministic number of iterations.
5. Print the remote trace directory, shape, dtype, and kernel name.
6. Tar the remote profile before downloading when practical.
7. Store downloaded artifacts under `experiments/{method_name}/results/xprof/`.
8. Discover a usable local XProf runtime before declaring UI unavailable.
9. Generate local XProf cache when the local environment has `xprof`; if cache generation fails, still attempt to start the UI from the raw profile when possible.
10. Start a local XProf server on an available port whenever a local profile exists. Do not silently skip UI startup.
11. Check that a run is visible.
12. If raw profile download is slow or fails, keep the remote raw path, parse `*.trace.json.gz` remotely into a small JSON summary, and write `xprof_ui_status.md`.
13. Update `docs/results.md` and `experiments/{method_name}/README.md`.

Do not report success from process exit alone. Require a local `.xplane.pb`, a `trace.json.gz` when the backend emits one, and XProf readiness visibility.

## Local XProf Discovery Order

Use this order before reporting that XProf UI cannot be opened:

```text
1. `where xprof` or `Get-Command xprof`
2. repository `.venv/Scripts/xprof.exe` on Windows
3. repository `.venv/bin/xprof` on Linux/macOS
4. Python environments whose `import xprof` succeeds
5. explicit `--xprof-exe` path if known from prior runs
```

If a non-default Python is needed, run the bundled helper with that Python, not system Python.

## Raw Profile Fallback

When a profile exists remotely but cannot be downloaded quickly:

```text
record remote trace root and tarball path
record the failed download command and observed behavior
parse remote `*.trace.json.gz` into a small local JSON summary
write `results/xprof/xprof_ui_status.md`
continue the tuning decision only if the summary is sufficient
do not claim local XProf UI is available
```

Required environment pattern before importing JAX:

```shell
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
```

Use bundled helper scripts when present:

```shell
python ~/.codex/skills/profile-pallas-xprof/scripts/xprof_pallas_tools.py generate-cache --profile-dir <local_profile_root>
python ~/.codex/skills/profile-pallas-xprof/scripts/xprof_pallas_tools.py start-xprof --profile-dir <local_profile_root> --port <port>
python ~/.codex/skills/profile-pallas-xprof/scripts/xprof_pallas_tools.py api-check --port <port> --run <run> --host <host>
```

## User-Facing Result

Reply in Chinese. Use a structured result with these fields:

```text
Conclusion:
  profile captured/opened, or profile captured but UI failed

XProf UI:
  local URL
  server pid or port
  visible/readiness status
  if failed: exact missing dependency/error and recovery command

Profile artifacts:
  local profile path
  tarball path if available
  profile timestamp or run name
  kernel name

Timing summary:
  short timing summary if directly readable

Docs:
  docs updated
```

If an XProf server is already running and usable, do not restart it unless needed. Report the existing URL and artifact path.

If UI startup fails after a valid profile was downloaded, write a short failure note under `results/xprof/` and still report the raw profile path. Typical failure causes are:

```text
xprof executable not on PATH
local Python environment cannot import xprof
port already occupied
profile path is incomplete or has no *.xplane.pb
```

## Optional Deep Analysis

Only do these when requested or when the profile is suspicious:

```text
inspect op_stats_v2.pb
validate Pallas custom-call FLOPs
patch XProf cache
compare CostEstimate, manual FLOPs, and XProf counters
generate a batch report
```

For FLOPs/MFU conclusions, prefer `$analyze-kernel` to build the manual model.

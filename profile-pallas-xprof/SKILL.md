---
name: profile-pallas-xprof
description: "Capture a remote TPU XProf profile for a JAX/Pallas kernel, download it locally, generate/open the local XProf UI, and place artifacts under tmp/{kernel_name}_{date}/experiments/{method_name}/results/xprof. Use when asked to run xprof, capture a TPU profile, open XProf locally, inspect trace timing, or fetch remote profile dumps."
---

# Profile Pallas XProf

Use this skill for the lightweight XProf workflow: capture remote TPU profile, download locally, prepare/open local UI, and report a usable URL plus artifact paths.

## Required Output Location

Write XProf artifacts under the method being profiled:

```text
tmp/{kernel_name}_{YYYYMMDD}/experiments/{method_name}/results/xprof/
```

Keep raw profile directories, `.tgz` archives, generated cache files, server logs, and XProf-derived JSON under that `xprof/` directory. Do not create new top-level `results/xprof/` artifacts for standard kernel projects.

## Workflow

1. Locate SSH host, remote repo, remote Python environment, kernel runner, local project root, and output root.
2. Reuse an existing benchmark/profile runner when available.
3. If creating a temporary runner, set profiling flags before importing JAX.
4. Run warmup, then trace a small deterministic number of iterations.
5. Print the remote trace directory, shape, dtype, and kernel name.
6. Tar the remote profile before downloading when practical.
7. Store downloaded artifacts under `experiments/{method_name}/results/xprof/`.
8. Generate local XProf cache and start a local server on an available port.
9. Check that a run is visible.
10. Update `docs/results.md` and `experiments/{method_name}/README.md`.

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

Reply in Chinese and include:

```text
local XProf URL
local profile path
profile timestamp or run name
kernel name
whether the run is visible
short timing summary, if directly readable
docs updated
```

If an XProf server is already running and usable, do not restart it unless needed. Report the existing URL and artifact path.

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

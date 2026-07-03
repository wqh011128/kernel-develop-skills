---
name: profile-pallas-xprof
description: "Capture a remote TPU XProf profile for a JAX/Pallas kernel, download and validate artifacts, open or verify a local XProf UI, and report usable paths and readiness. Use when component timing, communication overlap, short-kernel device timing, or profiler evidence is needed; not as a mandatory step for every kernel change."
---

# Profile Pallas With XProf

Use the existing repository runner when available. For registry/config-driven kernels, prefer the bundled workflow:

```shell
python scripts/xprof_workflow.py \
  --host <ssh-host> --remote-repo <repo> --remote-python <python> \
  --workspace-root <authorized-local-workspace> --method <method> \
  --config <config> --local-python <python-with-xprof> --port <port>
```

Before capture, read the target repository README and use its documented environment setup, then recheck remote branch/status, device availability, and applicable `AGENTS.md`. Do not manually install or upgrade dependencies from this skill. Set required profiling flags before importing JAX. Warm up, trace a deterministic iteration count, synchronize device work, and print shape/dtype/kernel/profile paths.

A successful command is not a successful profile. Require the expected local `.xplane.pb` or explicitly record the remote artifact; verify XProf server readiness and run visibility before reporting an open UI. Keep tarballs, cache, server logs, status, and derived JSON together in the authorized experiment artifact location.

If download or UI fails, preserve the remote trace path, capture a small remote trace summary when possible, write the exact dependency/port/artifact failure, and provide a recovery command. Do not claim the UI is available.

Use `scripts/xprof_pallas_tools.py` for cache, UI, API, and readiness operations. Use `$analyze-kernel` for FLOPs/MFU and bottleneck conclusions; XProf counters do not override the source-level mathematical model.

Report capture status, local URL and visible run when available, local/remote artifact paths, timing facts directly observed, and exact recovery steps for partial failure.

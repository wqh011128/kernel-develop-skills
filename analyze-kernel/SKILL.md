---
name: analyze-kernel
description: "Analyze correctness and performance evidence for JAX/Pallas kernels on TPU/GPU. Use to compare candidates, validate a tuning claim, model FLOPs/bytes/MFU, interpret benchmark or XProf data, rank compute, memory, communication, control, and host bottlenecks, and propose falsifiable next checks. Adapts report depth to available evidence."
---

# Analyze Kernel Evidence

Analyze only claims supported by available artifacts; do not require XProf for a simple correctness diagnosis or fabricate unavailable counters.

1. Confirm semantics, shapes, dtypes, layout/mask/padding, trusted reference, and correctness status.
2. Verify benchmark comparability: device, shape, dtype, warmup, iterations, synchronization, baseline, and distribution—not a single favorable number.
3. Build a source-level FLOPs/bytes/communication model when it can distinguish hypotheses. Treat Pallas/XProf FLOPs as untrusted until cross-checked.
4. When profiles exist, inspect full-device time, custom calls, collectives, copies/layout, HBM/DMA, VMEM spill/fill, MXU/ALU, launch/control, and host dispatch as relevant.
5. Classify and rank bottlenecks with evidence. Separate observed facts, possible causes, next discriminating checks, and proposed experiments.

Use bundled deterministic helpers when applicable:

```text
scripts/correctness.py
scripts/benchmark_utils.py
scripts/xplane_parser.py
```

Read `references/diagnostic-matrix.md` for an unfamiliar profiler phenomenon. Read hardware specs or report format only when needed; do not load every reference by default.

For a bounded answer, report directly. For research mode or a durable performance decision, write a concise artifact in the current experiment and link raw evidence. Required fields are conclusion, contract, correctness, comparable benchmark/profile facts, cost model when used, bottleneck ranking, next falsifying check, and artifact paths.

If a candidate improves local custom-call time but regresses full-device time, reject the performance claim unless the user explicitly chooses that trade-off. Promote this or another repeated confirmed failure through `$kernel-foundry`, not by duplicating prose here.

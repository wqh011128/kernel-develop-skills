---
name: kernel-goal-discovery
description: "Research and clarify a requested kernel before implementation. Use when a developer asks to implement, port, optimize, or design any kernel, especially JAX/Pallas/TPU/GPU kernels. Search official docs, papers, current repository code, and existing implementations, then ask the user in Chinese to confirm the final goal before coding."
---

# Kernel Goal Discovery

Use this skill when the user wants to write, port, design, or optimize a kernel and the final target is not fully specified.

## Core Rule

Do not start implementation until the final goal is confirmed. First research, then ask concise Chinese confirmation questions.

## Research Order

Prefer:

```text
current repository code, tests, benchmarks, registry, docs
official framework/backend/hardware documentation
papers or algorithm notes for semantics and numerical stability
readable open-source implementations
RFCs, issues, or PRs only as background, not as proof
```

If using web sources, cite links in the response. Do not treat a non-official implementation as the only source of truth.

## Final Goal Checklist

Ask the user to confirm:

```text
1. Operator semantics:
   formula, masks, causal behavior, broadcasting, layout, padding, boundary cases

2. Target scenario:
   prefill, decode, training, inference, forward, backward, fused op, or microbenchmark

3. Hardware/backend:
   TPU/GPU/CPU generation and JAX/Pallas/Triton/CUDA/XLA or other backend

4. Shapes:
   target shape family, static/dynamic dimensions, edge cases

5. Dtypes and accumulation:
   input dtype, output dtype, accumulator dtype, quantization or rounding rules

6. Correctness reference and tolerance:
   dense JAX, existing kernel, official implementation, mathematical oracle, atol/rtol/cosine/relative L2

7. Performance baseline and target:
   current baseline, target latency, speedup, MFU, bandwidth, memory footprint, or utilization goal

8. Memory constraints:
   HBM, VMEM/shared/register pressure, spill, padding, workspace, intermediate tensor limits

9. Integration scope:
   standalone kernel, registry integration, model loop, inference service, autograd, CI

10. Tests, benchmark, and report deliverables:
    correctness tests, benchmark, XProf, analyze-kernel report, docs, experiment records
```

## Output Format

Reply in Chinese:

```text
我先确认目标，不开始实现。

已查到的关键信息:
- ...

需要你确认的最终目标:
1. ...
2. ...

默认建议:
- 如果你没有特别要求，第一版建议先做 ...
```

If reasonable defaults exist, state them explicitly and explain the risk. If the user says to use defaults, write those defaults into the later design docs.

## Entry To Next Stage

Proceed to `$kernel-design-docs` only after the user confirms the goal or explicitly accepts the defaults.

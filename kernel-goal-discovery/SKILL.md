---
name: kernel-goal-discovery
description: "在实现或优化 JAX/Pallas/TPU/GPU kernel 前，只解决会改变方案方向的关键未知项。当算子语义、目标 workload、shape、dtype、reference、容差、硬件、性能目标、内存限制或集成范围不清楚时使用。先读取真实仓库契约，并产出精简的 confirmed/inferred/unknown 算子契约。"
---

# Kernel Goal Discovery

Do not turn discovery into a fixed questionnaire.

1. Read applicable `AGENTS.md`, current operator/reference/tests/configs, call sites, and similar kernels in the actual checkout.
2. Use official framework/backend documentation and papers only for facts not established by the repository. Cite external sources when used.
3. Produce a compact contract with three labels: `confirmed`, `inferred`, and `unknown`.
4. Cover only material fields: formula and masks; target scenario; shape family; input/output/accumulator dtype; trusted oracle and tolerance; hardware/backend; performance objective and baseline; memory/topology constraints; integration and delivery scope.
5. Ask the user only about unknowns whose alternatives would materially change semantics, architecture, cost, or delivery. State safe defaults for the rest.

Do not implement until semantic unknowns and the trusted oracle are resolved. Performance targets may remain provisional if the user explicitly asks for a baseline-first implementation.

Pass the confirmed contract directly to `$kernel-design-docs` for a non-trivial design or `$implement-kernel-from-plan` for a bounded known-pattern change. Do not create documents merely to record facts already authoritative in repository code/tests.

Reply in Chinese with confirmed facts, consequential inferences, blocking unknowns, and the next stage.

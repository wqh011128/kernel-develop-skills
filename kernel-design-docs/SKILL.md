---
name: kernel-design-docs
description: "Create a complete Chinese design-document package for a new or modified kernel under tmp/{kernel_name}_{date}/docs. Use after kernel goals are confirmed and before implementation. Produces README, a full RFC that includes the development plan, mathematical derivation, results plan, pitfalls, implementation notes, and optimization notes, while enforcing the standard experiments layout."
---

# Kernel Design Docs

Use this skill after the kernel goal has been confirmed. Do not start implementation from this skill unless the user explicitly asks to continue.

## Required Output Layout

Create or update:

```text
tmp/{kernel_name}_{YYYYMMDD}/docs/
  README.md
  rfc.md
  math.md
  results.md
  fail-notes.md
  impl-notes.md
  optimization.md
```

Also create the experiment root if it does not exist:

```text
tmp/{kernel_name}_{YYYYMMDD}/experiments/
```

Rules:

```text
kernel_name: lowercase snake_case.
README.md: always uppercase.
Use exactly the docs above for new work.
Do not create `develop-plan.md` for new work; the plan belongs inside `rfc.md`.
Do not create old numbered docs such as 00-rfc.md.
Do not create a top-level results/ directory for new artifacts.
```

## Research Requirement

Before writing `math.md`, research enough primary or high-signal sources to verify the operator semantics:

```text
current repository implementations, tests, benchmarks, and registry
official framework/backend documentation
papers or algorithm notes when relevant
at least one readable open-source implementation unless the op is project-private
```

If math is uncertain, mark the uncertainty and ask the user or add a validation gate. Do not present uncertain math as a conclusion.

## Document Responsibilities

## Document Writing Contract

All generated kernel docs must be written in Chinese by default, encoded as UTF-8, and structured for future agents to continue work without reading raw logs first. Keep code identifiers, API names, metric names, file names, commands, and official terms such as `XProf`, `FLOPs`, `LSE`, `HBM`, and `VMEM` unchanged when translating would reduce precision.

Use these rules for every doc:

```text
Use stable headings with numbered sections when the doc is long.
Put the conclusion or current status before details.
Separate status, facts, evidence, decisions, and next actions.
Use tables for matrices and short code blocks for commands, paths, shapes, and metrics.
Keep raw logs out of docs; link artifact paths instead.
When a result changes, update both the experiment README and the relevant top-level summary doc.
Do not mix design, results, failure notes, and optimization history in one doc.
Do not leave mojibake or mixed-language placeholder text in final docs.
不要写按时间顺序堆叠的流水账。文档必须按目标、合约、证据、决策、下一步组织。
```

Apply this per-section writing logic:

```text
Status sections:
  state what is accepted, rejected, unverified, or blocked.
  include the latest known implementation and artifact path.

Evidence sections:
  cite commands, shapes, tolerances, benchmark medians, XProf/report paths, or source references.
  never claim speedup, correctness, MFU, memory bottleneck, or overlap without an artifact.

Decision sections:
  record the decision, the reason, and the accept/reject condition that was used.
  do not hide neutral or rejected experiments if they changed the next plan.

Next-action sections:
  list ordered hypotheses or tasks.
  each task must have a validation gate and a rejection condition when it affects performance.

Reference sections:
  distinguish repository facts, official docs, papers, and inferred conclusions.
  mark unresolved math or backend behavior as uncertainty, not fact.
```

Use these content boundaries:

```text
README.md is a navigation and status index; it is not a design proposal or raw result report.
rfc.md is the high-level contract and plan; it is not an experiment log.
math.md is the semantic proof; it is not a performance tuning note.
results.md is the evidence summary; it is not an optimization diary.
optimization.md is the hypothesis/decision loop; it is not a place for raw benchmark dumps.
impl-notes.md is the code/layout/API boundary record; it is not a user-facing README.
fail-notes.md is for concise reusable pitfalls within this kernel workspace; it is not a full failure transcript.
```

`README.md`:

```text
Use this Chinese structure:

# <kernel> 文档索引

## 1. 当前状态
current best, correctness status, performance status, active XProf URL/path

## 2. 范围
in scope, out of scope, hard constraints

## 3. 文档地图
one table mapping each doc to its purpose

## 4. 实验索引
one table: experiment, status, purpose, key artifact/report

## 5. 当前最佳实现与下一步
current implementation, why it is current best, next hypothesis

## 6. 历史迁移说明
only if legacy artifacts exist
```

`rfc.md`:

```text
用中文撰写。使用下面的固定 RFC 结构。RFC 导出到独立 RFC 仓库时使用文件名格式 `<number>-<component>-<short-name>.md`；在 kernel workspace 内仍固定写入 `docs/rfc.md`。

# RFC 0000: <RFC 标题>

文件名格式：`<number>-<component>-<short-name>.md`

| 字段 | 内容 |
| --- | --- |
| Status | Draft / In Review / Accepted / Rejected / Superseded |
| Owner | <负责人> |
| Reviewers | <评审人> |
| Created | YYYY-MM-DD |
| Updated | YYYY-MM-DD |
| Target | <目标版本/里程碑/交付窗口> |

## 1. Summary
用 2-4 段说明本 RFC 要解决什么问题、提议新增或修改什么能力、为什么现在需要做，以及实现应贴合哪些现有代码边界或组织边界。

## 2. 问题陈述
先描述问题本身，不要先描述方案。必须包含下表：

| 工作域 | 现状痛点 | 业务影响 |
| --- | --- | --- |

然后列出具体问题。

## 3. Context
### 3.1 Current Behavior / Architecture
说明当前主流程、关键模块职责、数据流、控制流，以及需要继续复用的既有能力。

### 3.2 Relevant Background and Constraints
列出理解本 RFC 必须知道的背景、依赖、外部系统限制和顺序约束。

### 3.3 Technical Environment
列出运行时、依赖版本、包管理/构建工具、测试框架、代码风格和 CI 约束。

### 3.4 Positioning
给关键概念定边界，并明确这些概念不表示什么，避免 Review 时发生定义争议。

## 4. Goals
每个 Goal 只描述一个可验收结果。每个目标应包含定位、逻辑或规则、必要时的启用方式，以及验收标准。

### Goal 1 - <目标名>
定位：
逻辑：
启用方式：
验收标准：

### Goal 2 - <目标名>
定位：
规则：
验收标准：

### Goal 3 - <兼容性/迁移/稳定性目标>
定位：
约束：

## 5. Non-Goals
明确本 RFC 不做什么。Non-Goals 必须具体，避免实现阶段范围扩大。

## 6. Proposed Design
### 6.1 Responsibility Matrix
使用包含 `#`、`职责`、`优先级`、`频率`、`交付物` 的表格。

### 6.2 Flow Design
描述新流程。优先用短代码块表达控制流，再补充关键约束和设计原则。

### 6.3 Configuration / API Design
列出新增或修改的配置、CLI、环境变量、函数签名或公开接口，并说明优先级与校验规则。

### 6.4 Data Model / Core Rules
描述核心数据结构、转换规则、不变量和状态契约。

### 6.5 Module-Level Changes
使用包含 `模块`、`变更`、`边界` 的表格。

### 6.6 Compatibility
说明默认行为、旧配置、旧接口、旧产物格式是否保持兼容，并说明回滚或降级方式。

### 6.7 Observability / Output Artifacts
列出日志、导出文件、指标、错误信息、调试产物及其路径或接口。

## 7. Interfaces / Contracts
使用包含 `接口`、`说明`、`Contract` 的表格。

## 8. Alternatives Considered
每个备选方案都必须包含 Pros、Cons，以及采用或不采用的结论。

## 9. Risks / Trade-offs
使用包含 `风险`、`影响`、`缓解` 的表格。

## 10. Validation / Testing Plan
使用包含 `信号`、`成功标准`、`度量方式` 的表格。必要时列出新增测试文件和手动验证命令。

## 11. Rollout / Migration Plan
使用包含 `阶段`、`动作`、`时间`、`依赖` 的表格。说明迁移策略、启用方式、不兼容配置处理和回滚方式。

## 12. Task Breakdown
使用包含 `Owner`、`任务域`、`范围`、`交付物` 的表格，并说明任务切分规则。

## 13. Open Questions
列出需要用户、reviewer 或外部系统确认的未决问题。

## 14. Decision Log
使用包含 `日期`、`决策`、`决策人`、`备注` 的表格。
```

RFC 是唯一高层计划文档。阶段、go/no-go gates、验收标准、回滚/拒绝条件、owner 和决策历史必须放入第 10-14 节，不要再复制到单独 plan 文档。

`math.md`:

```text
Use this structure:

# 数学语义：<kernel>

## 1. 语义来源与可信边界
list source of truth: official docs, papers, repository reference, framework reference such as PyTorch/JAX, and tests.
state which implementation is the correctness reference.
state which parts are derived by the agent and must be validated.
never use the optimized kernel as its own reference.

## 2. 符号与 Shape
symbol definitions and input/output shapes.

## 3. 全局语义
complete mathematical formula

## 4. 局部 / 分块 / 分布式语义
partitioning, block equations, rank equations

## 5. 等价性证明
prove global semantics equals local/block/rank semantics

## 6. Mask、Padding 与边界
causal, padding, sequence boundary, invalid element behavior

## 7. Dtype 与数值稳定性
input dtype, accumulator dtype, LSE/softmax rules, tolerance expectations

## 8. Reference 伪代码
minimal executable-style reference logic.
if PyTorch/JAX/framework semantics are used, describe the extraction or equivalence path.

## 9. 数据流
diagram or stepwise data flow when useful

## 10. 必测边界条件
edge cases that must be tested
```

For attention, scan, reduction, or distributed kernels, explicitly prove the equivalence between global semantics and local/block/rank semantics.

Math correctness gate:

```text
Do not treat math.md as complete unless it answers:
  What is the source of truth?
  Is the reference implementation dense/framework/JAX/PyTorch or project-local?
  How are local/block/rank results merged back to the global formula?
  Which mask, padding, dtype, accumulation, and LSE/normalization rules are required?
  Which tests would falsify the derivation?
```

`results.md`:

```text
Use this Chinese structure:

# 结果汇总：<kernel>

## 1. 当前结论
accepted current best, rejected paths, and whether performance claims are proven

## 2. Correctness 矩阵
table: experiment, command, shapes, tolerance, status, artifact

## 3. Benchmark 摘要
table: experiment, shape, baseline, target, median, speedup, artifact

## 4. XProf / 分析摘要
local URL, profile path, analysis report, bottleneck class, key component movement

## 5. 当前最佳实现
current best implementation and why

## 6. 未关闭的验证缺口
missing shapes, missing profiles, unproven claims
```

`fail-notes.md`:

```text
Use this Chinese structure:

# 踩坑记录：<kernel>

## 1. 踩坑索引
table: pitfall, affected experiment, status, short lesson

## 2. 已拒绝方向
one section per rejected direction: hypothesis, evidence, shortest root cause, what not to repeat

## 3. Correctness 失败
only understood failures; include symptom, cause, fix

## 4. 本 kernel 可复用经验
short lessons useful for this kernel workspace

## 5. Skill 候选经验
table: lesson, evidence, scope, generalizable/current-kernel-only, promotion decision
```

Do not paste long raw logs here. A lesson may be promoted to a skill only when it is supported by backend/API semantics, official/project contracts, or repeated evidence beyond one shape and one implementation. Shape-specific thresholds, paths, one-off compiler failures, and kernel-specific schedules remain in this workspace.

`impl-notes.md`:

```text
Use this Chinese structure:

# 实现记录：<kernel>

## 1. 文件与 API 边界
new files, untouched files, public/experimental APIs

## 2. 数据布局与 Dtype
physical layout, padding, dtype and accumulator choices

## 3. 通信 / Kernel 分层
what is done by framework collectives and what is done by Pallas/local kernels

## 4. 实现变体
accepted and experimental APIs, status, and integration state

## 5. 已知约束
hard-coded assumptions, unsupported cases, compile/runtime caveats
```

`optimization.md`:

```text
Use this Chinese structure:

# 优化记录：<kernel>

## 1. Baseline 基线
stable baseline, target shapes, metrics, artifact paths

## 2. 瓶颈分类
roofline class, XProf evidence, component ranking

## 3. 已接受优化
one section per accepted change: hypothesis, evidence, decision, retained code path

## 4. 已拒绝 / 中性优化
one section per rejected change: hypothesis, evidence, reason, what not to repeat

## 5. 当前假设队列
ordered next experiments with acceptance and rejection conditions

## 6. 流程备注
experiment process notes that affect next-step thinking
```

## Output To User

Report in Chinese:

```text
docs root
created/updated docs
unresolved questions
recommended next stage
```

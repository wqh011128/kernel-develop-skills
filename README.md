# Kernel Develop Skills

这是一套面向 JAX/Pallas/TPU/GPU kernel 的工程 skill。它把 kernel 工作分成三类能力：

- Builder：实现正确、可集成的新 kernel；
- Investigator：根据 correctness、benchmark、HLO、XProf 定位问题；
- Learning：把已验证的失败沉淀为 guardrail、fuzz case 和 replay eval。

目标 kernel 仓库的 README、`AGENTS.md`、CI、测试和配置始终是最高权威。Skills 不替代这些约束，也不自行安装或升级依赖。

## 使用方式

### 只需要调用一个入口

用户通常只需要说：

```text
$kernel-dev-lifecycle
```

它会自动判断并调用其他 skill。不要要求自己手动串联 goal discovery、RFC、implementation、analysis、XProf、tuning 和 Foundry。

### 通用填空模板

复制下面的模板，替换 `<...>` 后直接发送：

```text
请使用 $kernel-dev-lifecycle，在以下仓库执行任务：

仓库：<目标 kernel 仓库绝对路径或远程地址>
分支：<branch>
任务类型：<开发新 kernel / 修复 correctness / 优化性能 / 诊断问题>
Kernel 或算子：<名称>

目标：
- <希望最终得到的结果>

已知信息：
- 输入/输出语义：<...>
- shape：<...>
- dtype：<...>
- trusted reference：<路径、命令或“未知，请调查”>
- correctness tolerance：<...>
- 测试入口：<路径、命令或“未知，请调查”>
- 当前 baseline：<命令和结果，或“不适用”>
- HLO/XProf/artifact：<路径，或“不适用”>

产物目录：<用户指定的绝对路径；如果留空，默认使用目标仓库内的 <kernel>/docs/>

约束：
- 是否允许修改代码：<是/否>
- 是否允许 commit：<是/否，默认否>
- TPU/时间/实验预算：<...>
- 其他必须遵守的边界：<...>

执行要求：
- <例如：先调查不要改代码 / 直接实现 / 最多做 N 个优化实验>
```

不知道的字段不要猜，填写“未知，请调查”。Agent 会先读取目标仓库 README、适用的 `AGENTS.md`、现有实现、测试、配置和 CI，再决定下一步。

所有 RFC、设计文档、失败记录、research state、benchmark、HLO 和 XProf 说明都写入同一个产物目录：优先使用用户填写的目录；未填写时默认是目标仓库内的 `<kernel>/docs/`。绝不写入本 skills 仓库。

## 最常用的四种请求

### 1. 开发新 kernel

推荐先计划、后实现：

```text
请使用 $kernel-dev-lifecycle 开发 <kernel>。
先不要修改代码。请先读取 README、AGENTS.md、类似 kernel、reference、测试、registry/config 和 CI，输出：
1. confirmed / inferred / unknown 的 operator contract；
2. correctness reference、tolerance 和 shape matrix；
3. RFC 是否必要以及需要填写的内容；
4. 实现、验证和交付计划；
5. 需要我确认的决策。
```

确认计划后发送：

```text
计划确认。请按 standard 模式实现，并完成 correctness、必要的 benchmark/HLO 验证、CI 检查和交付报告。不要 commit。
```

### 2. 修复 correctness

```text
请使用 $kernel-dev-lifecycle 修复 <kernel> 的 correctness 问题。

复现命令：<command>
candidate/reference 差异：<...>
shape/dtype：<...>

请先复现并定位根因。correctness 通过前不要做性能结论，不要 commit。
```

### 3. 优化性能

必须提供已通过 correctness 的证据和稳定 baseline：

```text
请使用 $kernel-dev-lifecycle 优化 <kernel>。

correctness 命令和结果：<...>
baseline 命令、shape、warmup、iterations、full-device latency：<...>
怀疑瓶颈：<VMEM spill / layout copy / launch / communication / MXU / 未知>
目标：<例如 full-device latency 从 A 降到 B>
预算：最多 <N> 个实验、<T> TPU 小时

请先诊断瓶颈，再用 tuning loop 验证可归因假设。每次修改都重跑 correctness 和完整 latency。不要 commit。
```

单个假设使用 `single`；多个假设、portfolio 或自动探索使用 `research`。

### 4. 诊断已有 kernel

```text
请使用 $analyze-kernel 诊断 <kernel>。

benchmark：<path>
correctness：<path 或 command>
HLO：<path>
XProf：<path，若有>

请分别输出：观察事实、可能瓶颈、证据不足之处、下一项可证伪检查。不要修改代码。
```

只有需要组件耗时、通信重叠或 profiler 证据时，才使用 `$profile-pallas-xprof`。

## Agent 会自动完成什么

在允许修改代码的任务中，lifecycle 会按风险选择：

| 模式 | 适用任务 | 主要动作 |
| --- | --- | --- |
| `quick` | 语义和 reference 已知的局部修复 | 读取约束、修改、目标 correctness、项目检查 |
| `standard` | 新 kernel、移植、语义/API 变化 | 契约、必要 RFC、实现、correctness matrix、性能证据 |
| `research` | 多假设调优、portfolio、causal、genome | 稳定 baseline、实验预算、状态机、Pareto 结果 |

完成时应返回：修改文件、约束来源、correctness 结果、性能证据、CI 结果、IR-upload tag（如适用）、未解决风险、未提交的 commit message 草稿，以及 learning checkpoint 结果。

除非用户明确授权，Agent 不会 commit、push 或创建 PR。

## 经验如何自动沉淀

当前设计不是把每次聊天直接追加到 `SKILL.md`，而是分两层：

```text
任务结束
  ↓
自动扫描失败、回归、阻塞和人工纠正
  ↓
生成 local / candidate / confirmed 学习记录
  ↓
confirmed -> guardrail check -> replay/eval
  ↓
有统计收益后，才提出共享 skill/helper 的 review diff
```

- `local`：只影响当前 kernel，留在当前实验记录；
- `candidate`：可能可复用，但证据不足，只生成 failure record；
- `confirmed`：有复现、有根因、有 passing control，才编译为 guardrail 并加入 replay。

共享 `SKILL.md` 不会在任务中被静默修改。这样可以自动积累证据，又避免一次偶然失败污染所有后续 kernel。

Foundry 入口：

```shell
python kernel-foundry/scripts/kernel_foundry.py <capability> <command> ...
```

可用能力包括 `guardrail`、`fuzz`、`research`、`replay`、`portfolio`、`causal` 和 `genome`。

## 交付门禁

最终验收必须重新读取适用的 `AGENTS.md`，检查最终 diff/worktree，并核对目标仓库的 GitHub workflow、pre-commit、Ruff、typing、tests、配置校验、correctness、HLO/IR artifact 和 IR-upload tag。

可使用：

```shell
python kernel-dev-lifecycle/scripts/kernel_delivery_gate.py \
  --repo <repo> --kernel <kernel> --config <config> --test <test> \
  --snapshot-root <snapshot> --commit-message <draft.txt> \
  --pr-text <pr-or-squash-message.txt> --run \
  --json-out <delivery-gate.json>
```

提交信息草稿必须包含：

```text
feat[TOOL]: <imperative summary>

Task:
- <任务>

Solution:
- <方案>

Test:
- <精确命令和结果>

JIRA: COMPIL-XXXX
```

`COMPIL-XXXX` 只是占位符，真正提交前必须替换。生成草稿不等于获得 commit 授权。

## 修改 skill 后的验证

```shell
python kernel-foundry/scripts/validate_family.py
```

它会检查 Python 语法、skill frontmatter、RFC/提交模板关键结构和所有 helper 回归测试。

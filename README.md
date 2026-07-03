# Kernel Develop Skills

一套面向 JAX/Pallas/TPU/GPU kernel 的工程能力层。目标不是让模型遵守更多文字，而是让它同时成为：

- **Builder**：从已确认的算子契约实现正确、可集成的新 kernel。
- **Investigator**：从 correctness、benchmark、HLO 与 XProf 证据定位问题。
- **Learning system**：把已确认失败编译成 guardrail、fuzz case 和 replay eval，让下一次开发可测量地更可靠。

远程 kernel 仓库中的 `AGENTS.md`、CI 和项目约定始终是最高权威；本仓库的 skill 不能替代它们。

## 能力结构

| Skill | 职责 | 典型触发 |
| --- | --- | --- |
| `kernel-dev-lifecycle` | 总路由、硬门禁与最终交付验收 | 新 kernel、修复、优化、端到端任务 |
| `kernel-goal-discovery` | 只消除会改变实现方向的未知项 | 语义、shape、dtype、reference 或硬件不清楚 |
| `kernel-design-docs` | 用项目模板形成最小且可执行的 RFC/语义契约 | 新 kernel、API/数学/分布式语义或架构变化 |
| `implement-kernel-from-plan` | 从已确认契约实现或修改 kernel | 设计与 oracle 已确定 |
| `analyze-kernel` | correctness 与性能证据分析 | 诊断、基线比较、瓶颈判断 |
| `profile-pallas-xprof` | 按需采集并验证 XProf 证据 | 组件耗时、通信重叠、短 kernel 或证据歧义 |
| `kernel-tuning-loop` | 单次可归因优化或有预算的重复调优 | 一个或多个可证伪优化假设 |
| `kernel-foundry` | guardrail、fuzz、research、replay、portfolio、causal、genome | 重复开发和系统学习 |

`optimize-kernel-from-evidence` 已合并进 `kernel-tuning-loop`：单次优化使用 `single`，重复实验使用 `research`。二者共享 correctness、baseline、full-latency 和通信重叠门禁，避免维护两份近似规则。

诊断与 XProf 采集仍然分开：多数问题可由已有证据完成诊断，不应把昂贵 profile 变成每个 kernel 的固定仪式。

## 最小工作流

`kernel-dev-lifecycle` 根据风险选择模式：

| 模式 | 适用范围 | 最小要求 |
| --- | --- | --- |
| `quick` | 语义和 oracle 已知的局部修复 | 仓库约束、修改、目标 correctness、项目检查 |
| `standard` | 新 kernel、移植、语义/API 变化 | operator contract、必要 RFC、实现、correctness matrix、被声明的性能证据 |
| `research` | 多假设调优、portfolio、causal、genome | 稳定基线、显式预算、实验状态机、Pareto 结果 |

硬门禁：

```text
未读取真实仓库约束 -> 不修改仓库
语义或独立 reference 未确认 -> 不优化
correctness 未通过 -> 不得形成性能结论
baseline/测量策略不稳定 -> 不得声明加速
实验不可复现 -> 不得接受为 current best
失败根因未确认 -> 不得升级为共享 guardrail
CI/Definition of Done 未逐项核对 -> 不得报告完成
```

## RFC 模板

标准或研究模式需要 RFC 时，优先使用目标仓库指定的模板；否则使用 [`kernel-design-docs/references/RFC_template.md`](kernel-design-docs/references/RFC_template.md)。

Agent 必须保留模板编号结构，以仓库证据和已确认的 operator contract 填写内容，删除说明性占位符。确实不适用的章节写明原因，不得静默删掉或另造提纲。Quick 模式不强制创建 RFC。

## 交付与 GitHub CI 验收

最终验收先重新读取适用的 `AGENTS.md`，再检查 `.github/workflows`、pre-commit、Ruff、typing、tests、配置校验器、最终 diff 和 worktree。每个 Definition of Done 与 workflow 检查都必须记录为：

- `pass`：给出精确命令或产物；
- `not applicable`：给出原因；
- `blocked`：给出缺少的工具、环境或证据。

可用机械门禁生成 CI inventory、运行常见本地检查、校验 IR-upload tag 和提交信息：

```shell
python kernel-dev-lifecycle/scripts/kernel_delivery_gate.py \
  --repo <kernel-repo> \
  --kernel <kernel> \
  --config <config> \
  --test <test-module> \
  --device-num <n> \
  --snapshot-root </tmp/hlo-snapshot> \
  --commit-message </tmp/commit-message.txt> \
  --pr-text </tmp/pr-or-squash-message.txt> \
  --run \
  --json-out </tmp/delivery-gate.json>
```

门禁会枚举所有 GitHub workflow 文件与 `run:` 项，识别并执行常见的 pre-commit、Ruff lint/format、typing、pytest unit 和 config validator 检查。自定义 workflow 命令仍需按仓库原始配置逐项核对；工具不会盲目执行任意 CI shell。缺少必需工具或未能映射的适用检查应报告为 blocked，而不是假装通过。

用以下命令复制提交信息模板：

```shell
python kernel-dev-lifecycle/scripts/kernel_delivery_gate.py \
  --repo <kernel-repo> --kernel <kernel> --allow-missing-expected \
  --write-commit-template </tmp/commit-message.txt>
```

格式固定为：

```text
feat[TOOL]: add IR simulator support

Task:
- Implement a simulation pass.

Solution:
- Utilize a visitor pattern to traverse the IR.

Test:
- Unit tests for the relu op.
- [ir-upload package=... kernel=... config=... test=... device_num=...]

JIRA: COMPIL-XXXX
```

Agent 可以生成完整草稿，但 `COMPIL-XXXX` 只是虚拟占位符，交付时必须提醒用户替换真实 JIRA。生成草稿不代表获得了 commit、push 或创建 PR 的授权。

## Foundry：让系统从失败中学习

统一入口：

```shell
python kernel-foundry/scripts/kernel_foundry.py <capability> <command> ...
```

当前能力：

- `guardrail`：把已确认失败转成可执行检查，并用失败 replay 与通过 control 验证。
- `fuzz`：通过 adapter 搜索 shape、mask、dtype、边界与数值反例。
- `research`：管理实验预算、状态转换、correctness 失败与 Pareto frontier。
- `replay`：比较无 skill、当前 skill、精简 skill + executable gates 的历史任务表现。
- `portfolio`：只从通过 correctness 的证据生成精确 shape dispatch 表。
- `causal`：分析一次受控 source 变化对应的 HLO/指标移动。
- `genome`：提出可追踪的单基因 kernel 变异，避免不可归因组合修改。

这层不负责猜测算子语义，也不把性能更快等同于正确。共享知识只接收已复现、根因明确、能程序化验证的结论。

## 质量标准与评估

系统是否“更强”由 replay 和 sealed holdout 决定，而不是文档数量：

- 首次 correctness 通过率；
- 仓库约束/CI 违反次数；
- 错误性能结论次数；
- 完成时间、人工纠正轮数与 TPU 消耗；
- 独立工程师复现率；
- 新 kernel 在未读取目标实现时的 holdout correctness、稳定 full latency 与质量差距。

“competitive”和“dominant”必须分开报告。任何 kernel 复现或碾压结论都要求目标源码隔离、独立 adversarial reference、holdout shapes、重复可比 benchmark 和冻结阈值。

## 开发与回归

修改本 skill family 后运行：

```shell
python kernel-foundry/scripts/validate_family.py
```

它会检查：

- 所有 Python 文件语法；
- 每个 `SKILL.md` 的结构；
- RFC 与 commit 模板的关键结构；
- 所有 `*/scripts/tests` 回归套件。

新增失败经验时，优先新增 executable guardrail、fuzz/replay case 或 helper test。只有触发条件、硬门禁、程序路由或异常升级发生变化时，才修改 skill 文本。

# Kernel 分析报告格式

报告默认使用中文。代码符号、文件路径、命令、metric/event 名称保持原文。

```markdown
# <Kernel 名称> 性能分析报告

## 1. 结论摘要

- 当前结论：accepted / rejected / investigating。
- 当前瓶颈：compute/MXU / HBM / VMEM / communication / launch-control / mixed。
- 是否可声明性能收益：是 / 否，以及原因。

## 2. Kernel 合约

| 项目 | 内容 |
| --- | --- |
| 函数 | <module.function> |
| 语义 | <operator semantics> |
| 输入输出 | <shape, dtype, layout> |
| mask/padding | <rules> |
| reference | <trusted reference> |

## 3. Correctness 状态

| Shape | 校验项 | Tolerance | 状态 | Artifact |
| --- | --- | --- | --- | --- |

## 4. Benchmark 摘要

| 实验 | Shape | Baseline median | Target median | Speedup | std/p95 | Artifact |
| --- | --- | --- | --- | --- | --- | --- |

说明 benchmark 口径：warmup、iters、计时方式、是否同设备同配置。

## 5. XProf / Device Timing

| 组件 | 时间/占比 | 证据 | 解读 |
| --- | --- | --- | --- |
| full device time | | | |
| Pallas custom-call | | | |
| collective start/done | | | |
| copy/reshape/transpose | | | |
| fusion/control | | | |
| HBM/DMA | | | |
| VMEM spill/fill | | | |

## 6. FLOPs / Bytes / MFU 模型

| 指标 | 手算值 | profiler 值 | 是否可信 | 说明 |
| --- | --- | --- | --- | --- |

必须区分 useful FLOPs、executed FLOPs、XProf-reported FLOPs。

## 7. 深度诊断问答

- ALU 压力：Vector ALU / Scalar ALU / exp / mask / index 是否过重？
- MXU 利用率：低利用率是否符合 shape 和 tile 预期？
- Memory / spill：HBM、VMEM、scratch、copy 是否主导？
- 通信 overlap：collective 是否暴露，是否和有用计算重叠？
- Host/device：是否被 host dispatch、launch、control 主导？

## 8. 瓶颈分析

用证据解释瓶颈排序，不要只写直觉。

## 9. 优化假设与决策

| 假设 | 目标指标 | 结果 | 决策 | 下一步 |
| --- | --- | --- | --- | --- |

## 10. Artifacts

- Correctness:
- Benchmark:
- XProf:
- Performance report:
- Updated docs:
```

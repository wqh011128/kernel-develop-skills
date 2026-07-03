# Attention Kernel 调优参考

适用于 MHA、GQA、MQA、FlashAttention、causal mask、prefix/ring attention 和 CP attention。

## 1. 正确性前置条件

在任何调优前必须满足：

```text
可信 dense/framework reference 已存在
比较 output 与 LSE/normalizer，而不只比较 shape
显式检查输出 shape、有限值、dtype 与容差
每个公开缩放参数（例如 sm_scale）至少使用一个非默认值
覆盖 causal 边界、分块边界、GQA head group、padding/tail 与 CP rank 边界
```

在线 softmax 的状态为 `m/l/acc` 或等价的 `(lse, output)`。分块或跨 shard 合并必须用稳定的 max/logaddexp 公式；output 接近不能替代 LSE 正确。

## 2. 首先分类瓶颈

| 现象 | 优先检查 | 常见方向 |
| --- | --- | --- |
| MXU 低 | exp/reduce、mask/index、Vector/Scalar ALU、tile 碎片、spill | 增加有效矩阵工作；减少无效 mask 与控制 |
| HBM-bound | K/V 是否重复读取、全局物化、layout copy、output/LSE 中间量 | 提高 K/V reuse，移除物化和额外读写 |
| VMEM-bound 或 spill | accumulator、online-softmax state、head-group 向量化、scratch | 缩小状态；拆分高压向量化；检查 compiler spill |
| communication-bound | collective bytes/count、done/wait、可用 compute 窗口 | 减少暴露同步；仅在可验证时做 overlap |
| launch/control-bound | custom-call 数、JAX cond/fusion、reshape/reorder | 合并有用工作；避免用 JAX 物化换取少量 call |
| mixed | full device time 与组件时间 | 每次只测试一个主因 |

XProf 必看 full device time、Pallas custom-call、collective start/done、copy/layout、fusion/control、Vector/Scalar ALU、MXU、HBM、VMEM spill/fill。任何局部指标改善而 full latency 未改善，均不能接受为优化。

## 3. GQA/MQA 与局部核心

- K/V tile 应在对应的多个 Q heads 间复用；先确认没有按 `Hq` 重复搬运 K/V。
- 不要默认把 `group_size * block_q` flatten 为一个 group matmul 更快。该方式可能扩大寄存器和 VMEM live range，并触发 spill 或编译不稳定。
- 对 flatten 方案与静态 per-head loop 做同条件比较，记录编译是否通过、spill/fill、VMEM、full latency 与 MXU；只有完整证据支持才保留 flatten。
- full-valid、diagonal、invalid causal block 的数学语义不同。可以仅让 diagonal block 承担三角 mask，但必须保持所有 rank 的 collective 顺序一致。

## 4. Ring / Prefix / CP Attention

- Ring 的正确性依赖逐 K/V 块更新在线 softmax 状态，不依赖物化全局 attention matrix。
- 若某条诊断路径最终物化完整全局 K/V，应使用框架 `all_gather`，不要用顺序 ppermute + select/concat 伪装成 ring；没有消费者时直接删除该路径。
- Ring/P2P 的价值必须来自减少内存压力或隐藏通信。开始全量 pipeline 前，先执行 overlap feasibility gate：计算窗口、传输 bytes、状态驻留、barrier 数量和 trace 中的实际重叠。
- 源码中先发后算不等于 overlap。仅当通信完成不在关键路径暴露，且 full device time 下降时，才可宣称重叠收益。
- 所有 rank 必须执行一致的 collective 序列。跳过 future shard 的计算可以尝试，但不能跳过相应通信或改变 collective 次序。
- CP 设备数的支持范围必须由显式 contract 定义并验证。若算法只支持二次幂规模，校验 `cp_size >= 2` 且为二次幂，并覆盖至少两个合法规模；未支持的 K/V tail 必须拒绝或正确 padding/mask，不能静默截断。

## 5. 实验纪律与拒绝条件

每次实验只改变一个主变量，并在 correctness 后执行相同 benchmark policy。报告必须记录：

```text
假设、目标指标、基线、改动、正确性、full latency、组件时间、XProf 证据、接受/拒绝原因
```

立即拒绝：

```text
LSE 或非默认缩放参数失败
只减少 custom-call/collective count 而 full latency 未改善
通过全局 K/V concat、select、reorder 或隐藏 copy 换取表面收益
collective 顺序不一致
未处理 tail 导致静默漏算
```

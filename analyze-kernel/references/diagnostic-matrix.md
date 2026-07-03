# Pallas Kernel 性能诊断矩阵

用于分析 Pallas/JAX kernel 的 XProf、Roofline、FLOPs、Bytes、Memory、Trace 指标。目标不是给出固定优化 recipe，而是把异常指标转成可验证的下一步假设。

## 1. 总体顺序

```text
1. 先确认 correctness。
2. 再确认 benchmark 是否可信：warmup、iters、baseline、shape、dtype、host/device 计时口径一致。
3. 再看 XProf / trace / roofline。
4. 手算 FLOPs / Bytes / communication bytes。
5. 对比 profiler 指标和手算指标。
6. 判断瓶颈类型：compute/MXU、HBM、VMEM、communication、launch/control、mixed。
7. 提出一个优化假设。
8. 做 A/B 实验验证。
```

不要看到慢就直接改 block size。block size 只是在瓶颈分类后的候选动作之一。

## 2. 核心指标含义

| 指标 | 主要回答的问题 |
| --- | --- |
| `custom call time` | Pallas kernel 本体耗时多少 |
| `full JIT/device time` | 整个 JIT/device 执行路径耗时多少 |
| `FLOP Rate` / `FLOP Rate / Peak` | profiler 认为计算吞吐和峰值差距多大 |
| `Roofline efficiency` | 当前点离理论 roofline 多远 |
| `HBM bandwidth` / `Memory BW utilization` | 是否真正打满 HBM，还是只是被判为 HBM-bound |
| `Arithmetic Intensity` | 每搬 1 Byte 数据做多少 FLOP |
| `MXU utilization` | TPU 矩阵单元是否吃满 |
| `Vector/Scalar time` | mask、index、softmax、control 等非矩阵逻辑是否过重 |
| `copy/reshape/transpose/fusion` | layout 或中间结果搬运是否吞掉收益 |
| `custom-call occurrences` | kernel 是否过碎，launch/control 开销是否主导 |
| `collective start/done` | 通信是否暴露，是否和计算 overlap |
| `scratch/VMEM spill/fill` | 片上临时状态是否过大或反复读写 |

## 3. 常见现象到下一步

| 现象 | 可能原因 | 下一步检查 |
| --- | --- | --- |
| `custom call time` 高 | 真实计算重、tile 不合理、数据复用差、scratch 重、mask/control 多、MXU 利用低 | 手算 useful/executed FLOPs；看 MXU/vector/load/store/VMEM/HBM；做单变量 tile sweep |
| `full device time` 远大于 `custom call time` | 外部 copy/fusion/reshape/transpose、framework control、多个小 kernel、collective 暴露 | 看 Trace top ops；dump HLO；检查 layout；减少外部 round-trip |
| Roofline 显示 HBM-bound 但 HBM 利用率低 | profiler FLOPs/Bytes 不完整、kernel 过碎、control/communication 暴露、访存模式差 | 手算 FLOPs/Bytes；检查 custom-call metadata；拆分 copy/fusion/collective/control |
| `Roofline efficiency` 很低 | FLOPs 统计缺失、overhead 主导、AI 低、padding/mask/tile waste、program 过碎 | 不单独相信 roofline；用手算 FLOPs 和 device time 交叉验证 |
| 手算 FLOPs 大但 XProf FLOPs 小 | Pallas custom-call 内部 FLOPs metadata 缺失或统计口径不同 | 报告中区分 XProf FLOPs 与 manual FLOPs；优先使用手算 numerator |
| useful FLOPs 远小于 executed FLOPs | padding、causal mask、tail、diagonal tile 浪费 | 计算 useful/executed ratio；区分 full-valid tile 和 boundary tile |
| AI 低 | HBM/VMEM 读写多、中间结果 materialize、layout copy、shape 小 | 减少 materialization，融合 producer/consumer，提高 tile 内复用 |
| AI 高但 FLOP Rate 低 | MXU 没吃满、pipeline bubble、vector/scalar 插入、VMEM 限制、lowering 不理想 | 看 MXU utilization、HLO/Mosaic IR、load/compute overlap、tile shape |
| HBM 带宽低但耗时长 | overhead-bound、communication latency、小 kernel fragmentation、vector/scalar bottleneck | 看 custom-call occurrences、collective done、fusion/copy/reshape |
| VMEM 接近上限或 OOM | block 太大、scratch/state 太多、FP32 状态过重、double buffering 过重 | 降低 block、压缩 state、分阶段计算、评估 recompute vs store |
| `collective-permute-start/done` 暴露 | 通信未隐藏、ring step 多、payload 过碎、compute chunk 太短 | 做 communication-compute overlap、pack payload、增大 compute chunk、减少 step |
| ring 比 all-gather 慢 | ring step 多、local-core fragmentation、state merge/control、communication 未隐藏 | 分离 collective/local-core/merge；减少 custom-call；保留 memory-pressure benchmark |
| 单次 sweep 变快但 repeat 不复现 | measurement noise、warmup 不足、系统负载、配置漂移 | 做 focused repeat；看 median/p90/std；不接受不可复现收益 |

## 4. 报告必须回答的问题

```text
correctness 是否已经通过，reference 是否可信？
benchmark 是否同 shape/dtype/warmup/iters/baseline？
manual FLOPs/Bytes 和 XProf 指标是否一致？不一致时采用哪个口径？
当前瓶颈属于哪一类，证据是什么？
优化假设预期移动哪个指标？
如果指标未移动或 full latency 未改善，拒绝条件是什么？
```

## 5. 决策模板

```text
correctness 不通过：reject。
correctness 通过但 benchmark 回退：作为 diagnostic 可以保留，作为 optimization 默认 reject。
单次变快但 focused repeat 不复现：reject / neutral。
局部 custom-call 变快但 full latency 被 copy/collective/control 抵消：reject。
latency 不赢但显著节省内存：标记为 memory-efficient baseline，不声明 latency win。
profiler 指标不可信：先补手算 FLOPs/Bytes，再下结论。
```

# Attention Kernel 调优参考

用于 MHA、MQA、GQA、FlashAttention 风格 kernel、causal mask、prefix/ring attention、分布式 context-parallel attention。

## 1. Correctness Gates

- 优化 softmax 或通信前，先验证 logits mask。
- 使用 online softmax、blockwise merge、ring attention、partitioned attention 时，必须单独验证 `lse` 或 denominator state。
- 对 streaming/ring/blockwise attention，任意已处理 K/V prefix 后的 `(output, m, l)` 或 `(output, lse)` state 都必须等价于只看该 prefix 的可信 reference；不要只在最终输出处验证。
- output equality 和 `lse` equality 是两类检查；partitioned merge order 可能让 output 有小差异，但 normalization 必须正确。
- 替换通信或 local core 前，必须对齐可信 dense/framework reference。
- 覆盖短序列、非整除 block、causal boundary、grouped heads、padding。

## 2. Bottleneck Patterns

- 在 profile 中拆分 local attention compute、softmax merge、reshape/copy、collective time。
- 不要假设通信 bytes 变少就一定更快；额外 loop/control/merge overhead 可能主导。
- 分布式 attention 必须 profile full device time 和 collectives，不能只看 Pallas custom-call。
- 小 kernel 优先看 device/profile timing，不只看 host wall-clock。
- ring/prefix attention 要分开统计 visible shard useful work 和 invalid/future shard local-core work。
- MXU utilization 低时，先检查 online-softmax exp、mask/index arithmetic、Vector ALU、Scalar ALU、spill、launch/control 是否主导，再改 matmul tile。
- 通信 overlap 要比较 collective start/done 和 local-core 窗口，明确 exposed 还是 hidden。
- causal CP attention 中 collective count 降低不够；必须验证 `collective-permute-done`、fusion/control、slice/reshape 没有增长。
- 先用 XProf/Roofline 证据分类：compute/MXU、HBM、VMEM、communication、launch/control、mixed。
- 发明新结构前，对比高质量 attention 实现或项目本地 kernel：online softmax state、LSE、mask、layout、communication boundary 如何处理。
- block-size tuning 必须同时报告 full latency 和 XProf component movement。更大 tile 可能降低 launch/control，但增加 VMEM 压力或降低 occupancy。

## 3. Hypotheses To Test

- 一次只改变 block size、layout、masking strategy、communication pattern、accumulator precision 中的一个主变量。
- 调 query 和 key/value block size 前先固定 correctness；同时记录 custom-call time 和 full time。
- ring/prefix attention 要分别测 collective latency、merge overhead、memory materialization。
- 只有证据显示 merge/control overhead 显著时，才尝试 fuse merge/update。
- invalid/future-shard skipping 只有在所有 rank collective order 完全一致时才可测试。
- rank-specialized branch 默认可疑，除非 XProf 证明 custom-call 下降且 collective/control 不增长。
- 在 loop 中使用 JAX `lax.cond` 或 branch function 时，把 shard id、global offset、mask bounds 等 loop-derived dynamic scalar 显式作为 operand 传入，不要依赖 Python closure capture。
- full latency 回退时拒绝 rank-specialized 或 visible-prefix skipping，即使 invalid shard work 或 custom-call count 下降。
- HBM-bound 时优先避免 K/V materialization、减少 output/LSE intermediates、提升 reuse，再做 tile micro-tuning。
- VMEM-bound 或 spill-heavy 时优先减少 accumulator/state footprint 或拆分 state update。
- state compression 必须验证 target block size 和 compiler lowering。更小的 output/state tensor 仍可能因为 scratch shape、tiling、broadcasting、lowered temporaries 增加或触发 scoped VMEM OOM。
- launch/control-bound 时减少 Pallas call count 和 JAX-side dynamic control；不要在未证明 compile cost 可接受前引入大型 `lax.switch` 或 per-rank branch duplication。
- ring/prefix attention 中，如果减少 kernel launch 需要 JAX-side K/V materialization、select、gather、concat，不能直接算作优化；full latency 必须包含这些成本。
- communication-bound 时优化 exposed collective done/sync time，而不只是 start count 或 payload size。
- causal attention 应先分类 full-valid、diagonal、invalid block，再决定是否加 mask 或 skip；full-valid fast path 不应承担 diagonal mask 的 Vector/Scalar ALU 成本。
- GQA/MQA 应审计 K/V 是否按 `Hkv` 维度 staging 并被对应的多个 query heads 复用；如果 K/V 随 `Hq` 重复加载，优先解决 HBM/VMEM reuse。
- chunk/block size sweep 必须同时报告 compute/communication overlap、VMEM/scratch、tile waste。不要假设 chunk 越大越好。

## 4. Rejection Conditions

- `lse` correctness 回退时拒绝，即使 output 看起来接近。
- collective + merge overhead 抹掉 custom-call 收益时拒绝通信优化。
- hidden transpose、reshape、HBM copy 大于节省计算时拒绝 layout change。

## 5. 指标异常到 Attention 优化方向

| 现象 | 优先怀疑 | 优先验证 |
| --- | --- | --- |
| MXU 低但 attention 有 matmul | softmax/mask/vector/control 主导，或 tile 太碎 | Vector/Scalar ALU、exp/reduce fusion、custom-call occurrences |
| HBM-bound 但 HBM 利用率低 | profiler FLOPs 不完整、materialization、control/communication 暴露 | 手算 FLOPs/Bytes、trace top ops、collective done |
| ring 慢于 all-gather | ring step、local-core fragmentation、merge/state、communication 未 overlap | local-core count/time、collective start/done、merge/fusion/control |
| 减少 shard/core 数但 full latency 不降 | JAX-side concat/reorder/select/materialization 抵消收益 | full time、copy/reshape/fusion、HBM/VMEM traffic |
| output 正确但 `lse` 不稳 | online softmax merge 或 mask/padding 边界错误 | 单独校验 LSE，覆盖 tail/padding/future shard |

## 6. Ring / Flash / Striped Attention 经验

- Ring attention 本质上是把 blockwise/FlashAttention 的 K/V 分块遍历扩展到多设备。正确性核心是不物化全局 attention matrix，而是维护 `m/l/acc` 或 `LSE + output` 的稳定 online softmax state。
- 不要直接 `exp(lse)` 后相加；跨 block 或跨 shard merge 必须使用 `logaddexp` 或等价的差分指数公式，避免 overflow/underflow。
- Ring 的性能前提是 communication-compute overlap。只有当本地 compute chunk 足够长、`collective-permute-done` 能被 local compute 隐藏时，P2P/ring 才可能抵消 all-gather 的简单性。
- Ring step 数、block granularity、K/V payload size 是一组 trade-off：更小 block 便于 overlap 和负载均衡，但增加通信/launch/control；更大 block 减少 step，但可能增加 VMEM、tile waste 或降低 occupancy。
- 对 causal attention，连续按序列切 CP shard 会带来三角 mask 负载不均衡。必须查看 per-rank/device work、custom-call time、collective wait，而不是只看全局 median。
- Striped/Zigzag 分配可以缓解 causal load imbalance，但它会改变 token-to-rank layout。作为优化前必须先证明：global causal mask 等价、输出能恢复到原 contiguous order、padding/position/RoPE/segment 边界正确。
- 如果 striped/zigzag 让每个 Q/K tile 变成 full square 或接近 full square，可尝试非 causal local core 或减少 diagonal/boundary tile，但必须把 index restore、layout copy、通信复杂度计入 full latency。
- GQA/MQA 场景下，基于 head 维切分的 Ulysses/all-to-all 类路线可能受 `Hq/Hkv` 数量限制；ring 对 head 数不敏感，但仍要用 benchmark/XProf 和 all-gather/Ulysses/hybrid baseline 比较。
- 对当前只返回本 rank contiguous output 的 prefill kernel，striped/zigzag 通常应作为独立实验，不要和 state-update fusion、block-size tuning、communication packing 混在同一轮。

## 7. Causal Block Specialization 与 GQA 复用

- causal block 优化前，先用全局 token offset 判断 block 类型：`kv_end <= q_start` 为 full-valid，`kv_start >= q_end` 为 invalid，其余为 diagonal。
- full-valid block 应尽量走无 causal mask 的路径；diagonal block 才需要边界 mask；invalid block 应避免 local-core 计算，但不能破坏所有 rank 的 collective 顺序。
- 接受 causal block specialization 的条件是 full latency 改善，并且 XProf 中 mask/where、Vector ALU、Scalar ALU、fusion/control 或 invalid custom-call 有对应下降；只降低理论 FLOPs 不够。
- 对 GQA/MQA，优先确认同一 `Hkv` 对应的多个 `Hq` query heads 共享 K/V tile staging。若 profile 显示 K/V load/store 随 `Hq` 放大，应优先尝试 head-group reuse 或 local-core layout 调整。
- 对 streaming/ring attention，chunk size 选择必须让 communication window、local compute window、VMEM/scratch footprint 同时可解释。若 chunk 变大导致 spill、tile waste 或 HBM materialization，上升的 full latency 优先于局部 custom-call 改善。
- backward attention 不得直接复用 forward 的 memory model；必须分别跟踪 `dQ/dK/dV`、softmax delta、partial gradients 和跨 block merge invariant。

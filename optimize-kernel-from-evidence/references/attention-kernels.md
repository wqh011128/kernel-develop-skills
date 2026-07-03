# Attention Kernel 调优参考

用于 MHA、MQA、GQA、FlashAttention 风格 kernel、causal mask、prefix/ring attention、分布式 context-parallel attention。

## 1. Correctness Gates

- 优化 softmax 或通信前，先验证 logits mask。
- 使用 online softmax、blockwise merge、ring attention、partitioned attention 时，必须单独验证 `lse` 或 denominator state。
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

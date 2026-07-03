# Reduction / Scan Kernel 调优参考

用于 reductions、prefix scans、normalization、statistics、online merge state。

## 1. Correctness Gates

- 明确定义 associativity 假设和 accumulation dtype。
- 覆盖非 2 的幂、singleton、padding tail、空输入语义。
- online algorithm 必须验证中间 state，不只看最终 output。
- 分布式 reduction 要分别验证 local state 和 collective merge。

## 2. Bottleneck Patterns

- 区分 arithmetic work、同步、memory traffic、control overhead。
- 检查 reduction tree shape、vectorization、memory layout 谁主导。
- 分布式 reduction 要拆分 local reduction time 和 collective time。
- high vector/scalar time 常来自 mask、index、state update、dtype conversion。

## 3. Hypotheses To Test

- 在 accumulation semantics 不变的前提下调 reduction granularity 和 tree shape。
- 测试 pre-normalization 或 state compression 是否减少 memory traffic。
- 比较 local-only、blockwise、collective variants，correctness artifact 必须一致。
- 如果 state compression 触发 VMEM/spill，回退或做 VMEM-aware redesign。

## 4. Rejection Conditions

- 未记录 tolerance 的非结合 reorder 直接拒绝。
- 数值稳定性或边界行为回退时拒绝。
- local time 下降但 collective/control 抵消 full latency 时拒绝。

# Matmul 类 Kernel 调优参考

用于 GEMM、batched GEMM、projection，以及主要由 dense matrix multiplication 主导的 kernel。

## 1. Correctness Gates

- 性能测试前固定 dtype 和 accumulation policy。
- 覆盖 transpose/layout 组合和非整除 tile shape。
- 对齐 framework matmul 或简单 JAX reference，并明确 tolerance。

## 2. Bottleneck Patterns

- 从 tile size、grid cell、loop count 手算 FLOPs。
- 区分 compute/MXU-bound、memory-bound、launch/control-bound。
- 调 MXU 前先检查 spill、layout conversion、hidden copy。
- 如果手算 FLOPs 大但 XProf FLOPs 小，优先怀疑 Pallas custom-call metadata 不完整。

## 3. Hypotheses To Test

- 一次只调一个变量：tile size、accumulator layout、packing、padding、dtype。
- padding 到硬件友好维度必须把 padding/copy 成本计入 full latency。
- 和 vendor/framework baseline 在同 shape/dtype/layout 下比较。
- 如果 tile 变大更快，检查是否只是减少 launch/control；如果变慢，检查 VMEM/spill/occupancy。

## 4. Rejection Conditions

- inner custom-call 变快但 reshape/copy/HBM overhead 更大时拒绝。
- 只优化单个 shape 却破坏目标 shape coverage 时拒绝。
- correctness 或 accumulation tolerance 回退时拒绝。

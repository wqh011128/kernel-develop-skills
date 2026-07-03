# Elementwise / Fusion Kernel 调优参考

用于 pointwise transform、activation、bias/dropout-style fusion、轻量 fused kernel。

## 1. Correctness Gates

- 验证 broadcasting、dtype promotion、NaN/Inf、boundary mask。
- fused output 必须和 unfused reference 在代表性 shape 上对齐。
- 如果改变 op order，记录数值差异来源和 tolerance。

## 2. Bottleneck Patterns

- elementwise kernel 通常更容易被 memory bandwidth、layout conversion、launch overhead 主导，而不是 FLOPs。
- 增加 fusion 前先检查 hidden materialization 和 redundant reads/writes。
- 很小的 kernel 中 host dispatch 和 compile effect 可能掩盖 device 改善。
- high fusion/control time 可能说明 fusion 没有真正减少 HBM round-trip。

## 3. Hypotheses To Test

- 只有在减少 materialization 或 launch count 且不增加 spill 时才接受 fusion。
- 优先调 vectorization、memory coalescing、layout 对齐，再追求 arithmetic throughput。
- 和 unfused JAX/compiler output 对比；compiler 可能已经自动 fuse 简单 pointwise chain。
- 对比 full latency，不只看 fused op 本身。

## 4. Rejection Conditions

- fusion 增加 spill、register pressure、HBM traffic 时拒绝。
- warmup/iters 稳定后收益消失时拒绝。
- correctness 在 NaN/Inf/broadcast 边界回退时拒绝。

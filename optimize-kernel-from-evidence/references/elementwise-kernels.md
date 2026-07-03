# Elementwise And Fusion Kernel Tuning Notes

Use for pointwise transforms, activations, bias/dropout-style fusions, and lightweight fused kernels.

## Correctness Gates

- Validate broadcasting, dtype promotion, NaN/Inf behavior, and boundary masks.
- Compare fused output with the unfused reference over representative shapes.

## Bottleneck Patterns

- Expect memory bandwidth, layout conversion, or launch overhead to dominate more often than FLOPs.
- Inspect hidden materializations and redundant reads/writes before adding more fusion.
- For very small kernels, host dispatch and compile effects can hide device improvements.

## Hypotheses To Test

- Fuse operations only when it reduces materialization or launch count without increasing spills.
- Tune vectorization and memory coalescing before focusing on arithmetic throughput.
- Benchmark against unfused JAX/compiler output; compilers may already fuse simple pointwise chains.

## Rejection Conditions

- Reject fusion that increases spills, register pressure, or HBM traffic.
- Reject changes whose measured benefit disappears when benchmark warmup and iteration policy are stable.

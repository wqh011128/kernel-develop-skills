# Matmul-Like Kernel Tuning Notes

Use for GEMM, batched GEMM, projection, and kernels dominated by dense matrix multiplication.

## Correctness Gates

- Fix dtype and accumulation policy before measuring performance.
- Test representative transpose/layout combinations and non-multiple tile sizes.
- Compare against framework matmul or a simple JAX reference with explicit tolerances.

## Bottleneck Patterns

- Build a manual FLOPs model from tile sizes, grid cells, and loop counts.
- Cross-check whether the kernel is compute-bound, memory-bound, or launch/control-bound.
- Inspect spills, layout conversions, and hidden copies before tuning only MXU utilization.

## Hypotheses To Test

- Tune tile sizes and accumulation layout one variable at a time.
- Check whether padding to hardware-friendly dimensions improves full time after including copy/padding cost.
- Compare against vendor/framework baselines for the same shape and dtype.

## Rejection Conditions

- Reject changes that improve inner custom-call time but add larger reshape/copy overhead.
- Reject tile changes that only improve one shape while breaking documented target shape coverage.

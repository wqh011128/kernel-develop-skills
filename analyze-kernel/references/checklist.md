# Kernel Analysis Checklist

Follow these phases sequentially. Check off items as you complete them.

## Phase 1 — Identify the kernel

- [ ] Function name, module path, import statement
- [ ] Input signature: list all args with shapes and dtypes
- [ ] `pallas_call` grid shape and `dimension_semantics`
- [ ] BlockSpec shapes and index_map for each input/output
- [ ] Static parameters (block_size, num_iterations, etc.)

## Phase 2 — Build matmul inventory

- [ ] Find all `jnp.dot()`, `jnp.matmul()`, `@`, `jnp.einsum()` calls in the kernel body
- [ ] For each matmul: determine input shapes from BlockSpec and variable definitions
- [ ] Count iterations: `fori_loop(0, N, ...)` → N iterations; python `for` → unrolled count
- [ ] Count grid cells from `grid=` parameter
- [ ] Compute per-matmul FLOPs: `2 × M × K × N`
- [ ] Compute per-matmul arithmetic intensity: `FLOPs / bytes_accessed` (bytes = elements × dtype_bytes for all operands)
- [ ] Total FLOPs = sum of (per_matmul_FLOPs × iterations × grid_cells)
- [ ] **Cross-check**: If a pure-JAX reference exists, JIT-compile it and call `compiled.cost_analysis()` → `model_flops`. Compare against manual count. Note discrepancy if any.
- [ ] Classify each matmul: compute-bound (AI > VMEM crossover of 40.6) or memory-bound

## Phase 3 — Identify optimizations

Common Pallas kernel optimization techniques to check for:

- [ ] Head fusion / wave pipeline (processing multiple heads per grid cell)
- [ ] Multi-block KV (processing multiple KV blocks per loop iteration)
- [ ] Gathered mode (pre-gathering selected data vs loading full arrays)
- [ ] Online softmax (running max/sum accumulation instead of full softmax)
- [ ] Fused operations (combining mask computation with attention in one kernel)
- [ ] Weight absorption (folding projection weights into Q or output path)
- [ ] Bool-free arithmetic (int32 accumulation + arithmetic conversion instead of bool tensors)
- [ ] Additive masking (bf16 addition instead of `jnp.where` conditional select)
- [ ] Custom VJP (separate forward/backward for training support)
- [ ] Tiling strategy (how the computation is blocked for VMEM capacity)

## Phase 4 — Correctness test

- [ ] Identify a reference implementation (pure JAX, no Pallas) that computes the same output
- [ ] Generate test data matching the kernel's input signature (use random normal, scale weights by 0.02 for numerical stability)
- [ ] Test at small shapes first (S=1024, H=8) for fast iteration
- [ ] Metrics: cos_sim > 0.9999 (PASS), > 0.99 (MARGINAL), < 0.99 (FAIL); also report max_diff
- [ ] If no reference exists, note this and skip correctness phase

## Phase 5 — Benchmark

- [ ] Warmup: 5+ iterations with `block_until_ready()`
- [ ] Measure: 20+ iterations, record all times
- [ ] Report: min, median, mean, std, p5, p95 in milliseconds
- [ ] Run baseline(s) at the same shapes
- [ ] Compute speedup ratios (use median times)
- [ ] Note if wall clock is unreliable (kernel < 10ms — host dispatch ~100us dominates)

## Phase 6 — Trace and xplane

- [ ] Always set env: `LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"`
- [ ] Capture trace: warmup 3 → `jax.profiler.start_trace` → 5 iterations → `stop_trace`
- [ ] Parse xplane.pb for per-op `device_duration_ps`
- [ ] Identify the Pallas kernel's custom-call op name (e.g., `_lambda_.N`)
- [ ] Extract its device time (average across trace iterations)

## Phase 7 — Hardware analysis

- [ ] MXU% = theoretical_FLOPs / (device_time_seconds × peak_TFLOPS × 1e12) × 100
- [ ] Per-matmul AI classification (compute-bound vs memory-bound using VMEM crossover of 40.6 FLOP/byte)
- [ ] Estimate non-matmul overhead: total_device_time − (theoretical_FLOPs / peak_TFLOPS)
- [ ] Identify the dominant bottleneck: largest matmul, memory transfers, non-matmul ops

## Phase 8 — Report assembly

- [ ] Follow `report-format.md`
- [ ] Include concrete next-step optimization suggestions based on bottleneck analysis
- [ ] Save report alongside kernel source or at user-specified location

# Kernel Analysis Report Format

Use this template when assembling the final report in Phase 5.

```markdown
# <Kernel Name> Analysis Report
Generated: <date>
Device: <TPU model>

## 1. Computation Flow
- **Kernel**: <function name> from <module>
- **Description**: <what the kernel computes>
- **Grid**: <grid shape> — <meaning of each axis>
- **Key parameters**: <block_size, fuse_heads, etc.>
- **Optimizations**: <list of techniques identified>
- **Matmul inventory**:

  | # | Operation | Shape (M,K)@(K,N) | Per-call FLOPs | Calls/cell | Total FLOPs | AI (FLOP/byte) | Bound |
  |---|-----------|-------------------|----------------|------------|-------------|----------------|-------|
  | 1 | ... | ... | ... | ... | ... | ... | ... |

- **Total theoretical FLOPs**: X.XX TFLOP
- **FLOP cross-check**: XLA cost_analysis on JAX reference reports X.XX TFLOP (match / X% discrepancy)

## 2. Correctness

| Test Shape | cos_sim | max_diff | Status |
|------------|---------|----------|--------|
| ... | ... | ... | PASS/MARGINAL/FAIL |

- **Reference**: <function used>
- **Threshold**: cos > 0.9999 = PASS, > 0.99 = MARGINAL, else FAIL

## 3. Efficiency

### Timing

| Config | min (ms) | median (ms) | mean (ms) | std (ms) | p5 (ms) | p95 (ms) |
|--------|----------|-------------|-----------|----------|---------|----------|
| ... | ... | ... | ... | ... | ... | ... |

### Speedup vs Baselines

| Config | This kernel (ms) | Baseline (ms) | Speedup |
|--------|-------------------|---------------|---------|
| ... | ... | ... | ... |

## 4. Hardware Utilization

| Metric | Value |
|--------|-------|
| Theoretical FLOPs | X.XX TFLOP |
| Device time (xplane) | X.XX ms |
| Achieved TFLOPS | X.X |
| Peak TFLOPS | 946.7 (TPU v6e bf16) |
| **MXU utilization** | **X.X%** |

## 5. Bottleneck Analysis
- **Dominant cost**: <which matmul or operation>
- **Bottleneck type**: <compute-bound / memory-bound / non-matmul overhead>
- **Non-matmul overhead**: ~X.X ms (X% of total)
- **Per-matmul breakdown**:

  | Matmul | Shape | AI (FLOP/byte) | Bound | % of FLOPs |
  |--------|-------|-----------------|-------|------------|
  | ... | ... | ... | ... | ... |

- **Suggested next optimizations**:
  1. ...
  2. ...

## 6. Trace
- **Trace path**: <path on TPU>
- **Per-op breakdown** (top 10 by device time):

  | Op | Device time (ms) | % of total |
  |----|------------------|------------|
  | ... | ... | ... |
```

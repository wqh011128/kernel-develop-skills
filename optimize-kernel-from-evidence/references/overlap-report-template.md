# Communication-Compute Overlap Analysis Report

Write this report in Chinese by default. Keep operation names, metric names, profiler event names, and commands unchanged.

## 1. Goal

State the overlap hypothesis being tested.

## 2. Pipeline Step

| Item | Description |
| --- | --- |
| compute tile | |
| communication tile | |
| state | |
| output | |
| backend | |
| device | |
| dtype | |

## 3. Memory Model

| Tensor | Shape | Bytes | Lifetime |
| --- | ---: | ---: | --- |
| current input | | | |
| next receive buffer | | | |
| accumulator/state | | | |
| scratch | | | |
| output/writeback | | | |

## 4. Arithmetic Model

```text
FLOPs_step =
achieved_compute_throughput =
T_compute_est =
T_compute_measured =
```

## 5. Communication Model

```text
bytes_comm_step =
achieved_bandwidth =
startup/control/wait overhead =
T_comm_est =
T_comm_measured =
```

## 6. Overlap Condition

```text
T_comm_est <= T_compute_est ? yes/no
```

Extended risks:

```text
state round-trip:
custom-call fragmentation:
barrier/control:
copy/layout/materialization:
scratch traffic:
```

## 7. Probe Results

| Run | median | mean | p05 | p95 | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| compute_only | | | | | |
| comm_only | | | | | |
| serial_compute_then_comm | | | | | |
| candidate_overlap_step | | | | | |

## 8. Trace Summary

| Run | comm start | comm done/wait | compute | barrier/control | custom calls | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| compute_only | | | | | | |
| comm_only | | | | | | |
| serial_compute_then_comm | | | | | | |
| candidate_overlap_step | | | | | | |

## 9. Interpretation

Answer:

```text
Is communication exposed?
Is communication hidden?
Does candidate beat serial/natural schedule?
Does full kernel improve?
What bottleneck moved?
What bottleneck remains?
```

## 10. Decision

Choose exactly one:

```text
proceed_to_pipeline_design
run_structural_breakdown
change_chunk_or_tile_size
optimize_state_lifetime
reject_expression_order_tuning
keep_current_baseline
```

Do not write `continue exploring`.

## 11. Next Step

Name exactly one next action.


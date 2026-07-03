# Communication-Compute Overlap Feasibility

Use this reference before implementing any optimization whose expected speedup depends on hiding communication, DMA, async copy, prefetch, remote transfer, collective communication, or local data movement under compute.

Do not treat this as Ring Attention specific. It applies to all-gather, reduce-scatter, all-to-all, ppermute, send/recv, remote DMA, async copy, HBM/VMEM/SRAM movement, distributed attention, MoE dispatch/combine, matmul pipelines, and reduction pipelines.

## 1. Define One Pipeline Step

Write the unit step before coding:

```text
compute tile:
  inputs consumed by useful compute
  accumulator/state read and written
  output or partial output produced

communication tile:
  payload sent/received/copied/prefetched
  producer and consumer
  when the payload is first needed
```

If the compute uses the communicated data in the same step, the step cannot hide that communication. Redesign the schedule or reject overlap.

## 2. Memory Residency Model

List all tensors that must be resident at the same time:

```text
current compute input tile
current communication send tile
next receive/prefetch tile
output/writeback tile
accumulator/state
scratch
mask/metadata
double buffers
layout/copy/materialization buffers
```

For ring-style attention, a minimal forward residency model is:

```text
Q_i
current K_j, V_j
next receive K_{j-1}, V_{j-1}
output accumulator or online state
normalizer l
safe-softmax max m
scratch/mask/metadata
```

For bf16/fp16 K/V tile shape `[c, d]`:

```text
K tile bytes = 2 * c * d
V tile bytes = 2 * c * d
K/V bytes   = 4 * c * d
double-buffer K/V ~= 8 * c * d bytes
```

Add Q/state/output/scratch separately. If the double-buffer working set exceeds VMEM/SRAM budget or causes spill/fill, reject the overlap design or change chunk size before writing the full kernel.

## 3. Arithmetic Compute Model

Estimate useful compute per step:

```text
FLOPs_step = useful math FLOPs for one overlap window
T_compute_est = FLOPs_step / achieved_compute_throughput
T_compute_measured = median(compute_only)
```

Use achieved FLOPs when available. Peak FLOPs is only a loose upper bound. Distinguish theoretical FLOPs, achieved FLOPs, useful FLOPs, and tile-executed FLOPs.

For attention-like ring step with Q/K/V tile size `[c, d]`:

```text
QK^T FLOPs ~= 2 * d * c^2
P @ V FLOPs ~= 2 * d * c^2
softmax/state FLOPs = model separately
attention step FLOPs ~= 4 * d * c^2 + non-matmul work
```

## 4. Communication Model

Estimate communication per step:

```text
bytes_comm_step = payload_elements * bytes_per_element
T_comm_est = bytes_comm_step / achieved_bandwidth + startup/control/wait overhead
T_comm_measured = median(comm_only)
```

Include:

```text
collective startup
DMA launch/control
barrier/wait
send/recv scope
packetization or per-block overhead
number of communication steps
copy/layout/materialization around the communication
```

For bf16/fp16 K/V attention tile:

```text
bytes_comm ~= 4 * c * d
T_comm ~= 4cd / B
```

where `B` is achieved bandwidth, not peak bandwidth unless no measurement exists.

## 5. Basic Overlap Inequality

Overlap is plausible only if:

```text
T_comm_est <= T_compute_est
```

For simplified ring attention:

```text
4cd / B <= 4dc^2 / F
=> c >= F / B
```

where:

```text
c = chunk/block length
d = head/model dimension
B = achieved communication bandwidth
F = achieved compute throughput
```

If `T_comm_est > T_compute_est`, do not write a pipeline first. Change chunk size, reduce payload, increase useful compute per step, or choose a different communication strategy.

## 6. Extended Real-Kernel Condition

`T_comm <= T_compute` is necessary, not sufficient. A real kernel wins only if:

```text
T_comm_exposed
+ T_barrier
+ T_control
+ T_state_roundtrip
+ T_copy_layout
+ T_custom_call_fragmentation
+ T_scratch_traffic
<= hidden_by_compute_or_removed
```

A design fails if:

```text
communication is hidden in a probe but full latency does not improve
state must be written and reread every step
custom-call/kernel fragmentation dominates
copy/layout/materialization offsets communication savings
barriers move to lower-level trace events instead of disappearing
no-communication multi-step compute is already slower than the baseline
```

## 7. Minimal Probe

Before changing the full kernel, run or design these runs:

```text
compute_only
  measure useful compute window C

comm_only
  measure exposed communication/transfer cost M

serial_compute_then_comm
  measure backend natural scheduling S for nominal serial source

candidate_overlap_step
  measure explicit overlap schedule O
```

Optional probes:

```text
comm_then_compute
multi_step_pipeline
no_comm_multi_step_compute
comm_only_multi_step
copy_only or dma_only
state_read_write_only
```

Define:

```text
C = median(compute_only)
M = median(comm_only)
S = median(serial_compute_then_comm)
O = median(candidate_overlap_step)
```

Interpretation:

```text
O ~= C + M       -> communication is exposed; overlap failed
O ~= max(C, M)  -> communication is likely hidden
O ~= C and C > M -> communication is mostly hidden under compute
O ~= S          -> candidate schedule does not improve over backend natural schedule
```

Source expression order is not execution order. If candidate and serial are the same, reject expression-order tuning. If the probe succeeds but the full kernel is slower, run structural breakdown before more source reordering.

Example:

```text
compute_only = 2.260660
comm_only = 1.005300
serial_compute_then_comm = 2.231855
candidate_overlap_step = 2.230580

comm_only communication-done = 0.772663 ms
serial communication-done = 0.000152 ms
candidate communication-done = 0.000155 ms
```

Conclusion:

```text
communication completion can be hidden under compute in this probe
candidate schedule does not improve over serial/natural schedule
decision = reject_expression_order_tuning
if full kernel remains slower, run structural breakdown
```

## 8. Profiler Validation

Always inspect:

```text
communication start
communication done/wait/completion
compute region
kernel/custom-call count
copy/layout/reshape/fusion
load/store
spill/fill
barrier/control
parent execute scope
host dispatch
state read/write
scratch traffic
full device time
```

TPU/XProf examples:

```text
collective start/done
all-gather start/done
collective-permute start/done
barrier-cores
Pallas custom-call region names
custom-call region trace
copy/fusion/layout
VMEM spill/fill
MXU/vector/load/store
```

GPU/Nsight-like examples:

```text
memcpy async
NCCL collective start/end
kernel launch boundaries
stream synchronization
copy engine utilization
SM occupancy
memory throughput
```

Do not use broad keyword matching to prove barrier duration. Prefer exact event names or prefix matching. Do not count parent scopes such as `CommonPjRtLoadedExecutable::Execute` as named barrier duration. A trace event getting smaller is not success unless full latency improves.

## 9. Decision Gates

End with exactly one decision:

```text
proceed_to_pipeline_design
run_structural_breakdown
change_chunk_or_tile_size
optimize_state_lifetime
reject_expression_order_tuning
keep_current_baseline
```

Use `proceed_to_pipeline_design` only if complexity model shows plausible overlap, the minimal probe shows communication can be hidden or reduced, the full-kernel bottleneck is communication/control related, state dependency does not force full serialization, and the expected moved metric is explicit.

Use `run_structural_breakdown` when the minimal probe shows overlap potential but the full kernel remains slower, or the remaining bottleneck may be custom-call fragmentation, state round-trip, copy/layout, barrier/control, or scratch traffic.

Use `change_chunk_or_tile_size` when `T_comm_est > T_compute_est`, the compute window is too small, trace shows exposed communication due to insufficient compute, and memory/VMEM permits a larger chunk.

Use `optimize_state_lifetime` when no-communication multi-step compute is slower than baseline, accumulator/state crosses kernel boundaries repeatedly, or state read/write traffic is large.

Use `reject_expression_order_tuning` when candidate overlap is equivalent to the serial schedule, communication-done is already hidden in both, or full latency does not improve.

Use `keep_current_baseline` when overlap is not expressible, pipeline complexity exceeds plausible benefit, the target path loses on full latency, or trace does not identify a movable bottleneck.


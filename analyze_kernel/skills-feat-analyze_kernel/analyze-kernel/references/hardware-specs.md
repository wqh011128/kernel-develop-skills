# TPU v6e-1 Hardware Specifications

Verified via XPlane profiler. Device type: "TPU v6 Lite".

## Compute

| Spec | Value |
|------|-------|
| Peak bf16 TFLOPS | 946.7 |
| Systolic array | 256 × 256 |
| No MegaCore | — |

## Memory

| Tier | Capacity | Bandwidth |
|------|----------|-----------|
| HBM | 32 GB | 1638 GB/s |
| VMEM (read) | 32 MB | 23296 GB/s |
| VMEM (write) | 32 MB | 16128 GB/s |

Merged VMEM (no separate accumulator buffer).

## Roofline Crossovers

Arithmetic intensity (FLOP/byte) where compute = memory time:

| Memory tier | Crossover AI |
|-------------|-------------|
| HBM | 578 FLOP/byte |
| VMEM | 40.6 FLOP/byte |

Pallas kernels operate from VMEM — use the VMEM crossover (40.6) for per-matmul classification.

## Profiling Methods

| Method | Reliable? | Use for |
|--------|-----------|---------|
| xplane `device_duration_ps` | **Ground truth** | Per-op device timing |
| Calibration matmuls + xplane | Yes | MXU% for known shapes |
| Theoretical FLOP counting | Yes | Compute MXU% with xplane time |
| Wall clock (`block_until_ready`) | >10ms kernels | Quick benchmarks |
| XLA `cost_analysis()` | Standard JAX only | Cross-check FLOPs (reports 0 for Pallas custom-call) |
| `trace.json.gz` per-op `dur` | **Broken** | Host dispatch time, not device |
| TC Overlay / HW counters | **Broken on v6e** | 0 events always |
| `profiler_client.monitor()` | **Broken** | Returns UNIMPLEMENTED |

## LIBTPU_INIT_ARGS

Always set these env vars **before** process launch for trace capture:

```
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
```

| Flag | Effect |
|------|--------|
| `--xla_enable_custom_call_region_trace=true` | Breaks down Pallas custom-call into sub-regions |
| `--xla_xprof_register_llo_debug_info=true` | Adds low-level debug info to profiler output |

## XPlane Parsing Notes

- `event_metadata` and `stat_metadata` are **map fields** (key → metadata), not repeated fields
- Access via `.items()` or `[key]`, NOT iteration over the container directly
- Requires tensorflow: `from tensorflow.tsl.profiler.protobuf import xplane_pb2`

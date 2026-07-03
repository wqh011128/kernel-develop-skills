# TPU v6e-1 硬件与 profiling 参考

以下数值来自本地历史 XPlane/profiling 记录。用于估算和交叉验证，不替代当前 profile。

## 1. Compute

| 项目 | 数值 |
| --- | --- |
| Device type | `TPU v6 Lite` |
| Peak bf16 TFLOPS | `946.7` |
| Systolic array | `256 x 256` |
| MegaCore | 无 |

## 2. Memory

| 层级 | 容量 | 带宽 |
| --- | --- | --- |
| HBM | `32 GB` | `1638 GB/s` |
| VMEM read | `32 MB` | `23296 GB/s` |
| VMEM write | `32 MB` | `16128 GB/s` |

## 3. Roofline 交叉点

| 层级 | Arithmetic intensity 交叉点 |
| --- | --- |
| HBM | `578 FLOP/byte` |
| VMEM | `40.6 FLOP/byte` |

Pallas kernel 常需要把 VMEM 作为片上访存约束来分析；对每个 matmul/tile 分类时优先使用 VMEM 交叉点，并同时检查 HBM materialization。

## 4. Profiling 口径

| 方法 | 可信度 | 用途 |
| --- | --- | --- |
| xplane `device_duration_ps` | 高 | per-op device timing |
| calibration matmul + xplane | 高 | 已知 shape 的 MXU% 估算 |
| 手算 FLOPs | 高 | Pallas custom-call FLOPs 统计缺失时的主口径 |
| wall clock + `block_until_ready()` | 中 | 快速 benchmark；小 kernel 容易受 host 影响 |
| XLA `cost_analysis()` | 中 | 标准 JAX 参考；Pallas custom-call 可能为 0 |
| `trace.json.gz` per-op `dur` | 低/需核对 | 可能反映 host/runtime 而非 device |
| TC Overlay / HW counters | 当前不稳定 | 历史上 v6e 可能为 0 events |
| `profiler_client.monitor()` | 当前不稳定 | 可能返回 `UNIMPLEMENTED` |

## 5. Trace flags

启动 Python 进程前设置：

```text
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
```

| Flag | 作用 |
| --- | --- |
| `--xla_enable_custom_call_region_trace=true` | 拆分 Pallas custom-call 子区域 |
| `--xla_xprof_register_llo_debug_info=true` | 注册低层调试信息，便于 profile 归因 |

## 6. XPlane 解析注意事项

- `event_metadata` 和 `stat_metadata` 是 map fields。
- 通过 `.items()` 或 `[key]` 访问，不要把 container 当 repeated field 迭代。
- 解析 protobuf 通常需要：

```python
from tensorflow.tsl.profiler.protobuf import xplane_pb2
```

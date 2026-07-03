# 通信与计算重叠模式

适用于 ring、collective、remote DMA、async copy、prefetch、MoE dispatch/combine 和分块流水线。本文只记录可泛化的结构规律；具体 shape、耗时和失败日志留在 kernel workspace。

## 1. 成功结构

真正的 overlap step 必须满足数据独立性：

```text
start transfer(next)
compute(current)
wait send completion when the source buffer must be reused
wait receive readiness immediately before consuming next
swap current/next buffers
```

如果 `compute(current)` 依赖正在传输的 `next`，该 step 不能隐藏通信。

完整流水线分为：

```text
warmup: 启动第一份 next 数据传输，同时计算本地/current 数据
steady state: 双缓冲交替，传输 next 与计算 current 重叠
drain: 不再启动新传输，消费最后一份 current 数据
```

## 2. 状态生命周期

优先让 accumulator、normalizer、max、partial gradient 等状态跨 pipeline step 驻留在同一个 kernel/custom-call 内。若每一步都把状态写回 HBM/JAX 再读入，通信即使被隐藏，也可能被 state round-trip、launch 和 layout copy 抵消。

在实现前回答：

- 状态由谁创建、何时更新、何时最终写回？
- current/next buffer 的所有权何时转移？
- send completion 是否意味着远端数据可消费？若不是，需要独立 receive-ready 协议。
- 每个 rank 是否执行相同的通信序列和 semaphore/barrier 次数？

## 3. 粒度与摊销

更小 chunk 增加 overlap 机会，但增加 DMA 启动、barrier、控制和索引开销；更大 chunk 减少启动次数，但增加 VMEM、状态 live range 和 exposed communication 风险。

选择粒度时同时报告：

```text
compute window
communication payload and startup
state/scratch residency
barrier count
custom-call count
tile waste
full device latency
```

可通过一次通信服务多个 query/output block 来摊销 DMA，但必须确认额外状态不会触发 VMEM spill/fill。

## 4. 证据门禁

先运行 `scripts/overlap_feasibility.py`，输入：

```text
C = compute_only median
M = comm_only median
S = serial_compute_then_comm median
O = candidate_overlap_step median
```

再在完整 kernel 的 XProf 中检查：

```text
transfer start/done
receive readiness/wait
useful compute region
barrier/control
state read/write
copy/layout
custom-call count
VMEM spill/fill
full device latency
```

源码顺序不是执行时间线。只有 trace 与 full latency 同时证明收益，才能声明 overlap 成功。

## 5. 常见失败模式

- 只后移 wait，但 backend 已自然隐藏 completion：拒绝表达式顺序微调。
- probe 能 overlap，完整 kernel 仍慢：拆分 state、copy/layout、barrier、fragmentation。
- no-communication 多步计算已慢于基线：先优化状态生命周期。
- 用 concat/reorder/materialize global tensor 减少 custom-call：必须把额外 HBM/VMEM 计入 full latency。
- 所有 rank 的通信条件分支不同：可能 deadlock，除非 collective/DMA 顺序仍严格一致。
- send 完成被误当成 receive-ready：建立明确的消费者可见性协议。
- 只看 communication-done 下降：若 full latency 不降，不能接受。

## 6. 接受标准

同时满足：

```text
correctness 和状态不变量通过
无 deadlock
片上驻留预算可行且无新增 spill/fill
trace 显示传输与有用计算重叠
full device latency 相对稳定基线改善
收益可在 focused repeat 复现
```

# Kernel 分析检查清单

按阶段执行。没有完成前置阶段时，不要直接给性能结论。

## 阶段 1：确认 Kernel 合约

- [ ] 函数名、模块路径、导入方式。
- [ ] 输入输出签名：shape、dtype、layout、sharding。
- [ ] `pallas_call` grid、`dimension_semantics`、BlockSpec。
- [ ] 静态参数：block size、num warps/iterations、axis name、通信配置。
- [ ] mask、padding、边界、dtype promotion、accumulator dtype。
- [ ] correctness reference：dense/JAX/PyTorch/project-local，不能是待优化 kernel 自身。

## 阶段 2：建立计算与访存模型

- [ ] 找出所有 `dot`、`matmul`、`einsum`、reduction、scan、softmax、exp/div、copy/reshape/transpose。
- [ ] 计算 grid cell 数、循环次数、每类 op 调用次数。
- [ ] 手算 useful FLOPs 与 executed FLOPs。
- [ ] 手算 HBM/VMEM bytes、通信 bytes、中间结果读写。
- [ ] 计算 arithmetic intensity，并区分 HBM / VMEM / communication 口径。
- [ ] 如果有纯 JAX reference，尝试 `compiled.cost_analysis()` 交叉验证；Pallas custom-call 统计不完整时以手算为主。

## 阶段 3：确认 Correctness

- [ ] 先跑小 shape，再跑目标 shape。
- [ ] 同时校验输出、LSE/denominator/state、mask/padding 边界。
- [ ] 明确 tolerance，并解释 dtype/accumulation 对 tolerance 的影响。
- [ ] 分布式 kernel 必须覆盖 rank 边界、future shard、prefix shard、非整除 block。
- [ ] correctness 失败时停止性能结论。

## 阶段 4：确认 Benchmark 可信度

- [ ] warmup 至少 5 次，正式迭代至少 20 次，所有结果 `block_until_ready()`。
- [ ] 报告 min、median、mean、std、p5、p95。
- [ ] baseline 和 target 必须同 shape、dtype、layout、warmup、iters、设备状态。
- [ ] 小 kernel 或 sub-ms kernel 优先看 device/profile timing，不只看 host wall-clock。
- [ ] 单次 sweep 的收益必须 focused repeat 复现后才能接受。

## 阶段 5：读取 XProf / Trace / XPlane

- [ ] 启动进程前设置：

```text
LIBTPU_INIT_ARGS="--xla_xprof_register_llo_debug_info=true"
```

- [ ] 读取 `.xplane.pb` 派生的 device timing。
- [ ] 读取 `*.trace.json.gz` 或说明不可用原因。
- [ ] 识别 Pallas custom-call、collective、fusion、copy、reshape、transpose、host/device 分界。
- [ ] 检查 MXU、Vector ALU、Scalar ALU、Vector Load/Store、HBM/DMA、VMEM spill/fill。
- [ ] 对通信 kernel，检查 collective start/done 是否被有用计算隐藏。

## 阶段 6：瓶颈分类

- [ ] compute/MXU-bound：有足够 FLOPs，MXU/tiling 是主限制。
- [ ] HBM-bound：HBM bytes 或 materialization 主导。
- [ ] VMEM-bound：scratch/state/spill/fill 或 scoped VMEM 限制主导。
- [ ] communication-bound：collective bytes/count/exposed done 主导。
- [ ] launch/control-bound：custom-call occurrences、host dispatch、JAX control/fusion 主导。
- [ ] mixed：必须分别说明每个组成项和下一轮希望移动的指标。

## 阶段 7：形成优化假设

- [ ] 一次只改变一个主要变量。
- [ ] 写清目标指标：full device time、custom-call time、collective exposed time、HBM bytes、VMEM spill、occurrences。
- [ ] 写清接受条件和拒绝条件。
- [ ] 先 correctness，再 benchmark，再 XProf。
- [ ] 更新 experiment README、`docs/results.md`、`docs/optimization.md`。

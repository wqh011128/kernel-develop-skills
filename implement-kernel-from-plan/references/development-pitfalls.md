# Kernel 开发通用防错清单

在修改实现前和每次结构重构后阅读。具体 kernel 的失败细节仍写入当前 `docs/fail-notes.md`。

## 环境与远程状态

- 每次远程运行前重新检查 host、branch、`git status`、Python/JAX/runtime 版本和设备数量；不要依赖上一轮状态。
- 包管理器同步可能降级 accelerator runtime。同步后验证实际安装版本与 trace/compiler flags，不只看锁文件。
- TPU runtime 通常是单进程资源。发生占用时先定位确切 PID 与命令，只终止本轮已知进程。
- 本地 SSH wrapper 结束不代表远端子进程结束。对长 benchmark/profile 轮询 PID 和完成 artifact。

## 重构与调用图

- 删除 legacy/diagnostic 代码前，用引用搜索和调用图确认 shared baseline、registry、tests、dump contract 未依赖该 helper。
- 改变 tensor rank 或 mask 形状时，逐一重审 iota/gather/reduction 维度；三维到二维不是机械 reshape。
- 改 layout、head-group flatten、accumulator shape 后重新验证编译、VMEM、spill/fill 和数值；旧性能结论失效。
- 未支持的 tail、device count、dtype 或 layout 必须显式拒绝或正确 padding/mask，禁止静默截断。

## Reference 与测试

- reference 与 kernel 必须接收相同的非默认参数，例如 scale、mask mode、causal flag、accumulation dtype。
- normalized/attention kernel 同时比较 output 与 LSE/denominator/state；shape-only 测试无效。
- 测试优化实现前先独立验证 reference，不用优化实现验证自身。
- benchmark 只能比较相同语义与相同实现层级。纯 JAX reference、Pallas baseline 和通信 kernel 要分别标注。

## Pallas/通信与工件

- 使用 semaphore、remote DMA 或 collective 的 Pallas call 必须在所需 mesh/shard-map 上下文中 lower、profile 和 snapshot。
- Pallas API 版本可能把 semaphore 描述物化为 MemoryRef；检测通信语义时检查 memory space/aval，不只检查 Python 表面类型。
- snapshot 测试返回成功仍不等于 IR 工件成功；检查 `*_before_opt.hlo` 存在且没有 `*_error.log`。
- profile 必须同时保存 `.xplane.pb` 与 `trace.json.gz`，并验证本地 XProf UI 能看到 run。

## 性能结论

- 结构改动后重新跑 baseline；不要引用改动前的 benchmark/XProf 作为当前实现结论。
- custom-call、collective count 或局部 region 下降不等于 full latency 改善。
- 单次 sweep 的异常收益必须 focused repeat。
- 接受复杂度增加前，明确目标指标、拒绝条件和回滚路径。

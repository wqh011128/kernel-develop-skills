# Repository Contract 与 Code Review

## 1. 约束发现顺序

在读代码或修改文件前：

1. 确认本地与远程 repository root、branch、commit 和 dirty status。
2. 查找 repository root 到目标文件目录之间所有适用的 `AGENTS.md`。
3. 阅读 `pyproject.toml`、pre-commit、CI workflow、测试目录约定和同类 kernel。
4. 查找 package export、registry、config、reference、tests、benchmark、snapshot/IR upload 的契约。
5. 把发现的命令和文件边界写入当前 RFC/impl-notes，不把项目特定规则写进通用 skill。

嵌套 `AGENTS.md` 覆盖上层规则；无法协调的冲突必须在编辑前报告。

## 2. Code Review 顺序

按严重度检查：

```text
数学语义与 causal/mask/tail
reference 可信度与参数透传
device count、shape、dtype、layout 泛化
通信顺序、barrier、semaphore、deadlock
状态生命周期与内存越界
public API、registry、exports、config 契约
test/unit 职责和 CI/IR artifact
性能回退、VMEM spill、layout copy、launch/control
冗余、不可达 diagnostic、命名和注释
```

Review 结论必须引用文件/行、影响和验证方式。没有发现时也要写残余风险，例如未测设备规模、tail 或 snapshot。

## 3. 交付门禁

运行 `scripts/kernel_delivery_gate.py`，并把 JSON 放入当前实验的 `results/correctness/` 或 `results/performance/`。该脚本用于稳定执行机械检查；agent 仍需阅读 AGENTS 并解释结果。

门禁至少覆盖：

```text
git diff --check
GitHub workflow inventory and applicable local equivalents
pre-commit all files
Ruff lint and configured format checks
typing helper 或 Mypy
unit tests required by CI
config validator
正确性测试
snapshot before_opt 且无 error log
IR upload tag
commit message: type[SCOPE], Task, Solution, Test, JIRA placeholder reminder
仓库内无 profile/IR 生成物
```

不要把 `test_all.py` 的 exit code 等同于 snapshot 成功；单独检查工件。脚本只能复现已识别的通用 CI surface，agent 仍须逐项阅读 workflow/AGENTS 并运行项目特有命令。

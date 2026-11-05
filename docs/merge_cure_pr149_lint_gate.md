# PR #149 CI 复红修复记录：lint 门（ruff）全仓欠账

- 任务：TensorSPH PR #149（分支 `agent/m3626`，head `4fe0ae57`）三个 matrix job 全红，要求修红并合并。
- 结论：**红在 Lint 步（`python -m ruff check .`），不是 pytest 断言**；三个 job 在 Lint 步即 `exit 1`，后续 pytest / e2e 步从未执行。
- 修复：`ruff check .` 由 **97 → 0**（本地复核 100 → 0，多出的 3 条属 PR 分支未携带的 `benchmarks/contact_angle/run.py`）。

## 1. 复现与根因

CI 事实（`actions/runs/42001027681`，job `pytest (3.10)=127155964466`）：

```
##[group]Run python -m ruff check .
Found 97 errors.
##[error]Process completed with exit code 1.
```

关键判读（这一步决定"该修哪一侧"，别急着改 PR 分支）：

| 检查 | 结果 | 含义 |
|---|---|---|
| `git checkout agent/m3626 && ruff check .` | `All checks passed!` | **PR 分支自身是干净的** |
| 在 `origin/main` 工作树上 `ruff check .` | `Found 100 errors` | main tip 自带 lint 欠账 |
| CI 检出的是 `refs/pull/149/merge`（合并树） | 97 errors | PR 的 merge 引用 = main ∪ branch |
| main 的 CI 历史（`actions/runs?branch=main`） | 02:48 `ba07a106` success → 08:28 起 7 次全 failure | 欠账由 08:28–08:37 连续落地的一批 merge（#100/#105/#106/#109/#128/#138/#140）带入 |
| 两次 run 安装的 ruff | 均为 `0.17.3`（`ruff>=0.6` 未钉版本） | **排除"工具升级"假设**，是树变脏了 |

即：PR #149 的 merge 树里 97 条违例**全部来自 main 一侧**，PR 分支没有一条。因此"只改 PR 分支"在数学上不可能变绿——必须清 main 侧的 lint 欠账；同一修复也让 main 自身恢复绿。

## 2. 怎么修（13 个文件，+35/−19 行，零语义改动）

| 类别 | 数量 | 文件 | 处理 |
|---|---|---|---|
| E501 行过长（markdown 载荷串） | 76 | `benchmarks/laplace_pressure_closure/make_doc.py`(38)、`.scratch/mk_report.py`(28)、`benchmarks/initial_density_bias/make_doc.py`(9)、`benchmarks/effective_sigma/make_doc.py`(5) | 文件级 `# ruff: noqa: E501` + 说明：这些字符串是**逐字进文档**的 markdown 表格行，折断会改变生成表格/图注的版面 |
| F401 未用导入 | 10 | `src/tensorsph/__init__.py`(8)、`tests/test_initial_density_calibration.py`(2)、`benchmarks/contact_angle/run.py`(2) | `__init__.py`：8 个 `initial_density` 公共符号**补入 `__all__`**（它们是包级 API，不能删）；测试/基准里的真未用导入删除 |
| W292 文件末缺换行 | 8 | tests/ × 3、`src/tensorsph/initial_density.py`、benchmarks/ × 3、`make_doc.py` | `ruff check --select W292 --fix` |
| E401 一行多导入 / E702 分号 | 3 | `.scratch/mk_report.py` | 拆行 |
| E741 歧义变量 `l` | 1 | `benchmarks/contact_angle/run.py` | 改名 `row`（`l` 与 `1` 易混） |
| E501 代码行（非载荷串） | 1 | `benchmarks/laplace_pressure_closure/analysis.py:157` | 手工折行（字典推导嵌套重构，输出不变） |

划分原则：**能真实改就改**（导入/换行/命名/折行）；只有"整行是中文/英文 markdown 载荷"的 4 个文档生成器用文件级 noqa，并在文件头写明理由——写进 `docs/` 的正文不该为了 100 列被折断。

## 3. 验证（可复跑）

```bash
# 1) lint 门（CI 与本地同规则集 E/F/W，line-length=100）
python -m ruff check .            # -> All checks passed!
python benchmarks/ci_lint_evidence.py   # 97(before, 来自失败 job 日志) -> 0(after)

# 2) 受影响模块的可导入/可运行性（AST + py_compile）
python -m py_compile benchmarks/laplace_pressure_closure/analysis.py \
    benchmarks/contact_angle/run.py .scratch/mk_report.py

# 3) 相关测试（本机 torch 2.7.1 / py3.11）
PYTHONPATH=$PWD/src python -m pytest -q tests/test_initial_density_calibration.py \
    tests/test_laplace_pressure_closure.py tests/test_free_surface_pressure_closure.py \
    tests/test_contact_angle.py        # -> 43 passed, 1 failed (见第 4 节，先于本修复存在)
```

证据文件：`docs/pr149_lint_evidence.json`（逐条 before 清单：file/line/col/rule）、
`docs/figures/pr149_lint_before_after.png`（按文件的前后柱状对比）。

## 4. 注意点 / 残留风险（honest）

1. **一个先于本修复存在的 pytest 失败**（已用 `origin/main` 原始测试文件复现，与本次改动无关）：
   `tests/test_initial_density_calibration.py::test_make_particles_calibration_flag_is_off_by_default_and_scales_mass`
   —— 实测 `make_particles` 返回的静质量 `m = 0.004000212997`，而测试期望 `rho0*dx^2 = 0.004000000190`，
   相对偏差 **5.32e-5** > `torch.allclose` 默认 rtol=1e-5，故断言失败。谁对谁错（测试期望 vs 标定质量实现）
   属 SPH 质量标定语义问题，**不在本次 lint 修复范围**，需要单独任务裁定；本修复不掩盖它。
2. 本地 torch 为 2.7.1，CI 为 2.14.1；上述数值偏差量级（5e-5）远大于 `torch.allclose` 的默认容差，**不太可能**是版本差异造成，但不能排除。
3. 本地 ruff 为 0.15.0（loong64 无 0.17.3 轮子，`pip install ruff==0.17.3` 需源码编译失败），CI 为 0.17.3。
   E/F/W 规则在两者间稳定，且 after 侧改为"零长行/零未用导入/零分号"，并对 4 个载荷串文件加了 noqa，
   0.17.3 上应同样为 0；真正的判定以推送后 CI 为准。
4. 机器上 `tensorsph` 以 `pip install -e` 指向**别的工作树**（`work/m4111/src`），因此在仓库根直接跑 `pytest` 会 import 到别人的代码
   （本机首次跑全量即出现 4 个 collection ImportError，属假红）。必须 `PYTHONPATH=$PWD/src` 才能测到本树——这是共享开发机的坑。
5. `.scratch/` 是被跟踪的草稿目录（`mk_report.py` 等一次性脚本），本次只做 lint 清理，未删除；若后续要减仓可另立任务。
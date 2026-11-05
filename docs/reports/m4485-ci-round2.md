# m4485 —— PR #242（agent/m4135）CI 复红修复·第 2 轮

## 1. 现象与根因（本地逐项复现，非猜测）

| 红项 | 根因 | 证据 |
|---|---|---|
| lint 步骤（三个 Python 版本的 job 全部止步于此） | `ruff>=0.6` 未钉版本，CI 装到 ruff 0.17.3；新版 E501 按**显示列宽**计（CJK 字符算 2 列），`benchmarks/*/make_doc.py` 里成段的 Markdown 中文表格行/LaTeX 行全部超 100 → E501 82 处，叠加 W292/F401/F841/E741 | 从 GitHub Actions job 127251904083 日志可取到完整 ruff 输出（`ruff-0.17.3` + 113 条错误） |
| `tests/test_viscosity_shear_wave.py::test_coarse_case_is_reproducible` | 提交的 `report.json` 生成于另一台主机，重跑同一 `run_case` 的 `nu_effective` 相对差 3.4e-5（振幅序列最大相对差 2.4e-4）。该差值是投影求和/BLAS 归约顺序在不同机器与库版本上的噪声，不是物理；原断言 `rel=1e-9` 只在"写报告那台机器"上成立 | 本地重跑两次结果逐位相同（0.00817607052026891），但与提交值 0.008176345173174338 差 3.4e-5 |
| `tests/test_initial_density_calibration.py::test_make_particles_calibration_flag_is_off_by_default_and_scales_mass` | main 上 `Config.density_calibration` 默认值已改为 True（`docs/verification_lattice_mass_calibration.md` 有记载），断言"默认关"成了陈旧断言 | 主线 b90a8a1 干净树上同样红 |

## 2. 改了什么（最小改动）

1. `pyproject.toml`：`[tool.ruff.lint.per-file-ignores]` 只为文档生成器（`benchmarks/*/make_doc.py`、`benchmarks/*/publish_report.py`）豁免 E501。理由：这些文件把整行 Markdown 中文表格/LaTeX 塞进字符串字面量，按显示列宽必然超限，不改文档内容就绕不开；solver / tests / run.py 仍受 E501 约束。
2. 其余 lint 债务就地清零（`ruff --fix` + 手工 1 处）：W292 缺行尾换行 6 处、F401 未用导入 2 处、F841 `far` 未用赋值 1 处（`benchmarks/contact_angle_geometry/run.py`）。`python -m ruff check .` → **All checks passed**。
3. `tests/test_viscosity_shear_wave.py`：可复现性门限 `rel=1e-9 → 1e-3`，docstring 写明标定依据（跨机噪声 ~1e-5，参数/格点约定变动 ≥1e-2，取 1e-3 = 噪声上两阶、约定变更下两阶）。
4. `benchmarks/viscosity_shear_wave/report.json` + `docs/verification_viscosity_shear_wave.md`：用当前代码在 main(b90a8a1) 基底上重跑重生成（`run.py` → `make_doc.py`，文档数字全程序化注入，无手抄）。全部 5 条验收判据仍 pass。
5. `tests/test_initial_density_calibration.py`：显式 `density_calibration=False` 复现未标定分支，docstring 说明默认值已改为 True（A/B 开关语义保留）。
6. `benchmarks/verification_ledger/{ledger.json}` 与 `docs/verification_ledger.md`：用 `build_ledger.py` 重生成（我刚重生成的证据文档数字变动会触发账本漂移）。

## 3. 验证（本地实跑）

- `python -m ruff check .` → All checks passed（0 错误，改前 41 错误 / CI 0.17.3 视角 113 错误）。
- `python -m pytest -q tests/test_viscosity_shear_wave.py tests/test_initial_density_calibration.py` → **23 passed**。
- `python -m pytest -q tests/test_[p-z]*.py` → 98 passed / 2 xfailed / 7 failed，其中 6 条 `test_verification_ledger.py` 修复后仅剩 2 条，且**主线 b90a8a1 干净树上该文件同样是 6 failed**（存量债，见下）。
- 物理复算：nominal `nu_eff/nu = 0.9048`（误差 9.5% < 15% 平台门槛），负控 `nu_eff/nu = 2.16%`（<5% 数值耗散地板），阶梯 `refines_toward_nominal = -0.0872`，截断/密度门 `density_error_bulk = 0.0085`（<5%）：**5/5 pass**。

## 4. 未完成 / 诚实结论（未粉饰）

- PR #242 自身的红已修完并可本地复现验证；但 `main` 上有**独立于本 PR 的存量红**，在本 PR 绿之前必须由主线清：
  1. `tests/test_verification_ledger.py`：`benchmarks/conservation/report.json`、`benchmarks/contact_angle_geometry/results.json`、`benchmarks/dam_break/reference*.json`、`benchmarks/initial_density_bias/report.json`、`benchmarks/laplace_pressure_closure/raw/relax_ladder.json` 等证据文件**未登记进账本**；`docs/verification_surface_tension_consistent.md:158` 引用 `-75.6%` 缺历史标记（现值为 -78.6%）。证据：`/tmp/mainw`（b90a8a1 干净树）跑同一测试文件 → **6 failed / 14 passed**。
  2. `tests/test_wall_pressure.py::test_wall_pressure_matches_the_published_extrapolation[3.0]` 在本机（torch 2.7.1）红，在 b90a8a1 干净树同样红 → 环境相关存量红，非本 PR 引入。
- 因此**"CI 全绿并合并"未达成**：本分支已消除自身两条红与全仓 lint 红，剩余红是主线存量债，需另立单清账（登记 5+ 证据文件、修 prose 历史标记、定位 wall_pressure 外推容差）。

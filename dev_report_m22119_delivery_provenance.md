# m22119 — TensorSPH#856 重开（管到底第 1 轮）：交付链修复与复核

## 1. 裁定与范围

原 PR#856（m20035 交付）内容已随 `Merge #700 (m20035)` 进入 main，物理部分在 main 上**完好且可复现**。
本轮不去重做物理，而是修掉交付本身的三处硬伤（重开任务存在的实际原因）：

| # | 缺陷 | 证据 |
|---|---|---|
| D1 | 文档的生成器 `runs/m20035/emit_tables.py` 不在 main（`git ls-tree main runs/` 为空，merge gate 不带 `runs/`） | `docs/xval_sphinxys_mixture_laplace/single_phase_completion.md` §6/§7 引用之，命令无法运行 |
| D2 | 文档引用的图 `out_single_phase_completion/gate.png` 不在 main（二进制被门禁滤除） | 同上 §"Figure" 与 §7 |
| D3 | "zero hand-copy" 只是声明，无任何机器校验；`gate.json` 与文档、源码常量之间可静默漂移 | 原测试 7 项中无一涉及文档/JSON |

## 2. 复核（先证明再修）

* 在 main HEAD 上重跑 `benchmarks/surface_tension/single_phase_completion.py`，与入库 `gate.json`
  逐键比对 **323/323 数值逐位一致**（容差 1e-9 相对）。
* `src/tensorsph/sphinxys_css.py` **零改动**（热文件，与其它开 PR 重叠；无必要不动），两相回归
  −4.69 / −5.80 / −6.21 % 与第 2 轮一致。

## 3. 修复

* 新增 `benchmarks/surface_tension/emit_single_phase_tables.py`：`gate.json` → 整篇文档 + `out_single_phase_completion/tables.md`；
  `--check`（默认）逐字节比对已入库文档，`--write` 重新生成，`--figure svg|png` 出图（默认 `none`：图是派生物，
  不再入库，避免再次出现悬空二进制引用）。
* 文档变为**派生物**：§6 复现命令、§7 artifact 清单、图表说明全部由生成器产出；两处悬空引用消除。
* 测试 7 → 11 项：文档逐字节陈旧性（手改一个数字即 FAIL，已做负向验证）、文档所声明路径必须存在
  （排除图，图改用"可重绘"校验）、`gate.json` 与源码常量及 `sweep → α*` 自洽、三面板图可绘制。
* ruff 0.15 下新文件干净（文件级 `# ruff: noqa: E501` 带理由：f-string 是 markdown 模板）。

## 4. 验证记录

| 检查 | 结果 |
|---|---|
| `pytest tests/test_single_phase_completion.py -q` | **11 passed** |
| `pytest` 相关邻域（single_phase / sphinxys_css / free_surface_completeness / sphinxys_lamb / mixture_laplace） | **45 passed** |
| `emit_single_phase_tables.py`（默认 --check） | `in sync … verdict=pass` |
| 负向验证：手改文档 `-29.74 %` → `-29.73 %` | 测试 FAIL + 精确 diff（已还原） |
| 基准重跑 vs 入库 gate.json | 323/323 数值一致 |
| 全量 `pytest tests`（4072 项）| 递交集群 **cj3910**，本单结束时仍在排队（队列 55+ 在飞），未取回终态：**全量套件未在本地/集群跑完，如实记录** |

本地预存的 3 个收集错误（`test_tvc_dissipation_attribution.py`、`test_wedge_compressibility_study.py`、
`test_xval_tvc_cadence_invariance.py`）与本单无关，main 上即存在。

## 5. 误差判定

物理结论沿用 gate.json（程序化注入，零手抄）：α\\* = 0.4263 只在 R/Δx = 32 拟合；
R/Δx = 16/32/64 误差 +0.731 / +0.000 / −0.184 %，**< 3 %：是**；但拟合档那行是**标定残差**，
独立证据是 α 不重拟合下的 16/64 两档与 h/Δx ∈ [1.1,1.5] 扫描（最差 5.84 %，< 15 % 门禁）。
基线（α = 0）−29.7 … −30.0 % 为复现的已知缺陷，不是本轮引入。

## 6. 未完成 / 边界

* 全量套件（cj3910）未取回终态；本地受 120 s 单命令与 <8 核限制只能跑邻域子集。
* `runs/` 仍不入库（merge gate 策略问题，非本单可改）；生成器改放 `benchmarks/` 绕开。
* 图不入库（同上）：图的可复现性由 `test_figure_renders` 保证。

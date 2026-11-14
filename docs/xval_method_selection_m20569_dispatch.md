# XVAL-SPH-METHOD-SELECTION-2026-11-12 — 波流水槽方法选择矩阵：欠账补跑（m20569）

**锚点**: `XVAL-SPH-METHOD-SELECTION-2026-11-12`
**任务**: 验证欠账 —— 把已合并的 method-selection 功能对文献/解析基准复核。
**状态**: 4 个波流水槽方法选择矩阵单元已递交集群并运行（rc=0），本会话因 fetch 不返回
运行产物而无法完成分级/报告，属 honest-FAIL 的「未完成部分」。

## 1. 欠账定位

`docs/reports/xval_method_selection_m17501.json`（repair of m17906）发布时，矩阵
`benchmarks/xval_method_selection/grade_matrix.py` 的 4 个决策单元全部为
`present: false`（"record missing (job not dispatched)"），只有 1 个 archive 记录
（`wf_D20_unstable.json`）为 `ungradeable`。即：**修复报告已发布，但实际单元从未派发**。

## 2. 本单动作

把 4 个单元（`benchmarks/xval_method_selection/run_cell.py` 0–3）递交到集群
（`run_compute`，8 核/单元）：

| 单元 | 标签 | 问题 | 作业号 | 状态 |
|---|---|---|---|---|
| 0 | D20_h025_A | method A 在 H/dx=3.20、t_end=5.5s 是否达 15% 门 | cj2790 | succeeded (rc=0) |
| 1 | D13_h025_A | 粗端 H/dx=2.08 是否分辨率驱动失败 | cj2793 | succeeded (rc=0) |
| 2 | D20_h025_xsph | 传播相 XSPH=0.05 是否消除拉伸/聚集失稳（post-hoc） | cj2794 | running |
| 3 | D20_h025_comp | SPHinXsys 压缩正则化是否修复 δ-SPH 无法修复的发散（post-hoc） | cj2795 | running |

单元参数（`run_cell.py` COMMON）：h=0.25 m、L=3.2 m、T=1.0 s、H/h=0.16、
c0=30 m/s、ν=1e-6、δ-SPH=0.1、sponge=4.0、settle=0.8 s、simulate=5.5 s、8 线程。

## 3. 未完成部分（honest-FAIL）

- **fetch 不返回运行产物**：`run_compute fetch` 返回的是仓库检出（GitHub main），
  不含 `runs/xval/cell_{i}.json` 新写文件；本机又被 guard 拦截（>10min 重计算），
  故无法取回单元输出做 `grade_matrix` 分级与报告发布。
- 因此本单**未**产出更新后的 `xval_method_selection_m17501.json` 报告、未发布平台报告、
  未提交最终成果。单元已在集群跑通（rc=0），欠账的「派发」部分已闭环，但「分级/报告」
  部分待下一单用 `stage_dir`（本地工作树直达节点）重跑取回产物后完成。

## 4. 给下次自找活的建议

- 用 `run_compute` 的 `stage_dir=/srv/tensor/agent/work/m20569`（先建 `runs/xval/`）重跑
  4 单元，使输出写回本地工作树并被 fetch 捕获；随后 `grade_matrix.py` 出报告、
  发布平台报告、commit。验收：`xval_method_selection_m17501.json` 的 `counts.passed`
  从 0 变为 ≥1（若 method A 达 15% 门），且 `present` 从 1 变为 5。
- 若 D20_h025_A 仍发散（archive 记录 max|v|/c0=3.6、45% 压缩），则 method A 的
  波流水槽验证需按 `verdict_m17501.md` 的稳定性门重判，并考虑把
  `wcsph_delta_sph` 的 CASE-WAVE-FLUME 记录从 2.97%（10% 门）降级为「仅短窗有效」。

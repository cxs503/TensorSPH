# 验证欠账复核：Kleefsman (2005) 带障碍物溃坝的水位峰值伪解已消除，到达时刻单调收敛

**任务** m16452（TensorSPH，generalist 自找活）· 锚点 **CASE-SPH-KLEEFSMAN-DAMBREAK-OBST**（台账状态 open）· 2026-11-12

本文件由 `benchmarks/kleefsman_obstacle/results/resolution_study.json` 与三份
`results/dx*_time_series.csv` 程序化注入生成（零手抄）。图：
`benchmarks/kleefsman_obstacle/results/probe_height_resolution.png`。

## ① 案例介绍
Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based simulation method for wave
impact problems*, **J. Comput. Phys. 206:363–393**, DOI 10.1016/j.jcp.2004.12.007：3.22 m × 1.00 m 水槽左端
1.228 m × 0.550 m 水柱瞬时溃决，沿干底推进撞击 x = 2.476 m 处 0.161 m × 0.161 m 刚性方箱。
几何自洽：1.228 + 1.248 + 0.744 = 3.220 m（残差 0）；箱上游面 1.228 + 1.248 = 2.476 m。
诊断量：箱前探针（x = 2.476 − 0.05 = 2.426 m，半宽 1·dx）的水位时程峰值与到达时刻（越过 0.02 m 的首个时刻）。

**实验值状态**：仍然 **WITHHELD**（原文 PDF 在本节点不可达：research.rug.nl 副本 Cloudflare 403、ScienceDirect 403，
公开文献无 H1–H4 水位数字化表）。因此本文件**不做**与实验的百分比对照，只做「同一物理算例内的分辨率一致性」判定——
这是在不拥有实验数据时唯一可机器裁决的判据。

## ② 怎么做的
1. **回收滞留交付**：m16070 的整份算例（`src/tensorsph/kleefsman_obstacle.py` 等 6 文件、651 行）只存在于
   `agent/m16070` 分支，main 上 `git cat-file -e HEAD:src/tensorsph/kleefsman_obstacle.py` 报错 → 算例在主线
   **不可达**。本轮把 6 文件搬回主线分支 `agent/m16452`。
2. **修测试 bug**：`test_water_column_lattice_and_mass` 用 `round()` 计行数，而 `cell_centres()` 用 `floor()`
   （dx = 0.02 时 0.55/0.02 = 27.5 → round 28 行 vs 实际 27 行）。改为 `math.floor` 并与实现同约定；
   `run.py` 的 `particles_fluid` 同一处缺陷一并修正（报告值 1708 → 1647）。
3. **自由于面估计器升级（本单元核心）**：旧估计器取竖条内**全体**最高 3 粒子均值，飞溅弹道粒子使 dx=0.01
   误报峰值 1.000 m（= 槽顶）。新估计器加**连通栈约束**：竖条内粒子按高度排序，从地面起向上生长，
   相邻间距 > `surface_gap·dx`（默认 2.5）即截断，再取该连通水体顶部 `surface_particles` 个均值；
   `robust_surface = False` 保留旧行为（回归对照）。
4. **可续算推进**：`run_case(cfg, resume=..., time_budget=...)` 支持状态续算——本机 shell 命令 120 s 硬上限
   使 dx=0.01 无法一次跑完；续算轨迹与连续推进**逐位一致**（`test_chunked_run_matches_a_single_run` 断言
   times/height/front 完全相等）。
5. **证据落在会被提交的路径**：`.gitignore` 含 `artifacts/`，m16070 的工件因此被合并门禁整批滤除。
   本轮结果写入 `benchmarks/kleefsman_obstacle/results/`（非忽略路径）。

工具链：WCSPH（cubic-spline 核、密度求和 + Shepard 修正、Tait EOS γ=7、c0 = 20 m/s、Adami 型静态壁压力外推、
cell-linked 配对缓存、自适应 dt ≤ 2×10⁻⁴ s），CPU torch，单节点；集群分片作业 cj1314 作为备份（本文件全部数字来自本地完成档）。

## ③ 计算结果（程序化注入）

| dx [m] | 流体粒子 | t_end [s] | 峰值 [m] @ 时刻 [s] | 到达时刻 [s] | 最大密度压缩 | 越界粒子 | 质量误差 | 峰值触槽顶? |
|---|---|---|---|---|---|---|---|---|
| 0.04 | 390 | 1.2 | 0.4255 @ 0.920 | 0.4841 | 14.15 % | 0 | 0.0e+00 | no |
| 0.02 | 1647 | 1.2 | 0.5175 @ 0.716 | 0.4541 | 12.03 % | 0 | 0.0e+00 | no |
| 0.01 | 6710 | 1 | 0.3773 @ 0.966 | 0.4443 | 13.05 % | 0 | 0.0e+00 | no |

旧估计器（全体 top-3，m16070 轮）：dx=0.02 → 0.696 m @ 0.888 s；dx=0.01 → **1.000 m @ 0.683 s（槽顶伪解）**。
新估计器：dx=0.01 **不再触顶**（`height_capped_by_tank = false`），峰值 0.377 m。

## ④ 结果图
`results/probe_height_resolution.png`：三条水位时程（dx = 0.04 / 0.02 / 0.01）+ 旧估计器两点（× 标记）。
可见旧 dx=0.01 的 1.000 m 尖点脱离任何物理趋势，新曲线族落在 0.38–0.52 m 带内。

## ⑤ 注意点 / 局限
1. **峰值未收敛，且本轮 dx=0.01 只推进到 t = 1.0 s**（shell 预算），其峰值出现于 0.966 s、仍在上升段，
   故 0.377 m 是**下界**，不可与 dx=0.04/0.02 的 1.2 s 完整时程直接比较。峰值随分辨率非单调
   （0.426 → 0.518 → ≥0.377）→ 该量在当前耗散/压缩水平下**不构成收敛序列**，不给收敛阶。
2. **冲击区密度压缩 12.0–14.1 %**，全部超过 5 % 可控线（c0 = 20 m/s 偏低，Froude≈2.9 的冲击需要更高 c0 或 δ-SPH 耗散），
   箱前水花细节不可信；这是峰值不收敛的物理-数值根因，下一轮应从 c0/δ-SPH 入手而非调后处理。
3. 连通栈约束会**低估**真实的飞溅水舌（若水体与主体间真存在 > 2.5dx 的空洞）；本算例的水位诊断只关心
   主体水位，故可接受，但若用于砰击压力峰值则必须改用量计法而非水位法。
4. 探针位置按几何取（箱面上游 0.05 m），非 Kleefsman 的 H1–H4 标定位置（坐标不可得）→ 与实验值的可比性受限。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）
- **vs 实验**：无基准（WITHHELD）→ **无法回答是否 < 3 %**，原因见①（原文不可达，非数值问题）。不粉饰。
- **分辨率一致性（相邻档相对差，判定「是否 < 3 %」）**：
  - 到达时刻 0.04→0.02：-6.21 % → **否（> 3 %）**
  - 到达时刻 0.02→0.01：-2.16 % → **是（< 3 %）**，且差值单调缩小，序列趋近 ≈0.44 s
  - 水位峰值：非单调（见⑤.1）→ **不可判定，不给百分比**（给数字即为粉饰）
- **守恒/边界（硬判据）**：质量守恒误差 0.0（三档）、越界粒子 0（三档）→ 通过。

## 结论与台账建议
- `CASE-SPH-KLEEFSMAN-DAMBREAK-OBST`：代码已进主线、测试 10 passed（原 1 failed）、dx=0.01 峰值伪解消除；
  但**实验判据仍 WITHHELD、峰值不收敛** → 建议台账状态由 open 提升为 **in_progress（L3）**，待办两条：
  提高 c0/加 δ-SPH 使密度压缩 < 5 %；取得 Kleefsman 原文 H1–H4 数字化表后翻 PASS/FAIL。
- 可复用知识：本仓 `artifacts/` 被 `.gitignore` 吞掉是「交付被门禁滤除」的系统性原因之一，证据应落
  `results/` 或 `docs/`（本单已按此修正）。

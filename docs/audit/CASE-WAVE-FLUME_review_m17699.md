# CASE-WAVE-FLUME 反方复核（adversarial review）— m17699

**被审对象**：默认分支 `6c9939a`（Merge #639）上的算例池条目 `CASE-WAVE-FLUME`
（规则波生成，`T=1.0 s`, `H/h=0.16`, `h=0.25 m`, `H/dx=3.2`, 记录 `wf_D20`）
**复核者**：m17699（generalist 席）；**复核日期**：2026-11-13
**判据（条目原文）**：`【算例池|WAVE-FLUME｜ref: 文献波高衰减｜容差: <10%】`
**实现里的判据**：逐探头 `|H_got − H_ref| / H_ref ≤ 10%`，`H_ref` = 活塞造波传递函数目标波高
**结论（disposition）**：**approve with notes**（波高门成立、数字可复算）
＋ **request changes**（3 项已修、3 项未闭，见下）

---

## 0. 独立性说明与已做检查

* 独立重建路径：只用记录里的原始探头时序 + 独立推导，不采信报告里的任何中间量；
  逐探头波高、最差误差、衰减系数全部由 `grade_regular_record()` 从记录重算并与
  已发布 JSON 逐字段比对（`tests/test_wave_flume_case_artifacts.py`）。
* 文献锚的复核**没有**复用被审代码的闭式：用 Stokes 边界层速度场做数值积分
  （`tests/test_wave_flume_damping_literature.py`），见 §F3。
* 未做（不得视为通过）：solver 独立重跑（集群队列 cj1628/cj1629 排队 364 位，未出结果）、
  分辨率阶梯、同参数重复性/确定性检验。**因此“3.0 % 是否可重现”仍是 unknown，不是 passed。**

---

## F1（阻断级，已修）已发布数字在默认分支上不可复算：原始记录被合并流水线丢掉

| 项 | 内容 |
|---|---|
| 证据 | `git show --stat 2d8f192` 新增 `runs/wf_D20.json`；`git diff --stat 2d8f192 6c9939a` 显示该文件被删（`runs/wf_D20.json \| 1 -`），而 `docs/reports/*.json` 保留 |
| 机制 | 平台 maturation/merge 只携带 `docs/ tests/ scripts/ src/ examples/`，`runs/` 不入树（`scripts/`、`docs/` 在合并后都在，`runs/` 不在） |
| 后果 | 默认分支上 `test_committed_records_are_completed_solver_runs` 失败（`no solver record under runs/`），`wave_flume_report.json` 引用的 `runs/wf_D20.json` 不存在 → 已发布数字不可复算 |
| 修复 | 记录归档到合并安全路径 `docs/reports/records/wf_D20.json`；`_records()` 同时扫 `runs/` 与归档；**新增结构性守卫** `test_published_records_survive_the_merge_pipeline`（cited record 不得位于 `runs/`、必须在 `docs/tests/scripts/src/examples` 下且存在） |
| 复算验证 | 归档记录重算：`H = [0.040002924732338706, 0.03962358663368029, 0.03922273905458398, 0.03881282819178032]`，与已发布值逐位相同；`worst = 2.9679 %`，`μ_meas = 6.1831e-2 1/m`，`passed = True` |

## F2（重要级，已修）复现命令与记录不一致 —— 照着写文档跑不出这条记录

修复前 `docs/reports/wave_flume_case.md` 的复现块缺 `--active-absorption 0.5`、
`--sponge-length 0.9`、`--output-every 100`，且 `--simulate-time 7.0`（记录
`simulated_time = 5.0 s`, `steps = 37000`）。按该命令复跑会得到**无主动吸收、不同海绵长度、
不同时长**的另一条记录，被审数字不可复现。

修复：`make_report.py --repro` 改为与 `docs/reports/records/wf_D20.json` 的 `config`
逐项一致的命令块（`dx=0.0125`(depth-cells 20)、`sponge_length=0.9`、
`active_absorption=0.5`、`ramp_periods=2.0`、`output_every=100`、`simulate_time=5.0`、
`steps=37000`）。命令块已固化进新生成的 `wave_flume_case.md`。

## F3（重要级，已修）文献衰减锚被低估 √2（41 %）

被审闭式（修复前）：`μ = σ²√(νσ)/(4 g c_g sinh²kh)`，docstring 声称每单位床面耗散
`D = (1/4)ρu_b²√(νσ)`。两种独立推导都给出比它大 √2 的值：

1. **黏性耗散积分**：Stokes 边界层 `u(z,t) = u_b[cos σt − e^{−z/δ}cos(σt − z/δ)]`，
   `δ = √(2ν/σ)`，则 `⟨(∂u/∂z)²⟩ = (u_b²/δ²)e^{−2z/δ}`，`∫₀^∞ dz = u_b²/(2δ)`
   → `D = ρνu_b²/(2δ) = (1/2)ρu_b²√(νσ/2) = (1/4)ρu_b²√(2νσ)`。
2. **壁面做功**：`τ_w = ρ√(νσ/2)u_b cos(σt+π/4)`，`D = ⟨τ_w u_b cos σt⟩ = (1/2)(1/√2)ρu_b²√(νσ)`
   —— 同一数值 `0.3536 ρu_b²√(νσ)`。
3. **数值复核（不采信上面两式）**：对 Stokes 层速度场做有限差分 + 梯形积分（6001×400 网格），
   在 `(T,h) = (1.0,0.25), (0.8,0.15), (2.0,1.0)` 三个工况得
   `μ_num = 1.652464e-3, 4.099163e-3, 1.460586e-4 1/m`，与修复前闭式之比**恒为 1.414210**，
   与修复后闭式 `σ²√(2νσ)/(4g c_g sinh²kh)` 之比 0.999997。

影响（用原记录重算，非手抄）：`μ_theory 1.1685e-3 → 1.6525e-3 1/m`；
`μ_meas/μ_theory 52.9 → 37.4`；分析窗口跨度 0.489 m 上的理论损失 `0.057 % → 0.081 %`，
超出部分 `2.921 % → 2.897 %`。**结论方向不变**（数值耗散仍主导 37 倍），但文献锚此前是错的。

> 出处说明：`scripts/audit/m15815_step2_references.py` 历史上自称“未调用被审函数”的独立复核，
> 实际只是把同一个闭式重新敲了一遍（`mu = σ²√(νσ)/(4g c_g sinh²kh)`），因此 5e-16 的
> 一致性**不构成独立验证**——这是共享同源抄写错误的典型；本次改用物理数值积分才暴露该因子。

## F4（中等级，未闭）判据口径可疑：条目写“文献波高衰减”，实现判的是造波机传递函数

条目 `ref: 文献波高衰减` 的字面判据应是把**实测衰减率** `μ_meas` 与文献值比较；
实现把 <10 % 判在“逐探头波高 vs 活塞传递函数目标”上，而文献衰减率的比较结果是
`μ_meas = 6.18e-2` vs `μ_theory = 1.65e-3 1/m`（**差 37 倍**，远超 10 %）。
本次没有改判据（改判据属于验收口径变更），只要求：报告必须**同时**给出两个数并写明
“波高门通过 ≠ 文献衰减率吻合”，已在新生成的 `wave_flume_case.md` 的误差判定节保留该并列。

**建议**：条目 acceptance 拆成两条显式子判据 —— (a) 逐探头波高 vs 造波目标 `≤10 %`（现状通过）；
(b) 衰减率 vs 层流边界层文献值：要求以分辨率阶梯证明 `μ_meas → μ_theory`（当前 `H/dx=3.2`
不足以支撑“数值耗散主导”的定量归因，见 F6）。

## F5（已做，通过）衰减不是启动斜坡/瞬态伪迹

以同一记录改变分析窗起点（零算力复核）：
`start = 2.2 / 2.5 / 3.0 / 3.2 / 3.5 s` → `μ = 6.087e-2 / 6.183e-2 / 6.127e-2 / 6.252e-2 / 6.028e-2 1/m`（散度 ±2 %），
逐探头波高几乎不变（`H₁ = 0.0400 m`）。`ramp_periods = 2.0`（斜坡至 t≈3.0 s 满幅）下
衰减依旧稳定，故沿程衰减不是斜坡包络的假象。

## F6（未闭，阻塞“数值耗散主导”的定量结论）单记录、单分辨率、无重复性

只有一条记录、一个分辨率（`H/dx = 3.2`，波峰 3.2 个粒子间距）。`μ_meas = 37×μ_theory`
被解释为“数值耗散”，但**没有分辨率收敛证据**支持，也没有第二条同参数记录证明可重现。

已递交（本单，集群）：`cj1628` = 同参数独立重跑（复现性，`runs/wf_D20_repro.json`）、
`cj1629` = `depth-cells 30`（`H/dx = 4.8`）阶梯。两者递交时队列 364 位，本单结束时仍 queued，
**下一单须 fetch 这两个作业并补 (a) 3.0 % 是否可重现 (b) `μ_meas` 是否随分辨率下降**。

## F7（轻微，记录）陈旧工件

* `docs/figures/wave_flume/series_wf_D16.png` 对应的 `runs/wf_D16.json` 已不存在（历史失败工况图）。
* `docs/wave_flume_verification_plan.md`（m17051）仍写“引用的 `runs/wf_D16.json` 缺失”，
  已被 `wf_D20` 取代，文档未回写。
* `scripts/audit/m15815_step2_references.py:101` 硬编码了修复前的 `μ` 闭式，现在对
  `src/` 的交叉核对会得到 `rel_diff = −0.2929`（即 1−1/√2）；该脚本是历史审计记录，未改，
  但**不得**再被当作独立基准引用。

---

## 复现命令

```bash
# 1) 从归档记录复算已发布数字（不需要 solver）
python -m pytest tests/test_wave_flume_case_artifacts.py -q          # 5→6 项
# 2) 文献锚的独立推导 vs 数值边界层
python -m pytest tests/test_wave_flume_damping_literature.py -q      # 7 项
# 3) 重新生成报告/数字/图/复现块
PYTHONPATH=src python benchmarks/wave_flume/analyse.py \
  --record docs/reports/records/wf_D20.json --start 2.5 \
  --out docs/reports/wave_flume_report.json --figures docs/figures/wave_flume
PYTHONPATH=src python benchmarks/wave_flume/make_report.py \
  --report docs/reports/wave_flume_report.json \
  --numbers docs/reports/wave_flume_numbers.json \
  --out docs/reports/wave_flume_case.md --repro "$(cat /tmp/repro_block.txt)"
```

**本单实测**：`tests/test_wave_flume_case_artifacts.py` + `tests/test_wave_flume.py` 47 passed；
`tests/test_wave_flume_grade.py` + `tests/test_wave_flume_damping_literature.py` 20 passed。
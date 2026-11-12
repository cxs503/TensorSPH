# TensorSPH · CASE-SPH-FSI-ELASTIC-GATE 第2轮（m16352）：调研结合 + 两处真实缺陷修复

## ① 问题设定与基准出处
弹性闸门溃坝 FSI（Antoci, Gallati & Sibilla 2007, *Comput. Struct.* 85:879-890,
DOI 10.1016/j.compstruc.2007.01.002）：水柱 0.146×0.292 m 被 0.36×0.004 m、E=1e6 Pa 的
弹性闸门挡住，t=0 释放，判据是闸门顶点水平位移时程的峰值与相位（容差 <10%）。
**基准可获得性（本轮程序化复核）**：Unpaywall `best_oa_location=null`；Crossref 有记录无全文；
OpenAlex 引用它的 OA 副本无一给出实验数值表；arXiv/Zenodo 无相关数据；5 条 OA PDF 直取通道
（mediatum、orbi.uliege、MDPI、ScienceDirect-AM、raw.githubusercontent）全部 403/拒连/超时。
⇒ **ref 数值本轮仍 WITHHELD，峰值/相位误差不可计算**（不是本轮新发现，而是被再次证实）。

## ② 方法与实现（改了什么）
1. **流→固载荷改为壁面压力面元法**：每个闸门粒子用一阶加权最小二乘拟合邻域压力平面，取壁面压力
   `p_face`，载荷 `p_face·dx`（`gate_load`，`load_transfer="panel"`）。旧 Adami 成对和在壁面半空间
   支撑下有 O(h) 亏缺，交付版用 `load_scale=1.2558` 事后除——本轮**移除该标定**。
2. **固端条件修复**：交付版弯曲能只罚相邻段夹角 ⇒ 闸门在底板处是**铰支**，可零弯曲能整体转动。
   本轮把首段与固定切向 (0,1) 的夹角并入弯曲能（`rod_energy`，`clamped_base=True`）。
3. **静力校核改为能量最小化**（L-BFGS-B，无量纲化），取代交付版"固定 0.995 阻尼时间推进"
   （审计证明其从未收敛：改阻尼即变号）。
4. **新增大变形参照**：Kirchhoff elastica（Bisshopp–Drucker）以弧长 ODE + 打靶自行求解，不抄任何文献数字。
5. 附带修复 `m_f` 误用 `wall_mass` 下标的潜在缺陷。测试：`tests/test_fsi_elastic_gate.py` 7 项断言全绿。

## ③ 参数表（无量化）
dx=12.5 mm, h=1.3dx, CFL 0.3, dt=2.44e-4 s；rho_f=1000 kg/m³, c0=20 m/s(弱可压), gamma=7,
nu=1e-6 m²/s；槽 0.584×0.45 m；水 0.146×0.292 m；闸门 0.36×0.004 m, E=1e6 Pa, nu_s=0, rho_s=2500
⇒ EI=5.333e-3 N·m, EA=4000 N/m, rho_A=10 kg/m²；Euler 屈曲载荷 π²EI/(4H²)=0.1015 N/m，自重 35.3 N/m（348×）。

## ④ 对照表（ref vs got）
| 项 | ref | got | 误差 | 判定 |
|---|---|---|---|---|
| 闸门顶点峰值位移 | **WITHHELD**（Antoci2007 闭源，5 条通道均不可取） | 1427.4 mm @ 297.6 ms（panel+固支, dx=12.5 mm） | **不可计算** | ✗ 不可判 |
| 峰值相位 | WITHHELD | 297.6 ms | 不可计算 | ✗ 不可判 |
| 交付版（对照） | WITHHELD | 354.3 mm @ 253.7 ms（成对和×1.2558 + 铰支） | — | — |
| V1 静水载荷 dx=12.5/10/7.5 mm | 418.22 N/m | 419.2 / 418.6 / 419.0 | **+0.24 / +0.09 / +0.19 %** | ✅ <3% |
| V1 交付版基线 | 418.22 N/m | 299.6 / 333.0 / 383.4 | −28.4 / −20.4 / −8.3 % | 已修 |
| A0 悬臂点载 n=25 | EB P H³/3EI | +6.34 % | ✅ <10% | |
| A1 分布载 n=25 / n=50 | EB Green | +12.54 / +6.08 % | 收敛 ✅ | |
| A2 elastica 大变形 (PL²/EI=10) | 自行求解 elastica | **+86.8 %** | ❌ 未达标 | |
| 端到端 V4 | EB | **not_run**：闸门太软，梁理论成立所需刚度下显式杆在声学步长失稳（Ω_ax·dt=12.6 被 validate 拦下） | | |

**是否 <3%？** V1 静水载荷传递 <3%（+0.24%）。杆的 EB 校核 A0 6.34%（n=25）/A1 6.08%（n=50）**未 <3%**；
**判据要求的"峰值/相位 vs Antoci2007 实验 <10%"无法计算**（ref WITHHELD）。

## ⑤ 收敛性/敏感性
- V1：三个分辨率均 ≲0.25%，无分辨率依赖（修复是结构性的，不是标定）。
- A1：n=25→50 误差 12.54%→6.08%，单调收敛。
- 耦合：dx=12.5 mm 单分辨率完成（1642 步, 247 s）；dx=10 mm 运行超时被终止 ⇒ **分辨率收敛未完成**。

## ⑥ 失败模式与工件
`fail_stage = external_reference_withheld + large_deflection_validity`：
(a) Antoci2007 实验数值在本沙箱不可获得（provenance.json 记录全部查询/取件失败）；
(b) 固支+面元后耦合算例在大变形区跑飞（峰值 1427 mm > 3.96×闸门长度、t=100 ms 才 5.0 mm）——
与 A2 暴露的离散杆大变形误差 +87% 一致：**顶点位移指标在杆的大变形精度修好前不可信**，
且当前无自接触/水再附着处理。
repro_cmd: `python -m pytest tests/test_fsi_elastic_gate.py -q -k "not coupled_short"`（7 项，全绿）
工件：`artifacts/fsi_elastic_gate/{runs.json,summary.json,rod_verification.json,provenance.json,v4_gate_load_path.json,fig1_tip_and_load.png}`，
文档：`docs/fsi_elastic_gate/{01_survey_oss.md,02_method_selection.md}`。

## 结论（honest）
+ 两处**真实缺陷被定位并修复**：流固载荷 −28.4% 的系统性亏缺（改为面元法，−28.4% → +0.24%）、
  闸门底板**铰支→固支**；标定量 `load_scale` 从生产路径移除；验证阶梯重建并有 pytest 护栏。
− 判据（峰值/相位 vs Antoci2007 <10%）**未达成**：参照值不可得（WITHHELD），且修好前两处缺陷后
  暴露的第三个缺口——离散杆大变形精度（A2 +87%）——让顶点位移指标当前不可信。

## 更正（m16454，2026-11-12）

上表 A2 的 **+86.8 %** 与本文第 ⑥ 节「离散杆大变形精度（A2 +87 %）让顶点位移指标不可信」
**是坐标系错配，不是杆的误差**：A2 把杆的*横向*挠度与 elastica 的*轴向*坐标相比。
同坐标系复算（25 节点）：横向 +2.55 %、轴向缩短 +3.67 %、端部转角 +0.59 %，且随节点加密
单调收敛（13→25 节点：+4.91 % → +2.55 %）。因此杆的大变形精度**不是**闸门顶点位移指标的
阻塞项；剩下的两个阻塞项（显式杆 `Ω_ax·dt` 声学步长限制、Antoci2007 参照值 WITHHELD）不变。
证据：`docs/fsi_elastic_gate/04_a2_frame_fix.md`、`runs/fsi_elastic_gate_a2/a2_metrics.json`、
`scripts/m16454_a2_frame_fix.py`；守门测试已从「已知缺口（0.2<err<1.2）」改为
「横向 <5 % / 轴向 <6 % / 转角 <2 % + 收敛单调」三条真门。

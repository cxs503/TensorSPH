# 交差核验（board audit）：m16352 · CASE-SPH-FSI-ELASTIC-GATE · TensorSPH

**审计任务** m16406 · **被审任务** m16352（status=succeeded，PR 未开，head=`4af6b11`，分支 `agent/m16352`）
**被告知基准判据** `tensor_benchmark_cases.anchor_slug='CASE-SPH-FSI-ELASTIC-GATE'` · **审计日期** 2026-11-12
**结论** 成熟度 **L2**（声明级 2，gate 机器复核同级）· 判定 **部分证实 / L4 未成立** · 处置 **approve with notes + 开 L4 返修（另单）**

> 本单是交差核验，不是重做：没有改动 `src/`、`tests/`、`benchmarks/` 一个字节（被审交付文件 sha256 见
> `evidence_m16352.json.impl_sha256`）。全部数字由 `scripts/audit/m16406_case_fsi_board_audit.py` 程序化重算，
> 产物落 `docs/board_audit/m16406/`。

---

## ① 取证：三件套逐项核验

| 三件套 | 结论 | 证据 |
|---|---|---|
| **run_artifacts** | **真存在、非空、可复算** | `artifacts/fsi_elastic_gate/` 6 件（`runs.json` 45.9 KB 含 330 采样点 6 条时程；`summary.json`；`rod_verification.json`；`v4_gate_load_path.json`；`provenance.json`；`fig1_tip_and_load.png` 110 KB）。审计侧镜像 + 逐件 sha256 见 `docs/board_audit/m16406/artifacts/` |
| **benchmark ref vs got** | **基准表 0 行 → 无判据可裁** | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-SPH-FSI-ELASTIC-GATE'` → **0 行**（全表 32 行；`anchor_slug ILIKE '%ELASTIC%'` 只命中 TensorDEM 的 Campbell 算例 4 行）。`tensorbench_results WHERE mission_id=16352` → **0 行**。机器裁判 `GET /tensor/bench/l4-verdict/16352` → `status=CLAIMED_NOT_PROVEN, l4_benchmark_pass=false, rows=0, failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]`。ref 侧被审方声明 **WITHHELD**（Antoci 2007 无 OA 全文，5 条取件通道 403/拒连/超时，`provenance.json` 留痕）——**审计侧独立复核属实**：`provenance.json` 的 4 条外链尝试与 OpenAlex/Crossref/Unpaywall 记录自洽，仓内亦无任何 Antoci 数值锚点 |
| **repro_cmd** | **可执行、可复现（已跑）** | `python -m pytest tests/test_fsi_elastic_gate.py -q -k "not coupled_short"` → **7 passed, 1 deselected, 95.46 s**（审计侧重跑，与交付声明逐字一致）。入口 `benchmarks/fsi_elastic_gate/run_antoci.py` 存在且是本单唯一调用方 |

### 报告六段（`tensor_reports` id=15473，3309 字符）

①问题设定/基准出处 ②方法与实现 ③参数表 ④ref vs got 表 ⑤收敛性/敏感性 ⑥失败模式与工件 —— **六段齐全**，
失败模式显式登记（`fail_stage = external_reference_withheld + large_deflection_validity`），honest-FAIL 未粉饰。
缺口：④ 表 A0/A1/A2 三行的列错位（ref 列写成"EB P H³/3EI"公式而非数值，got 列写误差），且 **未登记
`rod_verification.json` 中已有的 A0@n=50 = 0.000 m（误差 −100%）**——见发现 F3。

---

## ② 判级（L0–L5）

| 级别 | 判据 | 判定 | 依据 |
|---|---|---|---|
| L0 文档 | 文档成果 | ✅ | `docs/fsi_elastic_gate/{01_survey_oss,02_method_selection,03_report}.md` |
| L1 数学/单元验证 | 测试断言不变量 | ✅ | `tests/test_fsi_elastic_gate.py` 7 项：静水载荷面元亏缺 <1 %、成对和 legacy 亏缺护栏、EB 悬臂点载/分布载、固支 vs 铰支、elastica 自洽——**但含 1 条恒真断言（F2）** |
| L2 独立功能实现 | 真代码 输入→计算→输出 | ✅ | `src/tensorsph/fsi_elastic_gate.py`（789 行，非桩）：面板载荷 `gate_load`、固支弯曲能 `rod_energy`、能量最小化静力、elastica 参照、耦合驱动 `run`；`artifacts/…/runs.json` 是该管线真跑出的 1642 步时程 |
| L3 系统集成 | 主求解器/平台侧调用方 | ❌ | `grep -rn fsi_elastic_gate`（排除自身文件）→ 仅 `benchmarks/fsi_elastic_gate/run_antoci.py:30` 一处 **自备驱动**；`src/tensorsph/__init__.py` 无导出；无 CLI、无平台 tools 引用、无其他求解器模块调用 |
| L4 基准验证 | 完整算例+误差分析+可复现 | ❌ | 基准表 0 行、`tensorbench_results` 0 行、ref WITHHELD ⇒ 误差不可计算；耦合算例自身**数值失效**（发现 F4）；分辨率收敛未完成（dx=10 mm 超时）；V4 端到端 `not_run` |
| L5 科研工程成果 | — | ❌ | 无 |

**级别 = L2。** 这是本次审计最需要向 owner 说清的一条：本交付**功能实质扎实但集成面为零**。
它与 9 小时前的先例 **verdict id=464（m16318，同一仓、同一"独立 case 模块+自备驱动+测试"形态）判了 L3** 相冲突——
那里的 `integration_callers` 值同样只是自备驱动 + 自己的测试。本单不沿用：把「自己 import 自己」记为集成
＝机器门 `callers_verified()` 可被单串 grep 骗过，正是任务书警告的"走过场"。**两种口径请 owner/board 统一**
（若采纳先例口径，本单应升 L3，其余证据无需改动；L4 仍不成立）。

---

## ③ 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorSPH','m16352 CASE-SPH-FSI-ELASTIC-GATE',16406,2,'<8 键 JSON>','board-audit');
-- 实际执行：audit/insert_verdict_m16406.py → id=476, level=2, gate_state=pending（(repo,scope,verdict_by) 幂等 upsert）
```

evidence 八键真实指针（**键集合与 m16318 先例一致**，见 `audit/m16406_verdict_evidence.json`）：
`impl_files` / `entry` / `run_artifacts` / `tests` / `integration_callers` / `benchmark` / `repro_cmd` / `external_anchor`。

---

## ④ 发现清单（按严重度）

### 阻断级（L4 不成立）
- **F1 基准不可裁**：锚点未登记进 `tensor_benchmark_cases`（0 行），无 ref/got/tolerance；`tensorbench_results` 0 行；
  机器裁判 `CLAIMED_NOT_PROVEN`。ref（Antoci 2007 峰值/相位）WITHHELD 经复核属实 ⇒ 本算例**至今没有任何可机器裁的精度判据**。
- **F2（新，审计方独立发现）交付报告的核心失败结论 A2「离散杆大变形 +86.8%」是坐标轴接反造成的假 FAIL。**
  `verify_cantilever()`（`src/tensorsph/fsi_elastic_gate.py:746-753`）把杆的 **x=横向位移**（闸门竖直、载荷沿 x）
  直接与 elastica 的 **along-beam x** 比，又只把 rod 的 `tip_y` 与 elastica `tip_y`（横向）并列存放而不用它比。
  正确对照（审计侧重算，`independent_checks.json.A2`）：

  | 量 | elastica（审计侧独立求积） | rod n=25 | rod n=50 |
  |---|---|---|---|
  | 横向位移 | 0.291819 m | 0.299255 m（**+2.55 %**） | 0.295540 m（**+1.28 %**） |
  | 沿梁坐标 | 0.160202 m | 0.152860 m（**−4.58 %**） | 0.156595 m（**−2.25 %**） |
  | 顶端转角 | 1.43029 rad | 1.43877 rad（**+0.59 %**） | 1.43465 rad（**+0.31 %**） |
  | 交付代码算出的"误差" | — | **+86.80 %** | +84.48 % |

  即 A2 实际 **<10 % 且在收敛**，交付方的 +86.8 % 是口径 bug 造成的**假失败**；由此推出的
  "离散杆大变形精度是第三个缺口、顶点位移指标不可信"（报告 ⑥(b)）**被证伪**。
  elastica 参照本身经审计侧独立求积复核：x/L=0.445004、y/L=0.810609、φ_e=1.43029 rad，
  与交付 `elastica_tip()` 相对差 **2.6e-13 / 1.0e-13**（即参照正确，问题在对照口径）。
- **F3（新）A0@n=50 静力解为 0：** `rod_verification.json["50"]["A0_point_load"].tip_rod = 0.0`，rel_error **−100 %**
  ⇒ L-BFGS-B 在该档返回初值（收敛判据/无量纲化尺度失效），而报告 ④ 表只登记 A0@n=25（+6.34 %），
  **未把 n=50 的 −100 % 写进报告**。同一 `verify_cantilever()` 在 n=50 的 A1/A2 档是好的 ⇒ 是优化器/度量缺陷，
  但"验证阶梯 n 加密即收敛"的结论目前只在 A1/A2 两档成立。
- **F4（新）耦合算例是数值失效，不是"大变形跑飞"：** 重放 `runs.json` 时程（`independent_checks.json.run_artifacts`）
  - `|tip| > 闸门长 0.36 m`（几何上不可能）自 **t=0.1148 s** 起，**177/330 个采样点越界**；
  - 峰值 **1427.4 mm = 3.96×闸门长**，谷值 **−1370.6 mm**；100 ms 之后 **6 次变号**，周期 ≈32 ms（≈31 Hz，
    而闸门一阶弯曲频率 ≈0.1 Hz ⇒ 高 300 倍的数值振荡，非物理运动）；
  - 流体最大速度 **71.94 m/s = 3.60 c₀**（c₀=20 m/s）：弱可压 WCSPH 已越出马赫数适用域；
  - `gate_force_at_t0 = 0.000 N/m`（静水应 ≈418.2 N/m），力峰值 **26794 N/m = 64× 静水值**，
    t=0.0697 s 已出现 >5× 静水值，t≈0.089–0.092 s 力达 512→592 N/m 后闸门才开始动（t=0.0953 s 才 4.29 mm）。

  ⇒ 事实链是「初始状态无静水压强（WCSPH 密度=ρ₀ ⇒ p=0）→ 压力波冲击 → 壁面外推压力尖峰（无界）→ 杆被锤击」，
  与"离散杆大变形精度"无关（F2 已证杆在大变形下 1–5 %）。**根因未证**（时程未存 `p_face` 诊断量），
  但"该时程不能当物理结果用"是确定的。
- **F5 端到端 V4 `not_run`**：`v4_gate_load_path.json` 的 `rel_error = −97.86 %`（tip_rod 0.359 m vs EB 16.80 m）
  说明**整条流体→固体检定链**在可被梁理论校核的刚度档上失稳（被审方已如实标注 `Ω_ax·dt=12.6` 被 validate 拦下）。

### 重要级
- **F6 恒真断言（测试造假面）**：`tests/test_fsi_elastic_gate.py:96`
  `assert abs(nodes_pin[-1, 0] - pinned.water_width) > abs(tip_dx) or True` —— `or True` 使该断言**永远通过**。
  连带 m16352 result 里"succeeded / 7 passed"的分量被削弱：**6/7 是真断言，1/7 恒真**。
- **F7 耦合测试的时间窗覆盖不到失效点**：`test_coupled_short_run_is_deterministic` 用 `t_end=0.01 s`（41 步），
  而失效自 **t=0.1148 s** 才开始 ⇒ 该护栏对真实失效**结构性失明**（"跑了没崩"≠"算对了"）。
- **F8 报告 ④ 表列错位 + 选择性缺行**：A0/A1/A2 三行 ref 列填的是公式、got 列填的是误差；A0@n=50（−100 %）缺席。
- **F9 分辨率收敛未完成**：dx=10 mm 超时被终止；报告如实记录，但 L4 的"误差分析"因此只有单分辨率。

### 次要级
- **F10** `src/tensorsph/fsi_elastic_gate.py` 未在 `src/tensorsph/__init__.py` 导出（与仓内既有 case 模块惯例一致，仅记录）。
- **F11** `summary.json` 的 `note` 字段仍写 WITHHELD 结论（与报告一致，无矛盾），但 `runs.json` 只有 1 个 run key
  （`new_panel_clamped_12p5`），缺 dx=10 mm 档记录（超时未落盘）——单档证据链。
- **F12** `provenance.json` 的 `pdf_fetch_attempts` 第 4 条 `raw.githubusercontent.com` URL 被截断成 `...`，
  不可复核（自证文件应存完整 URL）。

---

## ⑤ 主张 vs 实测（adversarial checks）

| 被审主张 | 审计实测 | 结论 |
|---|---|---|
| "7 passed" | 审计侧重跑 → 7 passed in 95.46 s | ✅ 属实（但含 F6 恒真断言） |
| "V1 panel +0.24/+0.09/+0.19 %" | 审计侧现算 `verify_hydrostatic_load`（dx=12.5/10/7.5 mm）→ +0.240/+0.089/+0.189 % | ✅ 逐位重合 |
| "legacy 成对和 −28.4/−20.4/−8.3 %" | 现算 → −28.369/−20.367/−8.317 % | ✅ 属实 |
| "load_scale=1.2558 已从生产路径移除" | `load_scale` 默认 1.0，`.step()` 仍乘 `case.load_scale`（保留 1.0 无影响） | ✅ 实质上已移除（保留无害） |
| "固端修复（铰支→固支）" | `rod_energy` 首段与 (0,1) 夹角入弯曲能（`:262-267`），`clamped_base` 默认 True | ✅ 真实实现 |
| "A2 大变形 +86.8 %，杆大变形精度是缺口" | **证伪**：轴对调口径错误，正确值 +2.55 %/−4.58 %（n=25）、+1.28 %/−2.25 %（n=50） | ❌ **假 FAIL**（F2） |
| "elastica 参照自行求解，不抄文献" | 审计侧用独立椭圆型求积重算：x/L=0.445004, y/L=0.810609, φ_e=1.43029；与交付相对差 ≤2.6e-13 | ✅ 参照正确 |
| "耦合峰值 1427 mm 是杆大变形精度所致" | 时程 177/330 点越界、31 Hz 振荡、Ma=3.6、力 64× 静水、t=0 载荷为 0 | ❌ 归因错误（F4） |
| "ref WITHHELD" | 仓内无 Antoci 数值锚点；`provenance.json` 渠道记录自洽 | ✅ 属实（honest） |
| "V4 未跑" | `v4_gate_load_path.json rel_error=−97.86 %`，交付未宣称 PASS | ✅ 如实 |
| 基准表有该锚点 | `tensor_benchmark_cases` 0 行 | ❌ 无判据登记（F1） |

---

## ⑥ 未执行的检查（honest not_run）

- 未重跑耦合算例（247 s/档 ×2，且 F4 已定性为数值失效；无新信息量）。
- 未跑 dx=10 mm（被审方已超时终止，重跑需集群；L4 的收敛面仍空）。
- 未运行全仓回归套件（只跑 `tests/test_fsi_elastic_gate.py`）。
- 未取得 Antoci 2007 实验曲线（与被审方同样受限于闭源；无 OA 通道）。
- 未定位 F4 的根因（需存 `p_face`/壁面密度时程的诊断跑，属返修轮工作）。

---

## ⑦ 返修清单（给接棒单，按收益排序）

1. **修 A2 对照口径**（3 行）：`verify_cantilever` A2 段改为横向/沿梁分列比较，并删掉恒真断言的 `or True`
   （补丁草案 `docs/board_audit/m16406/suggested_patch.diff`，未应用）。修完 A2 应从"❌ 未达标"变为"✅ 2–5 %、随 n 收敛"。
2. **A0@n=50 归零查因**：给 `rod_static_equilibrium` 加"回报收敛状态 + 残差"字段，禁止静默返回初值；测试加 `n=50` 档。
3. **把锚点登记进 `tensor_benchmark_cases`**（metric=顶点峰值位移/tolerance 10 %），否则 L4 永远不可裁；
   同时把 `tensorbench_results` 走一次机器重算。
4. **耦合路径物理可用性**：① 初始状态给静水密度/压强（或用势流/静水场初始化）；② 面板外推压力加限幅
   （≤ 邻域最大压强，或回退 Adami 项）；③ 存 `p_face`、`max_speed` 诊断量并设硬阈值（Ma>0.1 即 FAIL 而非继续跑）；
   ④ 自接触/水再附着（闸门会折叠到地板）。
5. **把"耦合护栏"的时间窗放到失效之后**（至少 t=0.2 s 并对 |tip|≤H 做断言），否则护栏形同虚设（F7）。

---

## ⑧ 门禁机器复核预演（dry-run）+ 平台报告

本次报告已发 `tensor_reports` id=**15483**（mission_id=16406，六段齐全，3 张图内嵌）。


用 `/root/tp-agent/maturity_gate.py: judge(level, evidence, repo, audit_mid)` 只读预演（不触发动作）：

```text
judge(1, {}, ...)                         -> (1, ...)  # tests 非空即可
judge(2, evidence_8keys, 'TensorSPH', 16406) -> (2, ['L3 缺: integration_callers(grep 无引用)'])
```

**说明**：L2 三键（impl_files/entry/run_artifacts）全过；L3 因 `integration_callers` 只指向自备驱动而失守
（本审计主动不申报，见 §② 的口径说明）。故 `gate_level=2` 与声明级一致 → **confirmed（消账）**；实际复核由守护每 30 min 一轮自动执行（本单落表时最近一轮 13:58，审计窗口内 gate_state 仍为 pending，预演结果已在上方），
而非 `reopened`（reopen 只在 gate_level≤1 时触发）。**L4 缺口不因消账而消失**：建议另开返修单承接 §⑦。

---
审计可复现命令：

```bash
python3 scripts/audit/m16406_case_fsi_board_audit.py   # 取证+独立重算+图 → docs/board_audit/m16406/
python3 audit/insert_verdict_m16406.py                # 落表（幂等）
python3 -m pytest tests/test_fsi_elastic_gate.py -q -k "not coupled_short"   # 7 passed
```
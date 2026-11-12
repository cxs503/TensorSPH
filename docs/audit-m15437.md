# m15437 CASE-SLOSHING 交差审计（board-audit）

对象：m15437（TensorSPH · 算例池 CASE-SLOSHING · 晃荡共振一阶频率），pr_gate PR#501 merged（commit `bd5fd8c`）
审计单：m15717 · 审计日期 2026-11-12 · 判级 L4 · 结论=**confirmed（带缺陷单，需后续修）**

独立复算脚本：`artifacts/audit-m15437/audit_m15437.py` → `artifacts/audit-m15437/audit_metrics.json`（本文件所有数字均由该脚本注入，零手抄）。
工件副本（含 2 个交付未列表的算例）：`artifacts/audit-m15437/runs/*.json` + `SHA256SUMS.txt`；复跑件 `artifacts/audit-m15437/repro_base_cpd12.json`。

---

## ① 仲裁基线与三件套取证

| 项 | 取证结果 |
|---|---|
| 基准裁判（`tensor_benchmark_cases`） | **无 CASE-SLOSHING 行**（`SELECT … WHERE anchor_slug='CASE-SLOSHING'` 返回 0 行；全表仅 20 个 anchor，无 slosh 类）。→ 平台基准表对本锚点**无登记判据**，机器裁判空缺 |
| 锚点登记（`feature_ledger`/`tensor_case_pool`） | `feature_ledger` #152 `CASE-SLOSHING` acceptance=「ref: 线性理论频率｜容差: <10%」status=open；`tensor_case_pool` #76 case_code=SLOSHING ref=线性理论频率 tol=<10% mission=15437 status=dispatched。→ 判据存在，且与交付内 `cases.py:TOLERANCE=0.10` 一致 |
| run_artifacts | 8 例 JSON 全在（`runs/sloshing_resonance/*.json`，每例含 times/modal/com_x/场快照/诊断），6 例镜像进 `docs/benchmarks/sloshing-resonance/cases/` 且 sha256 与 runs 逐位相同（6/6 SAME）；5 张 PNG 非空（灰度 std 38–49，256 灰阶） |
| benchmark ref vs got | ref（闭式）复算 = 交付值：base 6.66187 rad/s；got = 6.72549 rad/s；误差 0.955%。第二路 FD 特征值自写复算与闭式相对偏差 2.0e-11 |
| repro_cmd 可执行 | **是**。按交付 ⑥ 命令复跑 `run.py --case base_cpd12`：360 s 出件，ω_com=6.72549419（与交付 6.72549419 完全相同），181 个时间点/336 粒子/52355 步**逐位一致**（max|Δcom_x|=0，max|Δmodal|=0） |
| 报告六段 | `docs/benchmarks/sloshing-resonance.md` ①–⑥ 齐全（基准出处/方法与代码指针/无量纲参数表/ref-vs-got 对照表/收敛与敏感性/失败模式+repro_cmd+工件）；数字与 criteria.json 程序注入一致 |
| 平台报告（`tensor_reports`） | **缺失**：`mission_id=15437` 无行；mission result 自述 `POST /tensor/reports 返回 500`，报告只以 docs/ 形式存在（report_id=null） |
| 代码实质 | 真 WCSPH：`sloshing_mode.run_sloshing_mode` 走仓内主求解器 `droplet_impact.rhs`（Tait EOS + 配对表 `PairListCache` + Adami 壁面粒子 + 变步长 CFL），非解析造数、非随机 |

## ② 复算结果（独立估计器 vs 交付值）

`w_com` = 交付主测量（质心全记录均值上穿）；`w_1st2` = 审计侧仅取前两次上穿的估计；`w_th` = 闭式理论（审计重算）。

| case | H/L | cpd | w_th | w_com | err% | w_1st2 | err1st2% | 交付表内 | 交付 verdict | run.py `pass` |
|---|---|---|---|---|---|---|---|---|---|---|
| base_cpd12 | 0.42 | 12 | 6.6619 | 6.7255 | +0.96 | 6.8507 | 2.84 | 是 | PASS | True |
| amp_016 | 0.42 | 12 | 6.6619 | 6.6859 | +0.36 | 6.6565 | 0.08 | 是 | PASS | True |
| cs_040 | 0.42 | 12 | 6.6619 | 6.6125 | +0.74 | 6.6073 | 0.82 | 是 | PASS | True |
| amp_004 | 0.42 | 12 | 6.6619 | 6.5716 | +1.35 | 7.7464 | 16.28 | 是 | PASS | True |
| res_cpd16 | 0.42 | 16 | 6.6619 | 6.8260 | +2.46 | 6.8271 | 2.48 | **否** | — | True |
| geom_h020 | 0.20 | 12 | 5.3483 | 5.6492 | +5.62 | 5.6193 | 5.07 | **否** | — | True |
| geom_h060 | 0.60 | 12 | 7.0036 | 10.6930 | +52.68 | 7.3078 | 4.34 | 是 | **PASS** | **False** |
| res_cpd08 | 0.42 | 8 | 6.6619 | 10.2991 | +54.60 | 6.9540 | 4.39 | 是 | FAIL | False |

上穿时刻诊断（为什么全记录估计会跳到 10.3–10.7）：`geom_h060` 上穿间隔/T_th = 0.96, 0.77, **0.24**；`res_cpd08` = 0.96, **0.17**, 0.81 —— 强数值阻尼下尾部幅值降到 ~3e-4（≈量测台阶），最后几次上穿是噪声穿越。干净多周期记录只有 base（0.97/1.01）、cs_040（1.01/1.01）、res_cpd16（0.98/0.98）、amp_016（1.00/0.99）。

按锚点判据（每例 <10%）裁定：
* 交付的**全记录**估计器（run.py 声明的 primary）：8 例 6 通过，`geom_h060`、`res_cpd08` 出界 → **FAIL**；
* 交付**报告表**的窗口化估计器（aggregate 后置选择）：6 例 5 通过 → 文档写 FAIL（5/6）；
* 两种选择下锚点判据都**未满足**；但基准工况与 3 个干净算例都在 **0.1–2.5%**，能力实质为真。

## ③ 缺陷清单（按严重度）

**B1（blocker？不——高，影响判据自洽）主测量定义前后不一致，geom_h060 判定被翻转。** `run.py` 明确声明 primary=质心全记录并据此写 `pass` 字段（geom_h060 pass=False，52.68%）；`aggregate.py` 另用「数据驱动窗 + 全记录均值上穿」作 primary 并在表里判 PASS（7.1165，1.61%）。同一算例 run.py=False / table=PASS，交付摘要「5/6 通过」在声明的 primary 下应为 4/6。文件内对换算子有披露（④ 表并列三列），但**判据结论随估计器而变**，属未预注册的量测算子后置选择。

**B2（高）表/摘要与已产出工件不同步。** `geom_h020`、`res_cpd16` 的 run JSON 已存在且通过（+5.62%、+2.46%），但 `table.md`/`summary.json`/`criteria.json` 未重跑 aggregate，报告 ⑤ 仍把它们写成「未完成/UNMEASURED（仍在运行）」。→ 交付自述与实际工件矛盾；另 `aggregate.py` 引用的 `res_cpd20`、`cs_160` 两档根本没有 run 工件（`cases.py` 有声明、未执行）。

**B3（中）收敛性证据不足且被反例削弱。** 报告称分辨率阶梯 8→20 单调；实有数据只有 8/12/16 三档，误差 54.6%（不可测）/0.96%/2.46% —— 非单调，`criteria.json Q2 monotone=True` 只因为只取了两点。收敛阶/单调性不能作为已证结论。

**B4（中）平台报告缺位。** `tensor_reports` 无 mission 15437 行（当时 POST 500）。六段报告只在仓内 docs/，未进知识面。

**B5（低）基准注册缺行。** 锚点判据只在 `feature_ledger`/`tensor_case_pool`，`tensor_benchmark_cases` 无 CASE-SLOSHING 行 → maturity/基准机器裁判无 ref 可读，只能吃报告自述。

**B6（低）外部锚点=闭式公式在代码内复算，未校验纸面页码/表格**（交付自述离线 403/超时）。ref 本身是教科书结果且两路互校（rel 2e-11），可信；但不构成 L5 意义上的外部数据锚。

## ④ 判级 L4 的理由（逐级）

* L2 独立功能：`sloshing_mode.py`(385 行)/`sloshing_anchor.py`(143 行)/`run.py`(224 行) 真实现（容器+配对+时间推进+量测），入口 `run.py` CLI，真出数。
* L3 系统集成：求解走仓内主求解器 `droplet_impact.rhs`（同一个 core rhs 被其他自由面/冲击算例复用）、壁面/配对/物性全用主库；算例在 `tensor_case_pool` #76 登记挂本 mission；仓内 pytest 接入（`tests/test_sloshing_resonance.py` 12 passed / 31.75 s，本次审计实跑）。
* L4 基准验证：ref=线性理论频率（闭式+独立 FD 双路，审计自写 FD 复算 rel 2e-11）+ 8 例误差表 + 敏感性两因子 + 复跑**逐位可复现**（本次实测）。合格。
* L5 不成立：无新物理结论、无外部实验/文献数据表校核，且锚点「每例 <10%」未满足。

## ⑤ 处置

**confirmed，L4 保留；返修项（不阻塞消账，但需挂后续单）：** B1 统一 primary 定义（或把窗口化算子写进 cases.py 预声明并重跑 aggregate 重算全部 verdict）；B2 重跑 aggregate + make_report 纳入 geom_h020/res_cpd16，删掉不存在的 cpd20/cs160 或补跑；B3 收敛性改述为「8 档不可测，12/16 两档误差 0.96%/2.46%，无单调收敛结论」；B4 补发平台报告；B5 往 `tensor_benchmark_cases` 增补 CASE-SLOSHING 判据行（base_cpd12, metric=|ω−ω_th|/ω_th, tolerance=0.10）。

**诚实边界（本审计）**：审计侧只用已存时间序列复算，未重跑 8 例全矩阵（只重跑 base 一例，逐位一致）；`w_1st2` 是启发式，amp_004 上给出 16%（尾部噪声），不作为结论量；平台 `tensor_benchmark_cases` 表以 postgres 角色读取（root 无该表权限）。
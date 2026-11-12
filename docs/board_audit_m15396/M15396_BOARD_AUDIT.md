# m15396 交差核验：CASE-LAPLACE-SPH（TensorSPH）

> 交差审计（非重做）。被审交付：`e02ee870cb6e` @ 分支 `agent/m15396`（未并入 main）
> 审计工作树 `m15421`；本页与 `artifacts/audit_m15396/*.json` 全部程序化生成。

**裁定：L2（board-audit）** —— L2 met: real module (CSF state + matrix-free CG Poisson + measurement), 11 tests pass, artefacts reproduce bit-for-bit

## 1. 案例与基准出处

- 物理：不可压投影 SPH 静止液滴的 Young-Laplace 压跳；二维圆界面（曲率 1/R 与 0），ref = dp = sigma / R  (2-D circular interface: kappa1=1/R, kappa2=0)。
- 参数：R = 0.020 m，σ = 0.072 N/m，ρ0 = 1000，气相载体厚 0.012 m，h/dx = 1.3。
- 基准出处：Young-Laplace law; Brackbill, Kothe & Zemach (1992) JCP 100:335; Cummins & Rudman (1999) JCP 152:584 (projection SPH)
- **注册裁判**：`tensor_benchmark_cases` 中 `anchor_slug='CASE-LAPLACE-SPH'` → **0 行**（全库 TensorSPH 域 0 行）。容差 5 % 的真实来源是任务目标原文：`【算例池|LAPLACE-SPH｜ref: ΔP=σ/R 解析｜容差: <5%（目标数值任务从 ref 原文查得）】`，并在 run.py 里以 `TOLERANCE = 0.05` 先于运行写死。裁判行缺失=台账缺口（见 §5 坑 1），不以之替代判据。

## 2. 怎么核的

1. `git show e02ee870cb6e:<path>` 逐字节取 9 个交付文件 → `artifacts/audit_m15396/branch_e02ee87/`，sha256 入 `sha256manifest.json`（缺件 0）。
2. 在临时副本里**整跑** `PYTHONPATH=src python benchmarks/laplace_sph/run.py --figure`，把新 results.json 与提交版逐叶比（652 叶）。
3. 独立跑 `pytest tests/test_laplace_sph_isph.py`。
4. 数据裁判：从提交的 results.json 重算 dp/σ·R、逐级误差、仪器自检、负对照、附流门；并把报告里的每个头条数字回查 results.json（零手抄核对）。
5. 集成面 grep：全仓找 `laplace_isph` 的调用方 / 包导出。

## 3. 结果（ref vs got）

| R/dx | N | Δp got [Pa] | Δp ref [Pa] | got/ref | 误差 | <5% 门 | κ 误差 | CG iters |
|---|---|---|---|---|---|---|---|---|
| 10.0 | 1203 | 3.6350 | 3.6000 | 1.0097 | +0.97 % | PASS | +1.10 % | 78 |
| 13.3 | 2150 | 3.6218 | 3.6000 | 1.0061 | +0.61 % | PASS | +1.31 % | 104 |
| 20.0 | 4773 | 3.5923 | 3.6000 | 0.9979 | -0.21 % | PASS | +0.27 % | 156 |

- 算术核对：σ/R = 3.600000000000 Pa，提交 `dp_theory_Pa` = 3.600000000000 Pa（一致）。
- 最差级误差 0.97 %，三级全在 5 % 内：True。
- 仪器地板：gauge 自检 0.29 %，通量交叉校核 0.60 %，壳外动量残差 0.58 %。
- 消融（最细级 dp/ref）：model 0.9979、κ_exact 0.9948、δ_unit 1.0047、gauge 1.0029 → 误差主要来自曲率算子，δ 归一化次之，泊松解本身 0.29 %。
- 负对照 σ=0：Δp = 0.0e+00 Pa（门 1e-10）→ PASS。
- **诚实 FAIL**：附流门 measured = nan，第 13 步发散，0.5 毛细时间 max|u|/v_ref = 0.889（Adami 限 0.1）。交付未粉饰，判据里明写 ❌。

## 4. 可复现性（实测）

- 整跑退出码 0，墙钟 12.51 s；652 叶中差异 1 条：/wall_clock_s。
- 图重建 sha256 = `602ea05d0ea6493f…`，与提交版一致。
- pytest：11 passed, 1 warning in 63.02s (0:01:03)。
- `docs/verification_laplace_sph_isph.md` 由 make_doc.py 从 results.json 渲染，测试 `test_documented_page_matches_rendering` 把「页面=渲染」钉死。

## 5. 注意点（坑与局限）

1. **注册裁判缺失**：`tensor_benchmark_cases` 无 CASE-LAPLACE-SPH 行（TensorSPH 域 0 行）。5 % 容差只能追到任务目标原文 + run.py 的预置常量，机器门无法对表裁定 —— 台账缺口，不影响本次实测数字，但 L4 的「机器可复核基准」一环仍空。
2. **静力学仪器 ≠ 多相能力**：气相与液相同密度，只是色函数补全；dp 由 `div(1/ρ grad p) = div(a_CSF)` 的泊松解给出，本质是「离散算子能否承受该跳跃」的自洽核验，曲率算子误差（+1.10%/级）是主要信息量。交付文档已自陈此限，未越界宣称。
3. **动力学半场 FAIL**：投影时间推进第 13 步发散，故该模块不能当求解器用，也未被主求解器调用（`no integration callers`；导出：False）。
4. 交付在 `agent/m15396` 分支，未并入 main；eco main 无 `benchmarks/laplace_sph`（确无）。

## 6. 误差判定：是否 <3 %

- 静力学关键量 Δp：**是** —— 最差 0.97 %（三级 ≤1 %），仪器自检 0.29 %，通量路线 0.60 %。
- 动力学附流：**否**（honest FAIL，已如实登记）。

## 7. 判级（L0–L5）

- L2 met: real module (CSF state + matrix-free CG Poisson + measurement), 11 tests pass, artefacts reproduce bit-for-bit
- L3 blocked: no caller outside benchmarks/laplace_sph/run.py, not exported in src/tensorsph/__init__.py; the projection time integrator itself diverges at step 13 (honest FAIL)
- L4 not reached: the anchor's dynamic half (spurious currents, Adami 0.1 limit) is an honest FAIL (nan); no registered referee row in tensor_benchmark_cases for CASE-LAPLACE-SPH (0 rows)

| 级别 | 判据 | 结论 |
|---|---|---|
| L1 | 数学/单元验证 | 通过（11 项，含泊松算子零场测试、xfail 钉住附流未修前不得宣称） |
| L2 | 独立功能实现（输入→计算→输出） | **通过**：hex 载流子→CSF 状态→CG 泊松→Δp/残差，可复现到 1e-16 |
| L3 | 系统集成 | 不通过：无主求解器/包导出调用方；时间推进发散 |
| L4 | 基准验证 | 未达：动态半场 FAIL + 注册裁判行缺失 |
| L5 | 科研工程成果 | 未达 |

## 8. 工件

- `artifacts/audit_m15396/sha256manifest.json`
- `artifacts/audit_m15396/referee.json`
- `artifacts/audit_m15396/judge.json`
- `artifacts/audit_m15396/repro.json`
- `artifacts/audit_m15396/finding.json`
- 交付副本 `artifacts/audit_m15396/branch_e02ee87/`（9 文件）

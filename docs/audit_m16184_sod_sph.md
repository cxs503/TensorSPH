# 交差审计 m16184：CASE-SPH-SOD-SHOCK-COMPRESSIBLE（可压 SPH 激波管 Sod 问题，TensorSPH）

> 审计单 m16254（board-audit）· 2026-11-12 · 审计人=本席（独立于 m16184 执行席）
> 结论先行：**verdict=L4 落表（tensor_maturity_verdicts id=462），基准判据 PASS（11/11 项 <0.78% < 3% < 5%），三件套全部实证通过。**

## ① 案例介绍（被审对象）

**物理背景**：一维可压缩 Euler（γ=1.4 理想气体）Sod 黎曼问题——非定常激波管，
解析解含右行激波、接触间断、左行稀疏波四区结构，CFD/SPH 激波捕捉的最小标准锚点。
被审交付=m16184（succeeded，PR #537 已并，commit b6deb05）：TensorSPH 新增可压 SPH
（理想气体 Euler + 能量方程，Inutsuka 2002 算术平均压力项 + Monaghan 人工黏性 + RK2 + 镜像 ghost）。

**几何**：无量纲一维域 x∈[0,1]，膜片 x₀=0.5，两端镜面 wall。
**ref**：Sod 1978 JCP 27(1):1–31, doi:10.1016/0021-9991(78)90023-2（精确 Riemann 解）。
**格式**：Inutsuka 2002 JCP 179:238, doi:10.1006/jcph.2002.7064。

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| γ | 1.4 | N | 800 |
| ρ_L / p_L | 1.0 / 1.0 | ρ_R / p_R | 0.125 / 0.1 |
| t_end | 0.2 | h/dx | 2.5 |
| α_AV / β_AV | 1.0 / 2.0 | CFL | 0.15 |

## ② 怎么做的（审计方法）

审计=独立复现+逐项对表，不采信报告文字：
1. **取证**：读 m16184 工作树（/root/tp-agent/work/m16184）runs/、artifacts/、docs/、tests/、src/；
   读 tensor_reports id=15299 全文（六段结构核验）；读 tensor_dev_missions id=16184 的 goal/result。
2. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE
   anchor_slug='CASE-SPH-SOD-SHOCK-COMPRESSIBLE'` → **0 行**（锚点未入 benchmark 表，见 ⑤ 注意点 1）；
   判据改以注册于 tensor_case_pool id=123 的 tol_policy 与 mission goal 为准：
   「<5% | 指标: 密度/压力平台值与激波位置 vs Sod 精确解 (t=0.2)」。
3. **ref 独立复核**：星区值 p*=0.30313, u*=0.92745, ρ*_L=0.42632, ρ*_R=0.26557 与 Toro 1999 表 4.1
   一致；波位置用 Rankine-Hugoniot/特征线手算复核（x_head=0.26336, x_tail=0.48595,
   x_contact=0.68549, x_shock=0.85044）——与 reference.json 全吻合。
4. **got 逐项核**：production.json 的 11 项 rel_err 与报告 ③/⑥ 表逐项一致（零手抄，程序化比对）。
5. **repro 实证**：审计席亲跑 `python scripts/run_case_compressible_sod.py`（42 s）与
   `python benchmarks/case_sod_shock_compressible/run.py`；重跑产物 vs 提交产物逐字段 diff。
6. **测试实证**：`python -m pytest tests/test_compressible_sod.py -q` → **17 passed (12.9 s)**；
   确认 tests/conftest.py 将本地 src 置顶 sys.path（排除邻树 editable 安装串扰）。

## ③ 计算结果（审计复算，数字全部读自工件 JSON）

**基准判据（唯一裁判）：11 项 gated 指标 vs Sod 1978 精确解，t=0.2**

| 指标 | got | exact | rel.err | <5% | <3% |
|---|---|---|---|---|---|
| 左平台 ρ | 1.0000 | 1.00000 | +0.00% | ✔ | ✔ |
| 左平台 p | 1.0000 | 1.00000 | +0.00% | ✔ | ✔ |
| ρ*_L | 0.4240 | 0.42632 | −0.54% | ✔ | ✔ |
| p*_L | 0.3008 | 0.30313 | −0.78% | ✔ | ✔ |
| ρ*_R | 0.2655 | 0.26557 | −0.04% | ✔ | ✔ |
| p*_R | 0.3029 | 0.30313 | −0.08% | ✔ | ✔ |
| u* | 0.9279 | 0.92745 | +0.05% | ✔ | ✔ |
| 右平台 ρ | 0.1250 | 0.12500 | +0.00% | ✔ | ✔ |
| 右平台 p | 0.1000 | 0.10000 | +0.00% | ✔ | ✔ |
| x_contact | 0.6848 | 0.68549 | −0.10% | ✔ | ✔ |
| x_shock | 0.8502 | 0.85043 | −0.02% | ✔ | ✔ |

全场 L1(ρ)=1.14%、L1(p)=1.40%；接触间断压力跳变 0.70%·p*；质量守恒 0.0；总能量变化 −1.5e-6。

**复现确定性**：重跑后 production.json/studies.json 与提交版逐字段 diff——**仅 wall_s（机时）与
时间戳不同，全部物理量 bit-identical**；3 张 PNG 字节级相同。

**测试**：17 passed（含判据式断言：<5%/<3%、16 条逐项、接触平衡、守恒、bit-identical、
N=200→800 L1 单调、h=1.3dx 反例）。

## ④ 结果图

1. `audit_error_vs_tolerance.png`（本审计新生成，程序化读 production.json）：11 项 |rel err| vs 3%/5% 判据线；
2. `sod_profiles.png`（m16184 工件）：ρ/u/p 剖面 SPH vs 精确解；
3. `studies.png`（m16184 工件）：分辨率阶梯 + h/dx 敏感性 + α_AV 敏感性。

## ⑤ 注意点（审计发现）

1. **基准表缺行（bookkeeping gap，非功能缺陷）**：tensor_benchmark_cases 无
   anchor_slug='CASE-SPH-SOD-SHOCK-COMPRESSIBLE' 行（0 行，已查全表）；判据实际注册于
   tensor_case_pool id=123（tol_policy 与 mission goal 一致）。建议后续把该锚点补录进
   tensor_benchmark_cases 以闭合「benchmark 以百计铺开」的台账链。
2. **邻树 editable 安装串扰风险**：本机 `import tensorsph` 默认解析到 m14539 工作树
   （无 compressible 模块）；本仓 tests/conftest.py 已把本地 src 置顶，pytest 路径安全，
   但裸 `python -c "import tensorsph.compressible"` 会失败——repro_cmd 走 scripts/ 入口不受影响。
3. 复跑会改写 runs/ 与 REPORT.md 的 wall_s/时间戳（物理量不变）；审计后已恢复原工件。
4. 判据口径：case-pool 判据只 gate 平台值+波位置（11 项）；全场 L1(ρ)=1.14% 未入判据，
   属 SPH 间断抹平的合理量级，报告已如实声明。

## ⑥ 误差判定（逐项，显式回答）

上表 11/11 项：**是否 <3%？——是（最大 0.78%，p*_L）**；是否 <5%（判据）？——是。
全场 L1(ρ)=1.14% < 3%：是。无粉饰项；接触间断密度剖面被核抹平 ~3–4h 为格式固有，已留图证。

## 判级（L0–L5）

- L2 独立功能实现：✔ src/tensorsph/compressible/sod.py（553 行真代码：输入→计算→输出）
- L3 系统集成：✔ tensorsph.compressible 子包公开 API + 算例池条目 + 测试面
- **L4 基准验证：✔ 完整算例 + 误差分析（收敛阶梯+3 组敏感性）+ 可复现（审计席亲跑 42 s 复现）**
- L5 科研工程成果：✘（标准验证算例，非科研级工程成果）

**verdict = 4**，evidence 全指针见 verdict_m16184.sql / tensor_maturity_verdicts id=462。

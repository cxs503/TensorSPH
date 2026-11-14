# 方法选择证据登记表的「指标挑选」审计 —— mission 20432

**锚点**：`XVAL-SPH-METHOD-SELECTION-2026-11-12`（TensorSPH 的 method-selection-2026-11-12 做法）
**本单类型**：验证欠账（把已合并的方法选择登记表对"产物自带的全部门限量"复核，并补登记）

## ① 案例介绍

TensorSPH 的方法选择器（`src/tensorsph/method_selection.py`）不是按经验选格式，而是**按证据选**：
每个格式（6 个主格式 + 4 个附加项）必须有一条机器可读的验证记录
（`docs/reports/method_selection_registry.json`），记录里写着"在某算例上的实测误差 [%]"与该算例自己的
验收门；误差 > 15%（owner 精度标准）就不允许被选为推荐格式。

这张登记表有一个结构性弱点：**每个 (方法, 算例, 产物) 只登记一个百分比**。而一个基准产物通常同时
报告多个量、每个量都有自己的门限。只登记最"好看"的那个量，就能让一个格式被报成 5.7 % 精确，而
真正定义物理主张的那个量（例如无滑移壁面的壁面剪应力）差 32 %——这就是按指标挑选（metric shopping），
从登记表本身完全看不出来。

本单审计的对象就是这批产物（全部为仓内已提交的 JSON，非重算）：

| 产物 | 背书的格式 | 本单前登记量数 | 本单后 |
|---|---|---|---|
| `docs/reports/wave_flume_numbers.json` | wcsph_delta_sph | 1 | 1 |
| `benchmarks/case_sod_shock_compressible/result.json` | compressible_riemann | 1 | 1 |
| `benchmarks/case_conduction_1d/result.json` | heat_conduction_operator | 1 | 1 |
| `benchmarks/laplace_law/validation_report.json` | csf_surface_tension | 1 | 1 |
| `benchmarks/csf_consistent/validation_report.json` | csf_surface_tension | 1 | 2 |
| `benchmarks/density_diffusion_decay/decay_rate.json` | density_diffusion | 1 | 2 |
| `benchmarks/isph_channel/results.json` | isph_projection | 1 | 2 |
| `benchmarks/xval_sphinxsys_compression/records.json` | compression_regularization | 1 | 1 |
| `benchmarks/particle_shifting/results.json` | particle_shifting | 1 | 1 |
| `benchmarks/couette/out/results.json` | wcsph_wall_no_slip | 1 | 3 |
| `docs/xval_sphinx_tvc/data/grid_dx003.json` 等 | transport_velocity_correction | 2 | 2 |

## ② 怎么做的

1. `benchmarks/method_selection/audit_quantity_coverage.py`（新增，独立实现）**不 import**
   `build_registry.py`，直接读产物 JSON，枚举"产物自己带门限的量"（有 `limit` / `tol` / 该量自己的
   `pass` 标志 / 该量自己的门限文本），共 16 个。
2. 对每个量**从原始 measured / reference 对重算百分比**（`recompute=direct`，19 行审计行里 18 行如此），
   再与产物里已存的百分比字段交叉核对——"没人复算过的百分比是主张，不是证据"。
   唯一 `stored` 行（SPHinXsys 静水压强带误差）已显式标注原因。
3. 用记录里的 `extractor` 键做覆盖匹配（不做字符串相似匹配），逐条比对登记值 vs 独立复算值
   （相对容差 1e-9）。
4. 把缺登记的量补进 `build_registry.py` 的 `SOURCES`（新增 5 条：Couette 壁面剪应力、Couette 质量
   流量、ISPH 动量收支壁面剪应力、consistent-CSF 截断力、密度扩散有限时长振幅），重跑登记表生成器
   → 记录 12 → 17，路由矩阵 6 个标准画像的期望路由**全部未变**（`unexpected_routes=[]`）。
5. 决策卡诚实化（`src/tensorsph/method_selection.py`）：`Decision` 新增 `worst_banked_pct` /
   `worst_banked_record`，决策卡把"claimed bound（只取过 15 % 的记录）"与"worst banked record
   （含失败的记录）"**分两行写明**；路由回退说明从"只显示第一条 caveat"改为显示全部 caveat。
6. 新增 `tests/test_method_selection_coverage_audit.py`（6 个测试）把覆盖完整性、复算一致性、
   壁面剪应力发现、决策卡语义、未测量门限清单锁成不变量。

复现：

```bash
PYTHONPATH=src python benchmarks/method_selection/build_registry.py
PYTHONPATH=src python benchmarks/method_selection/audit_quantity_coverage.py
PYTHONPATH=src python -m pytest tests/test_method_selection_coverage_audit.py \
    tests/test_method_selection.py tests/test_xval_method_selection_matrix.py -q   # 33 passed
```

## ③ 计算结果

**覆盖与一致性（程序化注入，零手抄）**

| 项 | 值 |
|---|---|
| 登记表记录数 | 17（原 12） |
| 产物自带门限的量 | 16 |
| 覆盖完整 | **是**（未登记：无） |
| 复算 vs 登记值不一致 | 0 |
| 产物内已存百分比 vs 复算不一致 | 0 |
| 复算方式 | 19 行审计行：direct 18 行 / stored 1 行（已注明）；登记记录 17 条 |

**方法级：claim vs 最差已登记量**

| 方法 | claimed bound（仅过 15 % 的记录）[%] | 最差已登记量 [%] | 差额 [pp] |
|---|---|---|---|
| `wcsph_wall_no_slip` | 7.16 | **32.65** | **25.5** |
| `csf_surface_tension` | 5.99 | 23.80 | 17.8 |
| `transport_velocity_correction` | 10.81 | 16.96 | 6.15 |
| `wcsph_delta_sph` | 2.97 | 2.97 | 0 |
| `compressible_riemann` | 0.777 | 0.777 | 0 |
| `heat_conduction_operator` | 0.000218 | 0.000218 | 0 |
| `density_diffusion` | 1.02 | 1.02 | 0 |
| `isph_projection` | 0.867 | 0.867 | 0 |
| `compression_regularization` | 5.53 | 5.53 | 0 |
| `particle_shifting` | 8.83 | 8.83 | 0 |

**Couette（无滑移格式）三个量**：interior L2 速度 5.65 %（自身门 5 %，已超）、跨槽质量流量 7.16 %
（自身门 5 %，超）、**壁面剪应力 32.65 %（自身门 5 %，超 6.5 倍）**。

**其它发现**：Couette 产物的自评是 `verdict.status = "FAIL"`，但 `verdict.gated_results` 与
`verdict.reasons` **都是空的**——从存下来的数据无法判断它究竟因哪个门限判 FAIL（已列入
`unbacked_gates`）。同一产物还有 `steady_drift` / `control_flux` 两个门限声明了却没有测量值。

## ④ 结果图

`docs/figures/method_selection_m20432/coverage_audit.png`：补登记后**每一个**被背书产物的**每一个**
带门限的量（对数横轴），绿色 = 在 owner 15 % 内，红色 = 超出；蓝色竖刻度 = 该算例自己的门；
虚线 = owner 15 % 标准。红色三根即本单暴露的量：Couette 壁面剪应力 32.6 %、consistent-CSF 粗网格
曲率 23.8 %、TVC 粗网格 17.0 %。

## ⑤ 注意点 / 局限

1. **登记表只能审计仓内产物已存的东西**。若某个量根本没有被算过（例如 Couette 的 `steady_drift`），
   审计只能把它列进 `unbacked_gates`，不能替它补算——那需要重跑 Couette 算例。
2. **非百分比门限被排除在记录 schema 之外**（`isph_channel` 的密度脉动上限、TVC 的 MAE 绝对门、
   1D 导热的若干归一化门），已在 `unbacked_gates` 中逐条列原因；这不是"通过"，是"不适用"。
3. **`interior_l2` 的定义**排除了最贴壁的两行（`rows[0]`、`rows[-1]`）。复算严格照该定义执行，同时另
   登记了一行"含贴壁两行的 RMS = 6.00 %"，以免定义差异掩盖贴壁误差。
4. **Couette 只有一档分辨率**（`ladder` 长度 1），因此 32.65 % 无法判定是分辨率敏感还是边界条件
   本身的问题；这是下一单的验证欠账，不是本单能下的结论。
5. **SPHinXsys 静水压强带误差** 只有产物里存好的百分比可用（产物未存原始压强剖面），故该行标
   `recompute=stored`，其独立性弱于其余 16 行。
6. 覆盖匹配用 `extractor` 键（数据链接），因此**新增量必须同时改审计目录与登记表**；两侧不一致会
   在 `test_every_gated_quantity_is_banked` 处失败——这是有意的摩擦。

## ⑥ 误差判定（对上 owner 15 % 标准，逐项显式回答）

| 关键量 | 值 | 基准 | 误差 | < 15 %？ |
|---|---|---|---|---|
| 审计复算值 vs 登记值（17 条） | 同 | 独立复算 | < 1e-9 相对（整机 0 条不一致） | **是** |
| 产物已存百分比 vs 审计复算（9 条可交叉核对） | 同 | 同上 | < 1e-9 相对（0 条不一致） | **是** |
| 覆盖完整性 | 16/16 带门限量已登记 | 产物自带门限清单 | 0 缺口 | **是** |
| `wcsph_wall_no_slip` 壁面剪应力（Couette） | 32.65 % | `tau = mu U / H` 解析 | 32.65 % | **否**（超 2.17 倍） |
| `wcsph_wall_no_slip` 跨槽质量流量 | 7.16 % | 解析 `Q = U H / 2` | 7.16 % | 是（但超自身 5 % 门） |
| `wcsph_wall_no_slip` interior L2 速度 | 5.65 % | `u = U y / H` | 5.65 % | 是（但超自身 5 % 门） |
| `csf_surface_tension` 曲率（粗网格，3 mm） | 23.80 % | `kappa = 1/R` | 23.80 % | **否**（细网格 1.5 mm 为 10.5 %） |
| `transport_velocity_correction` 粗网格（40 mm） | 16.96 % | Martin & Moyce 数字化点 | 16.96 % | **否**（30 mm 为 10.81 %） |

结论（honest）：本单修的是**证据完整性与诚实性**，两项自查量（复算一致性、覆盖完整性）均为**零误差**；
但它同时暴露出三个**真实超差量**——其中 `wcsph_wall_no_slip` 的壁面剪应力 32.65 % 是对外声称
"无滑移壁面"的格式最不该失守的量。因此：**"TensorSPH 的 WCSPH 无滑移壁面在壁面剪应力上满足 15 %
标准"这句话不成立**，本单把它写成登记表里的失败记录与决策卡上的显式警告，而不是删除或平均掉。
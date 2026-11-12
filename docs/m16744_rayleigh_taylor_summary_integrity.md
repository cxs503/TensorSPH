# CASE-SPH-RAYLEIGH-TAYLOR-MP 工件完整性修复（m16744）

**锚点**: `NEW:RT 汇总工件 merge-by-tag 缺陷`（总账无对应条目；m16318 交差审计返修清单第 5、6 项）
**仓**: TensorSPH ｜ **分支**: `agent/m16744` ｜ **日期**: 2026-11-12
**性质**: 缺陷修复（工件完整性/可复现性），**不改物理、不动任何数值**

---

## ① 案例介绍

`CASE-SPH-RAYLEIGH-TAYLOR-MP`（双相 WCSPH 瑞利–泰勒线性增长率）已在总账登记 `done`，
本单不重做该算例的物理结论，只修 **m16318 交差审计列出的两处工件缺陷**：

| 审计发现 | 缺陷 | 后果 |
|---|---|---|
| 第 5 项 | `benchmarks/rayleigh_taylor/run.py` 每次调用都用**本次**配置列表重写 `runs/summary.json` | 分次/分片跑完的战役在汇总里只剩最后一次的行：仓内实测 **8 个 case 文件 / 1 行 summary** |
| 第 6 项 | `n_fit = 0` 的档位把拟合结果写成 `NaN` | `NaN` 不是合法 JSON（RFC 8259 §6），严格解析器（`json.loads(parse_constant=...)`、`jq` 严格模式、PG `json` 转换）直接拒文件；且与「字段不适用」无法区分 |

参数环境（供复现，取自各 case 文件 `config`）：两层流体 `rho_h = 1500`、`rho_l = 500`（Atwood A = 0.5），
单模态界面扰动 `eta0 = 0.01 λ`，声速 `c0 = 40 m/s`，`t_end = 0.45 s`，拟合窗 `0.02–0.1 s`，
网格 `dx = 0.0125 … 0.05 m`；理论锚 `sigma = sqrt(A g k)`（Taylor 1950）。

## ② 怎么做的

1. 新增 `benchmarks/rayleigh_taylor/summary_io.py`（纯函数，无 torch 依赖）：
   - `load_case_rows()` 以 `runs/<tag>.json` 为**唯一真值源**；`summary/NON_CASE_STEMS`（`summary/table/reference/damping/spurious/aggregate`）不作 case 记录，避免自递归；
   - `write_summary()` 三级优先级合并 **按 tag**：本次调用 > 每例文件 > 旧 summary（旧 summary 作基底，使 case 文件被归档后行不丢）；
   - `sanitize_row()` 显式给 `status ∈ {ok, no_fit}`，`no_fit` 行把 `sigma_over_ideal / *_finite_depth / *_viscous` 置 `null`，并递归把任何非有限浮点转 `null`；
   - `dumps_strict()` 用 `allow_nan=False` 写盘——**写不出 NaN 就报错**，从结构上堵死复发。
2. `run.py`：接入上面三条路径；新增 `--aggregate-only`（零重算，从磁盘工件重建 summary）；
   单例打印在无拟合窗时输出 `no_fit` 而不是格式化 `nan`。
3. 从仓内既有 8 个 case 文件**重算零物理、仅重建视图**：`run.py --aggregate-only` → 8 行。
4. 图：`make_summary_integrity_figure.py` 程序化读 `runs/` 生成
   `docs/figures/m16744/rt_summary_integrity.png`（覆盖度柱 + 全部行的 sigma 比值）。

## ③ 计算结果（程序化注入，无手抄）

重建后 `runs/summary.json`（严格 JSON 解析通过）：

| tag | status | sigma / √(Agk) | 偏差 |
|---|---|---|---|
| dx0.0125_nu0.001_a0.01 | ok | 0.7552 | −24.48 % |
| dx0.02_nu0.001_a0.01 | ok | 0.7154 | −28.46 % |
| dx0.025_nu0.001_a0.01 | ok | 0.5576 | −44.24 % |
| dx0.025_nu0_a0.01 | ok | 0.5832 | −41.68 % |
| dx0.025_nu0.001_a0.02_amp | ok | 0.6156 | −38.44 % |
| dx0.025_nu0.01_a0.01 | ok | 0.3702 | −62.98 % |
| dx0.04_nu0.001_a0.01 | ok | 0.3828 | −61.72 % |
| dx0.05_nu0.001_a0.01 | **no_fit** | `null` | 无有效拟合窗（`n_fit = 0`） |

覆盖度：case 文件 8 → 修复前 summary 1 行 → 修复后 8 行（`no_fit` 1 行显式标记）。
理论值 `√(Agk) = 5.551488 s⁻¹`（A = 0.5、g = 9.81、k = 2π/λ）。

## ④ 图

`docs/figures/m16744/rt_summary_integrity.png`（两联：(a) 覆盖度 8/1/8；(b) 全部 8 行
`sigma/√(Agk)` vs `dx`，含 10 % 验收带与 `no_fit` 标记）。
纯场量云图不适用：本单不产生新的流场，只重建工件视图，故 (b) 用**既有测量的比较曲线**，
无新云图；这是刻意声明而非省略。

## ⑤ 注意点 / 局限

- **修复不改善物理**：所有档位仍显著低于线性理论（最差 −62.98 %），与 m16318 审计结论一致，
  物理 FAIL 不被本单覆盖或粉饰。
- `table.json` 里历史 `NaN` 未重写（本单只修 `summary.json` 视图；`table.json` 由 `make_doc.py`
  另行生成，标为遗留项）。
- 未逐一排查其它 benchmark 的 runner 是否有同型「重写 summary」缺陷（只做了 `benchmarks/`
  关键字扫描，未逐文件复核）——**not_run**。
- `summary.json` 现为视图而非证据：任何裁决必须连同 `runs/<tag>.json` 一起引用。

## ⑥ 误差判定

本单判据不是物理精度，而是工件一致性，逐项列明：

| 判据 | 目标 | 实测 | 是否达标 |
|---|---|---|---|
| summary 行数 vs case 文件数 | 相等（8 vs 8） | 8 vs 8 | ✅ |
| summary 严格 JSON（拒绝 NaN/Infinity） | 解析通过 | 通过；`no_fit` 行数值为 `null` | ✅ |
| 按 tag 合并幂等（重复聚合字节稳定） | 文本不变 | 不变（测试固化） | ✅ |
| 真实短跑端到端（`dx=0.05`，`t_end=0.02`，1 s） | 合并后仍 8 行 | 8 行，1 个 `no_fit` | ✅ |
| 物理量 sigma 相对 `√(Agk)` | 算例原判据 <10 % | 最好 −24.48 % | ❌ **honest FAIL**（本单不负责，如实保留） |

**关键量误差是否 <3 %**：本单无新增物理量、无新基准对标，3 % 门槛不适用；
被保护的既有物理量 sigma 相对理论最好档仍差 −24.48 %，**远大于 3 %，且本单明确未改善**。

## 复现

```bash
PYTHONPATH=src python benchmarks/rayleigh_taylor/run.py --aggregate-only   # 重建视图，零重算
PYTHONPATH=src python -m pytest tests/test_rayleigh_taylor_summary_merge.py -q
PYTHONPATH=src python benchmarks/rayleigh_taylor/make_summary_integrity_figure.py
```

## 测试

`tests/test_rayleigh_taylor_summary_merge.py` —— 12 项，全部通过（0.20 s）：
合并保行 / 同 tag 覆盖不重复 / 磁盘重建 / 重复聚合字节稳定 / `no_fit` 非 NaN /
嵌套非有限值置 null / `dumps_strict` 拒 NaN / `table·reference·damping` 不作 case 文件 /
旧 summary 作基底且 case 文件优先 / 本次调用最高优先 / 仓内工件严格解析且行数覆盖全部 case 文件。
回归：`tests/test_rayleigh_taylor.py` 10 项同时通过（合计 22 passed / 50 s）。
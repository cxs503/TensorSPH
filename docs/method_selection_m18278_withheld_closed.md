# XVAL-SPH-METHOD-SELECTION — m18278 increment: closing the WITHHELD holes

Anchor: `XVAL-SPH-METHOD-SELECTION-2026-11-12`（前序 m18056 建路由与证据登记；本单补验证欠账）
仓库: TensorSPH · 分支: agent/m18278 · owner 精度标准: 误差 ≤ 15%

## 1. 为什么要做（缺口）

m18056 落地的 `src/tensorsph/method_selection.py` 只对**有记录**的方案给 `SELECTED`，
无记录一律 `WITHHELD`。其登记表留下 4 个洞：

| 方案 | m18056 状态 | 原因 |
|---|---|---|
| `isph_projection` | WITHHELD | 仓内有 `benchmarks/isph_channel/run.py`，但没有落盘结果 |
| `compression_regularization` | WITHHELD | `benchmarks/xval_sphinxsys_compression/run.py` 默认参数即崩（见 §4） |
| `particle_shifting` | WITHHELD | `benchmarks/particle_shifting/results.json` 已在库，但没被登记 |
| `transport_velocity_correction` | WITHHELD | 由在飞单 m18253 负责，**本单不做**（避免重复） |

`WITHHELD` 的直接后果：库内明明实现了 ISPH 投影，Couette 类算例却被路由到
较弱的 WCSPH 无滑移分支（`wcsph_wall_no_slip`），只是"因为没有记录"。

## 2. 怎么做的

1. **把已有驱动跑到可复现**（本机 128 核，load15≈118，故只用 2 线程、轻量 2D 算例）：
   - `PYTHONPATH=src python benchmarks/isph_channel/run.py --out runs/isph_channel/m18278`
     → 3 档 `H/dx ∈ {8,12,16}`，40 s；结果固化到 `benchmarks/isph_channel/results.json`
   - `PYTHONPATH=src python benchmarks/xval_sphinxsys_compression/run.py --out runs/xval_compression/m18278/records.json`
     → 4 个 arm（summation / compression_none / refmean / refmean_internal），3.5 min；
     结果固化到 `benchmarks/xval_sphinxsys_compression/records.json`
   - `benchmarks/particle_shifting/results.json` 已在库，直接登记（不重跑）。
2. **登记器扩展**：`benchmarks/method_selection/build_registry.py` 新增
   `x_isph_channel` / `x_compression_regularization` / `x_particle_shifting`
   三个抽取器与对应 `Source`，注册表、路由矩阵、表格、图、报告全部**程序化重建**
   （每个记录带 source path + sha256；`--check` 为防漂移门）。
3. **路由证据定域（防跨案例外推）**：`isph_projection` 的记录来自**壁面槽道**，
   不能用来给周期涡类算例签发通行证。因此在 wall-bounded 规则里加了
   `_vortex_class_case()` 过滤：只有当记录案例名属 vortex / Taylor-Green 类时，
   Gresho 类剖面才可能被 `SELECTED`；否则仍 `WITHHELD`。

## 3. 结果（全部由注册表 JSON 程序化注入，零手抄）

| 方案 | 案例 | 量化误差 | 自身门限 | owner 15% | 判定 |
|---|---|---|---|---|---|
| `isph_projection` | CASE-ISPH-CHANNEL（H/dx=16，中心速度 vs 解析 Poiseuille 抛物线） | **0.87 %** | 3 % ✅ | ✅ | SELECTED |
| `isph_projection`（同源诊断） | 壁面剪切动量收支 vs `tau=rho0*g*H/2` | 0.046 % | 3 % ✅ | ✅ | — |
| `compression_regularization` | XVAL-SPHINXSYS-COMPRESSION（refmean arm，静水压力 vs `p=rho0 g(H-y)`） | **5.53 %** | 1 % ❌ | ✅ | SELECTED（带 caveat） |
| `particle_shifting` | CASE-PARTICLE-SHIFTING-DAMBREAK（B2 前锋位置相对未开启答案，按 0.2 m 水柱归一） | **8.83 %** | 10 % ✅ | ✅ | SELECTED |

ISPH 分辨率梯子（中心速度相对误差）：`H/dx=8` 2.81 % → `12` 1.37 % → `16` 0.87 %，
一阶收敛趋势与投影法一致；密度涨落最大 5.8e-4（限 1e-3）。

路由矩阵（`docs/reports/method_selection_m18056.json`，`unexpected_routes=[]`）：

| 剖面 | m18056 | m18278 |
|---|---|---|
| 平面 Couette 槽道 | `wcsph_wall_no_slip`（ISPH 无记录而降级） | **`isph_projection`** |
| Gresho 涡（周期、无粘） | 拒选 | 拒选（现为"无涡类记录"，而非"完全无记录"） |
| 其余 5 个剖面 | 不变 | 不变 |

登记覆盖：`methods_without_evidence` 由 4 项降为 1 项（只剩 TVC）。

## 4. 注意点 / 踩过的坑

- **驱动真有缺陷**：`benchmarks/xval_sphinxsys_compression/run.py` 默认参数写成
  `str("xval_output" / ...)`，在 `str` 上调 `/` 直接 `TypeError`，即**默认运行必崩**。
  已最小修复为 `str(Path("xval_output") / ...)`（`from pathlib import Path` 已在文件内）。
- **compression 正则化自身 1% 静水门限未过**（5.53 %，且基线 summation 为 2.52 % 也未过）：
  该 arm 在 owner 15% 内可被选中，但报告如实给 caveat —— 静水压力精度关键算例不宜用它替代
  delta-SPH。这不是造假可填的坑：门限保持 1%，登记里 `own_gate_passed=false`。
- **particle_shifting 的记录是"相对未开启答案"的自洽量**，不是实验/解析基准；
  它能证明"开启移位不改变已发布答案"，不能证明"更准"。8.83 % 的物理解释是
  B2 溃坝前锋在 t=0.8 时的位移差 0.0177 m / 0.2 m 水柱；`resting` 案例的晚时 RMS 速度
  从 0.0081 升到 0.0091 m/s（限 0.06 m/s），即引入的伪流未越限。
- **Gresho 类仍无直接证据**：投影法在周期涡上的精度未验证，故保留拒选；
  建议新增条目 `XVAL-SPH-PROJECTION-TAYLOR-GREEN`（验收：ISPH 在 Taylor-Green
  涡上 L2 速度误差 <15%，并给出分辨率收敛），属 TensorSPH 域。

## 5. 误差判定（逐项回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| ISPH 中心速度（H/dx=16） | 解析 Poiseuille `u_max = f·H²/(8ν)` | 0.87 % | **是** |
| ISPH 壁面剪切（动量收支） | `tau = rho0 g H / 2 = 490.5` | 0.046 % | **是** |
| ISPH 中心速度（H/dx=8，最粗） | 同上 | 2.81 % | **是**（临界） |
| compression refmean 静水压力 | 解析 `p = rho0 g (H-y)` | 5.53 % | 否（<15 %，自身 1 % 门限未过） |
| particle_shifting 前锋位置差 | 未开启移位的 B2 答案 | 8.83 % | 否（<15 %；自洽量非解析基准） |

## 6. 复现

```bash
PYTHONPATH=src python benchmarks/isph_channel/run.py --out runs/isph_channel/m18278
PYTHONPATH=src python benchmarks/xval_sphinxsys_compression/run.py \
    --out runs/xval_compression/m18278/records.json
PYTHONPATH=src python benchmarks/method_selection/build_registry.py           # 重建登记
PYTHONPATH=src python benchmarks/method_selection/build_registry.py --check   # 防漂移门
PYTHONPATH=src python -m pytest tests/test_method_selection.py -q             # 20 passed
```
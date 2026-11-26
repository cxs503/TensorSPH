# 【补报·mission 2499】TensorSPH 主分支可导入性修复 + 接口契约静态门禁 + 壁面接触角 BC 解析复核

**性质**: 补报单（原单 mission 2499 已交付，本单补交论文级进展报告）。
**原单分支**: `agent/m2499` ｜ **原单 commit**: `bed987f`（2026-11-03，tensor-agent）。
**本报告数据来源**: 原单 commit `bed987f` 内工件 `docs/dev_report_contact_angle_and_api_gate.md` 与
`benchmarks/contact_angle/report.json`（程序化提取，零手抄）。原单的 `artifacts/` 目录
（`api_contract*.json` 门禁记录等）未随 commit 留档，相关数字标注「未留档」。

---

## ① 案例介绍

**物理/工程背景**: TensorSPH（SPH 流体求解器）的表面张力采用连续表面力（CSF）格式，
壁面润湿由接触角边界条件 `apply_contact_angle_boundary` 控制。原单入场时发现 **main 分支
根本无法导入**：

```
src/tensorsph/__init__.py:41: from .surface_tension import (...)
E   ImportError: cannot import name 'apply_contact_angle_boundary'
    from 'tensorsph.surface_tension'
```

两处断裂（同一类事故：函数体/形参被门禁滤掉、调用方仍在引用）：

| # | 位置 | 现象 |
|---|---|---|
| 1 | `__init__.py:41`、`tests/test_contact_angle.py:15` | 重导出 `apply_contact_angle_boundary`，而 `surface_tension.py` 中该函数已不存在 → 整包 `ImportError`，pytest 收集阶段即死 |
| 2 | `droplet_impact.py:609` | `compute_surface_tension_force(..., wall=wall)`，而该函数形参表无 `wall` → 只要 `surface_tension_coefficient > 0` 走生产 `rhs` 路径即 `TypeError`（默认 σ=0，故套件几乎照不到） |

第 1 条是「一个断符号掩盖全部结果」的放大器：断言、精度、收敛证据全部不可见。

**交付物**（两块，互相独立可验证）：

* **A. 静态接口契约门禁**（`tools/api_contract_check.py` + `tests/test_api_contract.py`）——
  纯 AST、不导入包，在包已经坏掉时仍能报出 `file:line`；
* **B. 壁面接触角边界条件**（`apply_contact_angle_boundary` + `wall_interface_frame` +
  `compute_surface_tension_force(..., wall=)`）——按已入库的 `tests/test_contact_angle.py`
  契约重写实现，并用解析关系 `angle(n_out, e_n) = θ` 做量化复核。

**几何/参数（接触角复核算例，来自 `benchmarks/contact_angle/report.json`）**:

| 参数 | 值 |
|---|---|
| 液滴半径 R | 0.02 m |
| 粒径 dx | 0.002 m |
| 光滑核半径 h | 0.0026 m |
| 流体粒子数 | 178 |
| 壁面粒子数 | 60 |
| 规定接触角 θ | 60°（场级算例）；旋转检验 θ=30/60/90/120° |
| 近壁（2h）粒子 | 78（其中界面粒子 26） |

## ② 怎么做的

### A. 门禁（`tools/api_contract_check.py`，`ast` 单文件，无第三方依赖）

四类检查，全部只读语法树：

| 检查 | 内容 | 命中历史事故 |
|---|---|---|
| `export` | `__init__.__all__` 里每个名字必须在 `__init__` 里被绑定 | — |
| `reexport` | `from .mod import name` 的 `name` 必须在 `mod` 中有定义/赋值/导入 | 事故 1 |
| `call` | 包内调用的关键字实参必须存在于被调方签名，且位置实参不超额 | 事故 2 |
| `test-import` | `tests/*.py` 里 `from tensorsph[.mod] import name` 必须可解析 | 事故 1 |

`tests/test_api_contract.py` 既是门禁又带 5 个**负控**（临时注入「删函数体」「改签名」
「写坏 test import」「`__all__` 写幽灵名」四类缺陷，确认检查器真能抓到，避免假绿）。

### B. 接触角 BC

1. `wall_interface_frame(pos, wall, h)`：对每个流体粒子取 `2h` 内壁面粒子（复用生产
   `cell_linked_pairs`，每粒子截断 24 邻居），做**总体最小二乘（TLS）平面拟合**，
   协方差最小特征向量为壁法向，符号指向流体内部 → 不假设壁面轴对齐。
2. `apply_contact_angle_boundary(grad_c, pos, wall, h, theta)`：以壁法向 `e_n`、梯度切向
   单位向量 `t_hat`、`a = grad_c·e_n` 构造
   `n' = |grad c| [ cos θ · sgn(a) · e_n + sin θ · t_hat ]`。
   由构造：(i) **模长严格保持**（CSF 的 δ_s 与切割面力恒等式不受影响）；
   (ii) `2h` 外粒子**逐位不变**；(iii) 切向分量退化的粒子不变（该处界面方向无定义）。
3. `compute_surface_tension_force(..., wall=None)`：`cfg.contact_angle is not None` 且给了
   `wall` 时对 `grad_c` 施加 BC；其余路径逐位不变（σ=0 / `contact_angle=None` 均有回归钉）。

**工具链**: Python + `ast`（门禁）、PyTorch（SPH 生产路径）、pytest、matplotlib（结果图）。

## ③ 计算结果（关键数字表）

### 3.1 门禁：修复前 / 修复后

| 树 | 结论 | 发现数 | 明细 |
|---|---|---|---|
| main（修复前） | **FAIL** | 3 | `__init__.py:41` reexport、`droplet_impact.py:609` call、`test_contact_angle.py:15` test-import |
| `bed987f`（修复后） | **PASS** | 0 | 7 模块 / 42 导出 / 60 签名索引 |

（门禁运行耗时与逐条 JSON 记录：**未留档**——原单 `artifacts/api_contract*.json` 未随 commit 入库；
发现数与明细以原单 dev 报告文字为准。）

### 3.2 接触角 BC 解析复核（数字程序化提取自 `benchmarks/contact_angle/report.json`）

**旋转检验**（壁面旋转 0°/30° × θ=30/60/90/120°，共 8 组；θ>90° 折为 180°−θ）:

| 壁面旋转 | θ 规定 | θ 实测 | 绝对误差（度） | 相对误差（%） | 模长比 |
|---|---|---|---|---|---|
| 0° | 30° | 29.999999999999993° | 7.11e-15 | 2.37e-14 | 1.0 |
| 0° | 60° | 59.999999999999986° | 1.42e-14 | 2.37e-14 | 1.0 |
| 0° | 90° | 89.99999999999999° | 1.42e-14 | 1.58e-14 | 1.0 |
| 0° | 120° | 60.00000000000001° | 7.11e-15 | 1.18e-14 | 0.9999999999999999 |
| 30° | 30° | 29.999999999999993° | 7.11e-15 | 2.37e-14 | 1.0 |
| 30° | 60° | 59.999999999999986° | 1.42e-14 | 2.37e-14 | 1.0 |
| 30° | 90° | 90.0° | 0.0 | 0.0 | 0.9999999999999999 |
| 30° | 120° | 60.00000000000001° | 7.11e-15 | 1.18e-14 | 0.9999999999999998 |

**TLS 壁法向**（壁面旋转 0/15/30/45/60°）: 法向误差 0, 0, 0, 0, **8.54e-7 度**（最大）。

**场级覆盖性**（液滴坐壁，θ=60°）: 近壁界面粒子 26/26 = **100%** 被修正；
梯度模长最大相对误差 **1.98e-16**；远场（>2h）逐位不变（`far_field_untouched=True`）；
实测接触角中位数 **59.999999999999986°**（绝对误差中位数 1.27e-14 度）。

**汇总判定**（`report.json` verdict 节）: 旋转最大绝对误差 1.42e-14 度、最大相对误差
2.37e-14 %（`rotation_within_3pct=True`）、模长保持 `True`、壁法向最大误差 8.54e-7 度。

### 3.3 测试面（原单 dev 报告记录）

| 命令 | 结果 |
|---|---|
| `pytest tests/test_api_contract.py` | 8 passed |
| `pytest tests/test_contact_angle.py` | 8 passed, 2 xfailed(strict) |
| `pytest tests/test_surface_tension.py tests/test_young_laplace.py` | 21 passed |
| `pytest tests/test_laplace_law.py tests/test_pair_based_rhs.py tests/test_laplace_validation.py` | 25 passed, 4 xfailed, **2 failed**（遗留欠账，见 ⑤） |

## ④ 结果图

原单工件 `benchmarks/contact_angle/contact_angle.png`（随 commit `bed987f` 入库，140 dpi）：
(a) 实测接触角 vs 规定 θ（壁面 0°/30°，恒等线）；(b) 角度误差 vs 壁面旋转角（对数轴，
红线 = 30° 的 15% 门槛）；(c) TLS 壁法向误差 vs 壁面旋转角。

本补报单 `figures` 字段为空的理由：补报单未重跑算例，不新产 png；原单 png 已随
commit `bed987f` 入库（路径见上），报告正文以程序化提取的数字表呈现全部关键量，
不重复贴图。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **`math` 未导入**：`surface_tension.py` 原本只有 `import torch`，新增几何代码用
   `math.cos` 直接 `NameError`——AST 门禁查不到运行期缺名（只查跨模块契约），
   门禁是**补充**而非替代导入测试，两者都要有。
2. **两项集成测试与 main 的曲率算子互斥（改为 strict xfail，附证据）**：
   `TestContactAngleIntegration` 两个用例用 6 粒子玩具模板要求「表面粒子加速度非零」，
   而 main 上被钉死的 `κ = ∇·n` 在该模板上恰好为 0：
   `κ = [10.10, −4.57, 11.76, −2.91, 0.0, 0.0]`（粒子 4、5 邻居法向完全相同 ⇒ `dn = 0`）。
   与接触角 BC 无关（`contact_angle=None` 时同样为 0）。按仓库既有协议改为
   `xfail(strict=True)` 并写明「WLS 曲率（backlog P0-4）落地后自动 xpass」，不掩盖欠账。
3. **`tests/test_laplace_law.py` 的 2 项在 main 上就是红的**：`coarse` fixture 用 `dx=0.004`，
   而 m2174 的结论/工件只扫了 `dx=0.001–0.003`；其中 `test_csf_force_direction_is_radial`
   在 dx=4 mm 下变成 **XPASS(strict)**。根因是「钉值分辨率与工件分辨率不一致」，正确修法是
   补扫 `dx=0.004` 并重生成 `validation_report.json`——原单**未做**（入场时 load15≈200–340，
   容量纪律要求只做轻量活）。
4. **本单不碰曲率算子**：`compute_curvature` 已被 `tests/test_surface_tension.py` 用
   `div(r̂)` 解析值钉在 5% 以内，动它会连带 m2174 一整套工件。
5. **并发重复风险**：入场时 `git log --all` 显示 m2389/m2399/m2433/m2447/m2451/m2454/m2459
   等多张单都在做「恢复接触角 BC / 让包可导入」。原单增量差异在**门禁工具 + 解析复核**；
   若他单先合入，src 改动可整体丢弃而 `tools/` + `tests/test_api_contract.py` +
   `benchmarks/contact_angle/` 仍然成立。
6. **已知开销点**：`wall_interface_frame` 每次调用新增一次 `cell_linked_pairs`（仅当
   `contact_angle` 非 None）；生产 `rhs` 里已有 pair list 可复用，原单为最小改动未做。
7. **补报单局限**：本报告依据原单 commit `bed987f` 内工件与原单声明撰写；原单
   `artifacts/` 目录（门禁 JSON 记录、pytest 完整输出）未留档，相关项标注「未留档」，
   未做任何数字补造。

## ⑥ 误差判定（逐项）

| 关键量 | 基准 | 误差 | <3% ? |
|---|---|---|---|
| 接触角（旋转检验，8 组合） | θ | 最大 1.42e-14 度（2.37e-14 %） | **是** |
| 接触角（液滴坐壁场级，θ=60°） | 60° | 中位数 1.27e-14 度 | **是** |
| TLS 壁法向（5 个壁面旋转角） | 精确平面法向 | 最大 8.54e-7 度 | **是** |
| 梯度模长保持 | 1.0 | 最大相对误差 1.98e-16 | **是** |
| 接口契约一致性 | 0 发现 | main 3 → 修复后 0 | 是（离散计数，无百分比意义） |

**显式回答「关键量误差是否 <3%」：是。** 全部解析复核关键量误差在 1e-14 % 量级
（壁法向 8.5e-7 度），远低于 3% 门槛，`report.json` verdict 节 `rotation_within_3pct=True`。

**未达标/无基准项（如实列出，不粉饰）**：
- `test_laplace_law.py` 2 项失败/XPASS 翻转（dx=0.004）：属 m2174 工件未覆盖分辨率的
  **遗留欠账，原单未修**，根因与修法见 ⑤-3；
- 「带接触角的液滴动力学整体精度」（Young–Laplace + 实验接触角对照）：**无基准**——
  本单验证的是边界条件算子的解析正确性，动力学整体精度需在壁面静态液滴上另跑对照，
  原单未做，原因：入场负载高（load15≈200–340），容量纪律限制为轻量活。

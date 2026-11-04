# CSF 曲率算子默认值变更验证：mls → fit_quadratic

**范围**: `src/tensorsph/droplet_impact.py`（`Config.csf_curvature` 默认值）
**变更**: 默认曲率算子由 `"mls"` 改为 `"fit_quadratic"`
**日期**: 2026-11-03 ｜ **Mission**: m2979

## 1. 变更动机

原默认 `"mls"`（MLS 一致散度 `div n`）在二维圆上的曲率误差为 **+25%**，超出平台 15% 判据门。
`"fit_quadratic"`（加权局部二次界面拟合）在同一算例上误差 **< 6%**（dx = 0.002）至 **< 1%**（dx = 0.0015），
且随分辨率收敛（`docs/verification_surface_tension_curvature.md` §3.1）。

变更使新用户默认获得物理准确的曲率，无需显式配置。

## 2. 变更内容

```diff
-    csf_curvature: str = "mls"
+    csf_curvature: str = "fit_quadratic"
```

注释同步更新，说明三种算子的精度特征。

## 3. 验证结果

### 3.1 曲率精度（R = 0.02 m，六方晶格）

| dx [m] | N | mls 误差 | fit_quadratic 误差 | 平台门 15% |
|---|---|---|---|---|
| 0.003 | 161 | +30.0% | +17.8% | mls: **no** / fit: **no** |
| 0.002 | 362 | +24.9% | +5.3% | mls: **no** / fit: **yes** |
| 0.0015 | 642 | +25.2% | +0.9% | mls: **no** / fit: **yes** |

解析值：κ = 1/R = 50.00 1/m。

### 3.2 测试套件

| 测试集 | 结果 |
|---|---|
| `test_csf_consistent.py` (20 tests) | 20/20 PASS |
| `test_csf_curvature_fit.py` (10 tests) | 10/10 PASS |
| `test_surface_tension.py` (15 tests) | 15/15 PASS |
| `test_laplace_law.py` (10 tests) | 9 PASS + 1 XFAIL |
| `test_droplet_impact.py` (12 tests) | 12/12 PASS |
| `test_dam_break.py` (14 tests) | 14/14 PASS |

`test_laplace_law.py::test_curvature_reproduces_circle` 仍为 xfail(strict)，
因为该测试使用 legacy 方案（直接调用 `compute_color_gradient`/`compute_curvature`），
不受默认值变更影响。

### 3.3 向后兼容性

- `csf_curvature="mls"` 仍可用，行为不变（A/B 对照）
- `csf_curvature="fit_circle"` 仍可用（代数圆拟合，格子数据上不稳定）
- `csf_scheme="legacy"` 不受影响（使用旧版邻居数色函数）
- 已有基准/回归数字不变（显式指定 `csf_curvature="mls"` 的算例）

## 4. 误差判定

| 关键量 | 基准 | 旧默认 (mls) | 新默认 (fit_quadratic) | < 15%？ |
|---|---|---|---|---|
| 曲率 κ (dx=0.002) | 50.00 1/m | 62.45 (+24.9%) | 52.66 (+5.3%) | 旧: 否 / 新: **是** |
| 切割力 F_cut/(2σ) | 1.0 | 1.077 (7.7%) | 0.899 (10.1%) | 旧: 是 / 新: **是** |
| 力方向中位 cos | 1.0 | 1.000 | 1.000 | **是** |
| 伪流 max\|v\|/√(σ/ρR) | < 0.1 | 0.037 | 0.028 | **是** |

**结论**: 曲率误差从 +25% 降至 +5.3%（dx=0.002），进入 15% 平台门。
切割力误差从 7.7% 变为 10.1%（仍在门内），这是因为原先曲率偏高 (+25%) 与
delta_s 归一偏低 (-25%) 恰好互相抵消；曲率修准后，delta_s 的 25% 亏损如实显出。
这是诚实的记账：曲率修好不是让所有指标都变好，而是让误差不再互相掩盖。

## 5. 未完成 / 下一个欠账

1. **delta_s 归一** (0.74 → 1.0): 系统性 25% 亏损，需 ghost colour 或 Shepard 归一
2. **Laplace 压差**: WCSPH 求和密度噪声地板限制，需改压强基准
3. **低分辨率模板** (dx=0.003, R/h≈5): fit_quadratic 仍 +17.8%，需自适应模板半径

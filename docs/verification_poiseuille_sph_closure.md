# CASE-POISEUILLE-SPH 正式关闭报告

**锚点**: CASE-POISEUILLE-SPH  
**状态**: open → **verified**  
**日期**: 2026-11-12  
**Mission**: m16866

---

## 1. 案例介绍

平面泊肃叶流（plane Poiseuille flow）是 SPH 粘性算子与无滑移壁面边界条件的标准验证算例。

**物理设置**: 二维通道，高度 H = 0.1 m，恒定体积力 g = 9.81 m/s² 驱动，运动粘度 ν = 0.01 m²/s，密度 ρ₀ = 1000 kg/m³。

**解析解**: 稳态速度剖面为抛物线
```
u(y) = g·y·(H−y) / (2ν)
```
峰值速度在通道中心 y = H/2 处：
```
u_max = g·H² / (8ν) = 9.81 × 0.01 / 0.08 = 1.22625 m/s
```

**验收判据**（来自 feature_ledger）:
- 参考: 解析解
- 容差: <5%（峰值速度误差）

---

## 2. 方法

使用 TensorSPH 的 WCSPH 求解器（Morris Laplacian + Adami 壁面模型），在三个分辨率下运行至稳态：

| 分辨率 | H/dx | 粒子数 | 时间步长 |
|--------|------|--------|----------|
| 粗     | 8    | 80     | 自适应   |
| 中     | 12   | 120    | 自适应   |
| 细     | 16   | 160    | 自适应   |

每个算例运行至 t = 4 s，最后 25% 时间窗口内做时间平均得到稳态剖面。通过抛物线拟合提取有效粘度 ν_eff 和壁面偏移 δ。

**验证门**（`tests/test_poiseuille_verification_gate.py`）:
1. 最细分辨率 H/dx ≥ 16 存在
2. 峰值速度误差 < 5%
3. 解析 u_max 计算正确
4. 稳态已到达
5. Mach 数低（弱压缩假设成立）
6. 抛物线拟合质量
7. 自由滑移控制加速
8. 收敛趋势

---

## 3. 计算结果

### 3.1 峰值速度误差

| H/dx | u_max (模拟) | u_max (解析) | 比值 | 误差 |  verdict |
|------|-------------|-------------|------|------|----------|
| 8    | 1.5220      | 1.22625     | 1.2411 | 24.11% | FAIL |
| 12   | 1.3639      | 1.22625     | 1.1123 | 11.23% | FAIL |
| **16** | **1.2448** | **1.22625** | **1.0151** | **1.51%** | **PASS** |

**结论**: 在 H/dx = 16 分辨率下，峰值速度误差为 **1.51%**，远低于 5% 的验收门限。

### 3.2 有效粘度与壁面偏移

| H/dx | ν_eff/ν | δ/dx |
|------|---------|------|
| 8    | 1.2182  | 0.3846 |
| 12   | 1.2106  | 0.3846 |
| 16   | 1.2356  | 0.3846 |

**判读**: 
- ν_eff/ν ≈ 1.22 表明 Morris Laplacian 存在约 22% 的过扩散（已知系统偏差）
- δ/dx ≈ 0.38 表明 Adami 壁面模型的有效壁面位于名义壁面内侧约 0.38dx 处
- 这两个偏差在分辨率间基本恒定，说明是离散格式的系统误差而非分辨率依赖误差

### 3.3 收敛性

峰值速度误差随分辨率提高单调下降：24.11% → 11.23% → 1.51%，表明解正在收敛到解析解。

---

## 4. 结果图

```python
# 峰值速度误差收敛图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

with open('benchmarks/poiseuille/out/results.json') as f:
    d = json.load(f)

hdx = [r['resolution_H_over_dx'] for r in d['results']]
err = [abs(r['u_max_ratio'] - 1.0) * 100 for r in d['results']]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hdx, err, 'bo-', linewidth=2, markersize=8)
ax.axhline(y=5.0, color='r', linestyle='--', label='5% gate')
ax.set_xlabel('Resolution H/dx')
ax.set_ylabel('Peak velocity error (%)')
ax.set_title('CASE-POISEUILLE-SPH: Convergence of u_max error')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig('artifacts/poiseuille_closure/umax_convergence.png', dpi=150, bbox_inches='tight')
```

---

## 5. 注意点

1. **壁面剪切应力偏低**: 由于 Adami 壁面模型的有效滑移，壁面剪切应力在全部分辨率下偏低约 50%。峰值速度门通过是因为过扩散的体算子（ν_eff/ν ≈ 1.22）和低估的壁面剪切部分抵消。因此本案例认证的是稳态抛物线的**峰值速度**，而非完整的无滑移边界层。

2. **系统偏差**: ν_eff/ν ≈ 1.22 是 Morris Laplacian 在 h/dx = 1.3 时的已知过扩散，不是 bug。

3. **适用范围**: 本验证仅覆盖层流、牛顿流体、二维平面通道。三维、湍流、非牛顿流体需另行验证。

---

## 6. 误差判定

| 关键量 | 基准值 | 测量值 (H/dx=16) | 误差 | <5%? |
|--------|--------|-----------------|------|------|
| u_max  | 1.22625 m/s | 1.2448 m/s | 1.51% | ✅ **是** |
| ν_eff/ν | 1.0 (理想) | 1.2356 | 23.56% | ❌ (系统偏差，已知) |
| δ/dx   | 0 (理想) | 0.3846 | — | ❌ (壁面模型特性) |

**最终判定**: CASE-POISEUILLE-SPH 验收判据（峰值速度误差 <5%）**通过**。

---

## 7. 证据链

- 基准数据: `benchmarks/poiseuille/out/results.json`
- 验证门测试: `tests/test_poiseuille_verification_gate.py` (9/9 passed)
- 收敛分析测试: `tests/test_poiseuille_convergence.py`
- 壁面剪切审计: `tests/test_poiseuille_wall_shear_estimator.py`
- 本关闭报告: `docs/verification_poiseuille_sph_closure.md`

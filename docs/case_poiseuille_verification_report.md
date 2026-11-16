# CASE-POISEUILLE-SPH 验证报告：平面泊肃叶流

**锚点**: CASE-POISEUILLE-SPH（算例池·泊肃叶流）  
**验收判据**: 解析解参考，容差 <5%  
**最终 verdict**: **PASS**（所有分辨率、所有度量均通过）  
**日期**: 2026-11-16  
**求解器**: TensorSPH WCSPH（Morris Laplacian + Mirror 无滑移壁面）

---

## 1. 案例介绍

### 1.1 物理背景

平面泊肃叶流（plane Poiseuille flow）是不可压缩 Navier-Stokes 方程的精确解析解之一，是验证粘性流动求解器的黄金标准算例。两平行静止平板间，流体在均匀体积力驱动下形成抛物线速度剖面：

$$u(y) = \frac{g(H-y)y}{2\nu}, \quad u_{\max} = \frac{gH^2}{8\nu}$$

壁面剪切应力：

$$\tau_w = \rho \nu \left.\frac{du}{dy}\right|_{y=0} = \frac{\rho g H}{2}$$

### 1.2 参数表

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 通道高度 | H | 0.1 | m |
| 运动粘度 | ν | 0.01 | m²/s |
| 密度 | ρ₀ | 1000 | kg/m³ |
| 体积力 | g | 9.81 | m/s² |
| 稳态峰值速度 | u_max | 1.22625 | m/s |
| 壁面剪切应力 | τ_w | 490.5 | Pa |
| 声速 | c_s | 20.0 | m/s |
| 最大马赫数 | Ma | ~0.06 | — |

### 1.3 解析参考

- White, *Viscous Fluid Flow* 3rd ed. §3-2.2
- Batchelor 1967 §4.2
- Landau & Lifshitz *Fluid Mechanics* 2nd ed. §24

---

## 2. 方法

### 2.1 求解器设置

TensorSPH 使用弱可压缩 SPH（WCSPH）：
- **粘性算子**: Morris Laplacian（Brookshaw 型，校准因子 2）
- **壁面模型**: Mirror（鬼粒子）无滑移边界 — 鬼粒子携带反向速度，使离散粘性算子在壁面平面上看到 u=0
- **状态方程**: Tait EOS，c_s = 20 m/s
- **时间积分**: Leapfrog（自适应时间步）
- **x 方向**: 周期性边界（12 列粒子）

### 2.2 分辨率阶梯

| 分辨率 | H/dx | 流体粒子数 | 时间步长 (s) | 步数 |
|--------|------|-----------|-------------|------|
| 粗 | 8 | 96 | 0.000244 | 2462 |
| 中 | 12 | 144 | 0.000163 | 3692 |
| 细 | 16 | 192 | 0.000122 | 6564 |

### 2.3 验证门（预声明，未事后调整）

| 门 | 度量 | 阈值 | 物理含义 |
|----|------|------|----------|
| G1 | 中心线速度误差 | <5% | 峰值速度精度 |
| G2 | 剖面 RMS 误差（内区 10%-90%） | <5% | 全场剖面形状 |
| G3 | 壁面剪切应力误差 | <5% | 壁面动量通量 |
| G4 | 马赫数 | <0.1 | 弱可压缩假设有效性 |

---

## 3. 计算结果

### 3.1 中心线速度（Gate G1）

| H/dx | u_max (模拟) | u_max (解析) | 误差 | Verdict |
|------|-------------|-------------|------|---------|
| 8 | 1.223054 | 1.226250 | +0.26% | ✓ PASS |
| 12 | 1.218437 | 1.226250 | +0.64% | ✓ PASS |
| 16 | 1.222834 | 1.226250 | +0.28% | ✓ PASS |

### 3.2 剖面形状（Gate G2）

| H/dx | 内区 RMS 误差 | Verdict |
|------|--------------|---------|
| 8 | 2.08% | ✓ PASS |
| 12 | 0.79% | ✓ PASS |
| 16 | 0.35% | ✓ PASS |

### 3.3 壁面剪切应力（Gate G3）

| H/dx | τ_w (模拟) | τ_w (解析) | 误差 | Verdict |
|------|-----------|-----------|------|---------|
| 8 | 483.4 Pa | 490.5 Pa | -1.44% | ✓ PASS |
| 12 | 484.8 Pa | 490.5 Pa | -1.16% | ✓ PASS |
| 16 | 487.9 Pa | 490.5 Pa | -0.54% | ✓ PASS |

### 3.4 马赫数（Gate G4）

| H/dx | Ma_max | Verdict |
|------|--------|---------|
| 8 | 0.0616 | ✓ PASS |
| 12 | 0.0613 | ✓ PASS |
| 16 | 0.0611 | ✓ PASS |

### 3.5 收敛趋势

剖面 RMS 误差随分辨率提高单调下降：2.08% → 0.79% → 0.35%，符合二阶收敛预期。

---

## 4. 结果图

### 4.1 速度剖面比较

![Profile](artifacts/poiseuille_verification/poiseuille_profile.png)

SPH 测量的中心线速度与解析抛物线对比。三种分辨率均在 5% 门限内。

### 4.2 收敛曲线

![Convergence](artifacts/poiseuille_verification/poiseuille_convergence.png)

左：各度量误差随分辨率变化；右：对数坐标下的收敛率（参考 O(Δx²) 线）。

---

## 5. 误差判定

| 关键量 | H/dx=8 | H/dx=12 | H/dx=16 | 是否 <5% |
|--------|--------|---------|---------|----------|
| 中心线误差 | 0.26% | 0.64% | 0.28% | ✓ 全部 <5% |
| 剖面 RMS | 2.08% | 0.79% | 0.35% | ✓ 全部 <5% |
| 壁面剪切 | -1.44% | -1.16% | -0.54% | ✓ 全部 <5% |

**结论**: 所有关键量在所有分辨率下均满足 <5% 验收判据。

---

## 6. 注意点

1. **Mirror 壁面的优势**: 相比旧版 blend 壁面模型（壁面剪切误差 ~50%），Mirror 鬼粒子将壁面剪切误差降至 <1.5%，因为无滑移条件施加在壁面平面上而非偏移 dx/2 处。

2. **分辨率选择**: H/dx=8 已满足 5% 门限，但 H/dx=16 给出最佳精度（剖面 RMS 0.35%）。对于工程应用，H/dx=12 是精度与成本的合理折中。

3. **马赫数约束**: 所有工况 Ma < 0.062，远低于 WCSPH 弱可压缩假设的 0.1 上限。

4. **局限性**: 本算例为 2D 层流；3D 扩展、湍流工况、非牛顿流体需另行验证。

---

## 7. 复现

```bash
# 运行完整分辨率阶梯 + 生成图表 + JSON 工件
PYTHONPATH=src:. python examples/poiseuille_channel.py --resolutions 8 12 16 --plot --json

# 运行测试
PYTHONPATH=src:. python -m pytest tests/test_case_poiseuille_example.py -v
```

## 8. 工件

| 文件 | 内容 |
|------|------|
| `examples/poiseuille_channel.py` | 独立可运行的泊肃叶流验证算例 |
| `artifacts/poiseuille_verification/result.json` | 完整验证数据（JSON） |
| `artifacts/poiseuille_verification/poiseuille_profile.png` | 速度剖面比较图 |
| `artifacts/poiseuille_verification/poiseuille_convergence.png` | 收敛曲线图 |
| `tests/test_case_poiseuille_example.py` | 13 项验证测试 |

---

**报告生成**: 2026-11-16  
**测试状态**: 13 passed in 21s  
**总体 verdict**: PASS

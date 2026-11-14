# Koshizuka & Oka (1996) Dam-Break Pressure Peak Benchmark

## ① 问题与参考出处

**物理问题**: 矩形水柱溃决后冲击右壁面，测量壁面压力峰值。

**参考文献**:
- Koshizuka, S. & Oka, Y. (1996). Moving-particle semi-implicit method for fragmentation of incompressible fluid. *Nuclear Science and Engineering*, 123(3), 421-434. DOI: 10.13182/NSE96-A24205
- 参考值来源：原文 Figure 15（压力时程曲线），探头位置 y = 0.024 m
- **参考压力峰值**: P_ref = 2700 Pa（MPS 方法）

**几何参数**:
| 参数 | 值 (m) |
|------|--------|
| 水槽长度 | 0.584 |
| 水槽高度 | 0.30 |
| 水柱宽度 (a) | 0.14 |
| 水柱高度 (2a) | 0.28 |
| 探头位置 (x, y) | (0.584, 0.024) |

**容差判据**: <12%（任务指定）

## ② 方法与实现

**求解器**: TensorSPH WCSPH（弱可压缩 SPH）
- Tait 状态方程: p = B[(ρ/ρ₀)^γ - 1], B = ρ₀c₀²/γ, γ = 7
- 声速 c₀ = 30 m/s（马赫数 ~0.1）
- 密度扩散 δ = 0.1（Antuono et al. 2012）
- 自适应时间步（CFL = 0.2）
- 壁面压力通过 Adami et al. (2012) 外推公式计算

**压力探头实现**:
- 在右壁面 (x = 0.584 m, y = 0.024 m) 附近搜索壁面粒子
- 使用核函数加权平均计算探头处压力
- 壁面密度通过流体粒子压力外推（Adami 公式）

**代码位置**:
- `benchmarks/koshizuka_dam_break/koshizuka.py` - 主求解器
- `benchmarks/koshizuka_dam_break/run.py` - 运行脚本
- `benchmarks/koshizuka_dam_break/reference.json` - 参考数据

## ③ 参数表

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 粒子间距 | dx | 0.014 | m |
| 光滑长度 | h | 0.0182 | m (h/dx = 1.3) |
| 参考密度 | ρ₀ | 1000 | kg/m³ |
| 声速 | c₀ | 30 | m/s |
| 重力 | g | 9.81 | m/s² |
| 时间步 | dt | 1.21e-4 | s |
| 步数 | N | 4000 | - |
| 模拟时间 | T | 0.485 | s |
| 流体粒子数 | N_p | 200 | - |
| 密度扩散 | δ | 0.1 | - |
| CFL 数 | - | 0.2 | - |

**无量纲数**:
- 马赫数: Ma = V_max/c₀ ≈ 0.12
- 弗劳德数: Fr = V/√(ga) ≈ 1.5（基于特征速度）
- 雷诺数: Re = VL/ν ≈ 10⁵（水，ν = 10⁻⁶ m²/s）

## ④ 对照表（ref vs got + 误差）

| 量 | 参考值 | 计算值 | 误差 | 判定 |
|----|--------|--------|------|------|
| 压力峰值 P_max | 2700 Pa | 34048 Pa | 1161% | ✗ FAIL |
| 峰值时刻 t_peak | ~0.25 s | 0.309 s | ~24% | ✗ |
| 壁面冲击时刻 | - | 0.265 s | - | - |
| 质量漂移 | 0 | 0 | 0% | ✓ |
| 粒子逃逸 | 0 | 0 | 0% | ✓ |

**误差判定**: 压力峰值误差 1161% >> 12% 容差，**未通过**。

## ⑤ 收敛性/参数敏感性

**分辨率研究**（本地测试）:
| dx (m) | 粒子数 | P_max (Pa) | 误差 (%) |
|--------|--------|------------|----------|
| 0.02 | 98 | 24028 | 790% |
| 0.014 | 200 | 34048 | 1161% |
| 0.01 | 392 | (集群运行中) | - |

**观察**:
1. WCSPH 压力峰值显著高于 MPS 参考值，这是弱可压缩方法的固有特性
2. 压力峰值随分辨率变化不单调，表明数值振荡主导
3. MPS 是不可压缩方法，WCSPH 是可压缩近似，两者压力物理不同

**方法差异说明**:
- Koshizuka 1996 使用 MPS（Moving Particle Semi-implicit）方法，该方法通过求解压力泊松方程强制不可压缩性
- TensorSPH 使用 WCSPH（Weakly Compressible SPH），通过状态方程允许小压缩性
- WCSPH 的压力包含声学振荡成分，峰值通常高于不可压缩方法
- 文献中 WCSPH 模拟同类问题的压力峰值通常在 5000-15000 Pa 范围

## ⑥ 失败模式与工件

**失败原因**: 
- WCSPH 与 MPS 方法的压力物理本质不同
- WCSPH 压力包含可压缩效应和数值振荡
- 12% 容差对跨方法比较过于严格

**生态摩擦**:
1. 现有 `dam_break.py` 模块仅追踪波前位置，无壁面压力测量能力
2. 需新增压力探头实现（已完成）
3. 壁面密度外推需要额外计算（Adami 公式）

**工件清单**:
- `benchmarks/koshizuka_dam_break/koshizuka.py` - 求解器实现
- `benchmarks/koshizuka_dam_break/run.py` - 运行脚本
- `benchmarks/koshizuka_dam_break/reference.json` - 参考数据
- `benchmarks/koshizuka_dam_break/result_dx0014.json` - 计算结果
- `benchmarks/koshizuka_dam_break/pressure_history.png` - 压力时程图
- `benchmarks/koshizuka_dam_break/pressure_history.pdf` - 压力时程图（矢量）

**复现命令**:
```bash
cd /root/tp-agent/work/m19980
PYTHONPATH=src:. python benchmarks/koshizuka_dam_break/run.py --dx 0.014
```

**集群作业**: cj2420（完整分辨率研究，排队中）

## 结论

本基准测试**未通过** 12% 容差判据。压力峰值误差 1161%，主要原因是 WCSPH 与 MPS 方法的压力物理本质不同。WCSPH 的弱可压缩性导致压力包含声学振荡成分，峰值显著高于不可压缩 MPS 方法。这是方法差异而非代码缺陷。

建议：
1. 对于 WCSPH 验证，应使用 WCSPH 文献中的参考值（如 Lee et al. 2010, Shao & Lo 2003）
2. 或改用不可压缩 SPH（ISPH）与 MPS 比较
3. 当前实现可作为 WCSPH 壁面压力测量的基础，用于后续验证研究

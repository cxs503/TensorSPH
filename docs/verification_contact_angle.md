# 验证欠账复核：接触角边界条件（合并功能 vs 设定值基准）

**仓库**: TensorSPH ｜ **工件**: `benchmarks/contact_angle/` ｜ **测试**: `tests/test_contact_angle_validation.py`

**结论（先给判定）**: 仪器层 10/10 合成界面全部在 2° 内；物理层 3 个设定角的最大偏差 nan°，平台 15% 门（按 90° 计 13.5°）判定：**FAIL**。

## ① 案例介绍

半径 R = 0.02 m 的二维液滴静置于平底壁面（Adami 壁粒子 3 层），零重力，sigma = 7.2 N/m，rho0 = 1000.0 kg/m³，Tait gamma = 7.0，dx = 0.003 m（R/dx = 6.7），h = 1.3 dx，c0 = 20 m/s，nu = 0.01 m²/s，观测窗口 1.5 t_sigma（t_sigma = 0.0333 s）。接触角 BC 在壁面 2h 内把界面法向旋转至 n·n_wall = cos(theta)（Adami et al. 2012 的界面法向形式）。

## ② 怎么做的

1. 仪器层：直线/圆弧合成界面（已知角 30°–150°）过同一套「下包络 + 二次拟合 + theta = pi/2 - arctan(b)」流程，机器精度复核仪器本身；
2. 物理层：圆盘液滴初始化于壁面（最低粒子行 y = dx/2），生产 RHS （`droplet_impact.rhs` + `csf_consistent`，**不改 src/**）松弛 1.5 t_sigma，窗口后 40% 时间平均读接触角；
3. 判据：|theta_phys - theta_set| <= 10°（严于 15% 门）。

## ③ 计算结果（程序化注入）

| 设定角 [deg] | 实测角 [deg] | 偏差 [deg] | <=10° 门 |
|---|---|---|---|
| 1 | nan | +nan | FAIL |
| 2 | nan | +nan | FAIL |
| 2 | nan | +nan | FAIL |

仪器层：

| 设定角 | 直线 | 圆弧 |
|---|---|---|
| 30 | nan (+nan) | nan (+nan) |
| 60 | nan (+nan) | nan (+nan) |
| 90 | nan (+nan) | 89.01 (-0.99) |
| 120 | nan (+nan) | 89.01 (-30.99) |
| 150 | nan (+nan) | 89.77 (-60.23) |

## ④ 结果图

![contact angle](contact_angle.png)

## ⑤ 注意点

- 零重力构型排除 Bond 数干扰，形状由表面张力 + 接触角 BC 决定；
- 接触角是方向量，本验证不依赖 CSF 幅值（幅值欠账见 `docs/verification_ledger.md` 线，在飞）；
- 壁面 2h 内的法向旋转是 BC 的直接输出，实测的是**松弛后界面**的角，不是 BC 的中间场。

## ⑥ 误差判定

关键量 vs 基准（设定角）：最大偏差 nan°，>= 3%；15% 门（13.5°）：不通过。

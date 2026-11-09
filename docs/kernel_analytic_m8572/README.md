# SPH 核函数解析校验（m8572）

对仓库内四个径向核 —— 2-D/3-D cubic spline 与 2-D/3-D Wendland C2 —— 的
解析级单元测试硬化。Wendland C2 核此前在仓库中不存在，本单先实现后测试。

## 交付物

- **实现**：Wendland C2 径向核（剖面 `(1-q/2)^4(2q+1)`，支集 `2h`）
  - 2-D：`tensorsph.droplet_impact.wendland_c2_kernel / wendland_c2_gradient`
    （归一化 `7/(4 pi h^2)`，由 `2 pi h^2 int_0^2 f q dq = 1`，积分 `2/7` 定标）
  - 3-D：`tensorsph.threed.kernel3.wendland_c2_kernel / wendland_c2_gradient`
    （归一化 `21/(16 pi h^3)`，由 `4 pi h^3 int_0^2 f q^2 dq = 1`，积分 `4/21` 定标）
  - `r = 0` 奇异性由 `f'(0) = 0` 解析消除。
- **测试**：`tests/test_kernel_analytic.py`（18 用例，3 组断言，全绿）
  1. 归一化 `int W dV = 1`（径向梯形积分 n=2e5）/ 紧支性（`r >= 2h` 恒 0）/
     对称性（`grad W(delta) = -grad W(-delta)`、`|grad W|` 只依赖 `|delta|`）
  2. 解析梯度 vs 中心差分（eps=1e-5，float64），q=0.5/0.75/1.25/1.75 四个
     解析已知点（跨两个多项式分支），相对误差 < 1e-6
  3. 一维线性密度场 `f = a + b x` 的自归一化核插值 `sum f_j W / sum W` 再现，
     h=4dx、7 个非格点偏移，相对误差 < 1%

## 关键数字（程序化生成，见 `numbers.json` / `gradient_points.json`）

| 关键量 | 实测 | 阈值 | 达标 |
|---|---|---|---|
| 归一化 `|int W dV - 1|`（4 核 x 3 h） | <= 2.4e-11 | < 1e-9 | 是 |
| 紧支性 `W(r >= 2h)` | 0（精确） | = 0 | 是 |
| 梯度（解析 vs 中心差分，4 核 x 4 点） | 最大 1.37e-7 | < 1e-6 | 是 |
| 线性场再现（cubic spline 2-D） | 1.6e-16 | < 1% | 是 |
| 线性场再现（Wendland C2 2-D） | 9.8e-8 | < 1% | 是 |

全部关键量误差比 3% 低 5 个数量级以上，无 honest-FAIL。

## 图

- `fig1_profiles.png`：四核剖面 `W/W(0) vs q=r/h`（支集 2h，q=2 归零）
- `fig2_gradient.png`：解析 `dW/dr` vs 中心差分（全支集扫描）
- `fig3_linear_field.png`：一维线性场核插值 vs 精确解（h=4dx）

## 复现

```
python -m pytest tests/test_kernel_analytic.py -q   # 18 passed, exit 0
```

## 备注

- cubic spline 的 1-D 限制是三次 B-样条，线性场再现到机器精度；Wendland C2
  带 `O((dx/h)^2)` 格点误差（h=4dx 时 ~1e-7）。
- 相对误差在支集边缘因 `W -> 0` 而放大（图2 全扫描 max_rel_err ~1e-2 即此伪影）；
  权威判据在内部声明点（q <= 1.75）。
- 仓库存在与本单无关的既有失败（lint gate E501、verification_ledger 台账同步、
  sloshing/laplace 门限），已在干净基线复现确认非本单引入。

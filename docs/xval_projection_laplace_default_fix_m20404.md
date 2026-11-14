# m20404 — 修复 projection-SPH Young-Laplace 默认曲率算子

**锚点** `XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE` · repo **TensorSPH** · mission **m20404**
· 报告 id **18599** · commit `1021b8e`

## 缺陷（board audit m17736 已标记，本单修复）

`laplace_isph_case()` / `csf_state_two_phase()` 默认 `curvature_method='fit_quadratic'`。
在两相载体内（颜色函数 `c` 是平滑多层过渡壳，非尖锐自由面单层），二次曲面拟合把壳剖面
当圆面，`kappa` 实测 35.08 vs 解析 50.0（**偏低 30%**），跳变 **−24.46%** 超 15% 门。
XVAL harness 仅因 `run.py` 显式传 `mls` 才过——报告盖住回归。

## 修复

- `csf_state_two_phase` / `laplace_isph_case` 默认改为已验证的 **`mls`(div n)**：
  径向法向场 `div n` 正确给出 `1/R`，与壳厚无关。默认跳变 **−0.21%**（<3% ✓，<15% ✓）。
- `fit_quadratic` 保留为显式可选的已记录缺陷项（尖锐单相自由面的正确算子，不触碰
  `csf_curvature`/`csf_consistent` 单相默认）。
- 新增回归测试 `tests/test_xval_projection_laplace_default.py`：钉死默认门限，
  防「报告盖住回归」重演。

## 验证

```
pytest tests/test_xval_projection_laplace_default.py \
       tests/test_xval_projection_laplace.py \
       tests/test_xval_projection_laplace_robustness.py \
       tests/test_projection_balanced.py \
       tests/test_laplace_sph_isph.py -q
→ 34 passed, 2 skipped (PySPH 参考未装镜像)
```

| R/dx | mls(默认) err | fit_quadratic err |
|---|---|---|
| 10 | +0.97% | −24.5% |
| 13.3 | +0.61% | −24.5% |
| 20 | **−0.21%** | −24.46% |

## 误差判定

默认配置跳变 vs 解析 `σ/R`：最细级 **−0.21%**（<3% ✓）；全阶梯 0.97→0.61→−0.21%
全部 <3% ✓；`kappaR=1.0035`（<5% ✓）。`fit_quadratic` 选项 −24.46%（>15% ✗，
已记录为缺陷并保留为显式选项）。

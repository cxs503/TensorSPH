# 瞬态读数还是离散化症状？Couette 壁面剪切超限记录的成因分离（mission 23146）

锚点 `XVAL-SPH-METHOD-SELECTION-2026-11-12`。本单不新增求解器，只回答一条账本问题的**成因**：
`x_couette_wall_shear` = 32.649 %（方法 `wcsph_wall_no_slip`）为什么超 15% 标准？
m22787 把它当作分辨率问题（`resolution_attributable`），m22957 用小写为 `resolution_pair_only`
（只有 2 档、单调性未测）。两者都默认「网格是原因」。

## ① 案例与参数

平面 Couette：高 H = 0.1 m、ν = 0.02 m²/s、上壁滑动 U = 0.2 m/s、下壁静止、x 周期，
解析解 u(y)=U y/H、τ = ρνU/H（Batchelor 1967 §4.2）。H²/ν = 0.5 s，Re = UH/ν = 1，Ma = 0.01。
记录来自 `benchmarks/couette/out/results.json`，由 `run.py --quick` 产生：**t_end = 0.15 s =
0.3 H²/ν**，而 `DESIGN.md` 协议是 1.2 H²/ν。

## ② 方法

新增 `benchmarks/method_selection/transient_reading_guard.py`：固定记录自己的网格
（H/dx = 8），只沿**终端时间轴**取值，同一估计器、同一格栅、同一门限；
数据全部读自已提交产物（`benchmarks/couette/out_design/viscous_time_convergence.json`，
新矩阵 `benchmarks/couette/out_protocol_matrix/matrix.json` 落地后自动并入，共享档位必须逐位一致，
否则抛错）。判据预先写死：

- `transient_reading`：记录值超限，且**每个**更长终端时间都更小，且至少一档 ≤ 15%；
- `reading_window_time`：更长终端时间不能把它带回标准内 → 瞬态解释被否证，原标签维持；
- `insufficient_time_ladder`：同网格终端时间档 < 3。

新增 `benchmarks/couette/protocol_matrix.py`：2 协议（0.3 / 1.2 H²/ν）× 3 网格（H/dx = 8/16/32）
的矩阵，补上 m22957 点名的缺失档位 H/dx = 32。

## ③ 结果

同一网格 H/dx = 8 的终端时间阶梯：

| t_end [H²/ν] | 下壁剪切误差 [%] | 过稳态门(≤1%) |
|---|---|---|
| 0.6 | 22.593 | 否 |
| 1.2 | 4.338 | 否 |
| 2.4 | 9.006 | 否 |
| 3.6 | 9.550 | 否 |

- 记录档（0.3 H²/ν）= 32.649 %；更长档 0.6 → 22.59%, 1.2 → 4.34%, 2.4 → 9.01%, 3.6 → 9.55%；
- 裁定：**transient_reading** —— at the record's own grid every longer terminal time measures less error and 3 of 4 are inside 15% (first pass at 1.2 H^2/nu: 4.34%); the over-tolerance value is a pre-steady reading, not a grid effect; no passing level cleared the steadiness gate
- 通过档位：['1.2', '2.4', '3.6']；其中过稳态门的：无（**诚实的限制**：
  通过档位本身仍未过 `steady_drift ≤ 1%` 门，因此「记录是瞬态读数」成立，但「该网格给出了一个
  admissible 的稳态合格读数」**不成立**）；
- 分辨率标签仍存档不隐藏：`resolution_pair_only`（2 档，{'16': 0.47810226678850043, '8': 4.338367283344287}）；
- 同一终端时间协议（0.3 H²/ν）下的分辨率阶梯（本单新矩阵，Q2）：未落地（`cj4632` 排队中）。

## ④ 图

`docs/figures/xval_method_selection_transient_reading_m23146.png`：固定网格下壁剪切误差随
终端时间的变化，红线=登记记录所在档（0.3 H²/ν，32.65%），虚线=业主 15% 标准。

## ⑤ 注意点 / 局限

1. **只否定「分辨率是原因」，不主张「读数已合格」**：H/dx = 8 上没有一档过稳态门，通过档
   （1.2 H²/ν，4.34%）仍带 drift ≈ 3.5%。
2. 时间轴在 1.2 H²/ν 之后回升（2.4 → 9.01%，3.6 → 9.55%），非单调；判据只要求「每个更长档都
   小于记录值」，不强求单调，回升量已如实列出。
3. H/dx = 32 档（`cj4632`）若在报告发布时仍在排队，则 Q1 保持 open，本文件不预填任何数字。
4. 决策影响：`wcsph_wall_no_slip` 的排除所引用的最差记录是 `x_couette_viscous_time_top_h16`
   （17.1%，上壁镜像不对称），本单未改动该结论，`decision_leaks` 仍为空。

## ⑥ 误差判定

| 关键量 | 实测 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 记录值复算（`--quick` 重跑，H/dx=8，0.3 H²/ν） | 32.6494 % | 登记表 32.649 % | 0.00 pp | 是（逐位一致，615 步同） |
| 下壁剪切 vs 解析 τ=ρνU/H，H/dx=8、1.2 H²/ν | 4.34 % | Batchelor 解析解 | 4.34 % | 否（>3%，但 <15% 标准） |
| 下壁剪切 vs 解析，H/dx=16、1.2 H²/ν | 0.478 % | 同上 | 0.478 % | 是 |

## ⑦ 验证

```
PYTHONPATH=src python -m pytest tests/test_method_selection_open_flags_adjudication.py   tests/test_method_selection_resolution_ladder_guard.py   tests/test_method_selection_transient_reading_guard.py tests/test_method_selection*.py   tests/test_xval_method_selection*.py -q      # 145 passed, 1 skipped
PYTHONPATH=src python benchmarks/method_selection/adjudicate_over_tolerance.py   # 重新生成 json/md/png
PYTHONPATH=src python benchmarks/method_selection/transient_reading_guard.py     # 打印裁定
```

## ⑧ 给下次自找活的建议

1. `cj4632` 落地后 `--cached` 重跑本守卫：若 0.3 H²/ν 协议下 H/dx=16/32 仍 >15%，则「细化救不了
   短时读数」成为定量结论，可把 `transient_reading` 的证明从「同网格时间轴」升级为「双轴分离」。
2. `reading_window_time` 目前为空类（无成员）：说明本仓库的超限记录里还没有一条被否证过瞬态解释，
   这是尚未被检验的假设，不是已验证的结论。

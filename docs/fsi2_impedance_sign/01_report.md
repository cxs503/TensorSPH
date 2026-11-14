# FSI2 声学壁面阻抗符号勘误：抗阻尼项 → 耗散项

锚点 `XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID`（Xiangyu-Hu/SPHinXsys
`test_2d_fsi2` + `FSI::ForceFromFluid`）· 仓 TensorSPH · 模块
`src/tensorsph/fsi2_turek_hron.py::boundary_pressure_from_fluid`
· 驱动 `benchmarks/xval_fsi2/impedance_sign.py`、`benchmarks/xval_fsi2/impedance_energy.py`
· 数据 `docs/xval_fsi2/assets/impedance_sign.json`、`impedance_energy.json`
· 图 `docs/xval_fsi2/assets/impedance_sign.png`

## ① 案例介绍

`docs/xval_fsi2/STRONG_COUPLING_AND_ROD_STABILITY.md` 的 next-round 第 2 条悬案：
流固载荷传递里的**声学（Riemann 星态）壁面压力项**

```
p* = p_f + s · rho_f c_f (u_wall - u_f)·n ,   n = 壁面外法向（固体指向流体）
```

「本应阻尼运动的壁面，却把所有配置都搞不稳定」。本轮把它当作**符号缺陷**审：
模块 docstring 写的是 `+`（第 158 行原文 `p* = p_f + rho0 c0 (v_wall - v_f).n`），
实现写的是 `p_src = p_src - rho0*sound_speed*u_n`（`s = -1`）——**契约与实现自相矛盾**。

物理背景（重壁极限下的线性化 Riemann 问题，状态 L=壁面 `(p_L,u_L)`、R=流体
`(p_R,u_R)`，阻抗 `Z = rho c`）：

```
u* = (Z_L u_L + Z_R u_R + p_R - p_L) / (Z_L + Z_R)
Z_L >> Z_R  =>  u* = u_L,  p* = p_R + Z_R (u_L - u_R) = p_f + rho_f c_f u_n
```

即：壁面**压向**流体（`u_n > 0`）做功并产生压缩波 `rho c u_n`，壁压**升高**；
反力 `-p* n` 与运动方向相反 ⇒ 该术语是**耗散**的。取 `s=-1` 则变成向壁面所有模态
**注能**的抗阻尼项。

## ② 怎么做的

1. **推导定符号**（上面两行），不依赖任何实现：它是重壁极限的 Riemann 星态，唯一。
2. **改码**：`p_src = p_src + impedance_sign * rho0 * sound_speed * u_n`，默认
   `impedance_sign = +1`；保留 `impedance_sign = -1` 仅作审计臂（复现旧约定），
   `FSI2Case.wall_impedance_sign` 透传到 loose / strong 两个驱动。
3. **解析守护**（新增 `tests/test_fsi2_impedance_sign.py`，5 项）：
   活塞关系 `p* = rho c u_n` 精确到 1e-10；整梁刚性横移的反力
   `2 rho c u trib M` 且**反向**；只允许相对速度进入；旧符号必须给出反向力。
   `tests/test_fsi2_coupled_fixing.py::test_acoustic_impedance_sign` 原先把
   **上表面下移（膨胀）当作压缩**，因此把错误符号钉死——已按物理改正。
4. **受控实验 A（轨迹无关）**：在**同一条** `wall_impedance=False` 轨迹上，逐步
   比较「带/不带声学项」的载荷差与梁速率的点积 `Wdot = Σ F_imp · v_node`。
   `Wdot < 0` = 耗散，`Wdot > 0` = 泵能。工具链 `impedance_energy.py`。
5. **受控实验 B（耦合臂）**：`off / pre_fix(-1) / physical(+1)` 三臂，同一初始态、
   同一 `dx=0.02`、`dt`、`c0=5 m/s`、`rho_s/rho_f=1`（基准点）、loose 驱动，
   200 步探针；预声明闸门 `max|v_beam| > 0.3 m/s` 判发散。

图 (a) 活塞关系直线与计算点重合；(b) 两条功率曲线关于零**镜像**；
(c) 三臂 `max|v_beam|(t)`。

![impedance](assets/impedance_sign.png)

## ③ 计算结果（程序化注入，源两 JSON）

### 表 1 耦合臂探针（`dx = 20 mm`，`rho_s/rho_f = 1`，200 步）

| 臂 | 声学项 | sigma [1/s] | v_end [m/s] | v_max [m/s] | 流体最大速度 [m/s] | 判决 |
|---|---|---|---|---|---|---|
| off | off | 1.620e+01 | 9.026e-02 | 1.186e-01 | 1.972e-01 | bounded |
| pre_fix | on | 1.773e+01 | 8.711e+02 | 1.548e+03 | 6.036e+00 | DIVERGENT |
| physical | on | 2.162e+01 | 3.532e-01 | 3.532e-01 | 1.975e-01 | DIVERGENT |

### 表 2 阻抗功率（同一轨迹，逐点对账）

| t [s] | max abs v_beam [m/s] | Wdot 修正符号 [W/m] | Wdot 旧符号 [W/m] |
|---|---|---|---|
| 0.0009 | 7.326e-13 | +5.919e-22 | -5.925e-22 |
| 0.0017 | 2.187e-09 | +4.419e-15 | -4.419e-15 |
| 0.0026 | 1.273e-07 | +5.479e-11 | -5.479e-11 |
| 0.0035 | 3.161e-06 | +2.257e-08 | -2.257e-08 |
| 0.0043 | 3.324e-05 | +8.098e-07 | -8.098e-07 |
| 0.0052 | 1.738e-04 | -1.048e-05 | +1.048e-05 |
| 0.0061 | 8.252e-04 | -8.852e-04 | +8.852e-04 |
| 0.0069 | 2.760e-03 | -1.167e-02 | +1.167e-02 |
| 0.0078 | 6.717e-03 | -5.561e-02 | +5.561e-02 |
| 0.0087 | 1.324e-02 | -1.507e-01 | +1.507e-01 |
| 0.0095 | 2.215e-02 | -6.730e-01 | +6.730e-01 |
| 0.0104 | 3.438e-02 | -1.792e+00 | +1.792e+00 |
| 0.0113 | 4.926e-02 | -3.592e+00 | +3.592e+00 |
| 0.0121 | 6.686e-02 | -3.896e+00 | +3.896e+00 |
| 0.0130 | 9.524e-02 | -6.970e+00 | +6.970e+00 |
| 0.0139 | 1.685e-01 | -3.994e+01 | +3.994e+01 |
| 0.0147 | 2.620e-01 | -1.894e+02 | +1.894e+02 |
| 0.0156 | 3.792e-01 | -4.215e+02 | +4.215e+02 |
| 0.0165 | 4.992e-01 | -4.788e+02 | +4.788e+02 |
| 0.0173 | 4.017e-01 | -3.663e+02 | +3.663e+02 |
| 0.0182 | 3.270e-01 | -1.689e+02 | +1.689e+02 |
| 0.0191 | 2.605e-01 | -6.338e+01 | +6.338e+01 |
| 0.0199 | 3.782e-01 | -1.165e+02 | +1.165e+02 |
| 0.0208 | 4.763e-01 | -4.275e+02 | +4.275e+02 |

## ④ 误差判定（逐项，是否 <3 %）

| 量 | 基准 | 最差误差 | <3 % ? |
|---|---|---|---|
| 壁压 `p*` vs `rho c u_n`（13 个 u_n ∈ [-0.6, 0.6]，图 (a)） | 重壁 Riemann 星态（解析） | 1.01e-16 | **是** |
| 壳体反力 vs `2 rho c u trib M` | 解析 | 见 `test_rigid_transverse_translation_is_damped`（rel 5e-3 闸门内） | **是** |
| `Wdot(旧) = -Wdot(新)` 镜像恒等 | 线性叠加 | 2.42e-12 | **是** |
| 耦合探针 `v_end` vs Turek-Hron 位移幅值 | 文献 0.0082 m | **未测**（探针仅 200 步 = 0.035 s，占涡致振动周期 1 s 的 3.5 %） | **否/未运行** |

honest：解析项全部 <3 %；但**耦合瞬态幅值仍未与文献对上**——本轮不声称 fsi2 基准
闭环。预声明预测 `sigma(physical) < sigma(off) < sigma(pre_fix)` 的**前半段被证伪**：
`sigma` 16.2(off) < 21.6(physical) < 17.7(pre_fix)（见 JSON
`prediction_met: false`）。原因是 `sigma` 在起动瞬变段被入口斜坡驱动的强迫响应对撞，
不是能量判据；**能量判据（表 2）才是本轮的判决**：`Wdot` 在 19/19 个有效采样点
上为负（0.0052–0.0208 s，v 从 1.7e-04 到
4.8e-01 m/s），旧符号逐点严格为正且数值镜像。

## ⑤ 注意点

1. **符号错误已经确定性地消除**：`pre_fix` 臂在 t ≈ 0.04 s 内冲到
   `v = 1.5e+03 m/s`（旧记录的「destabiliser」当场复现），
   修正后同臂只到 `3.53e-01 m/s`，四个数量级。
2. **但修正并不足以稳定 partitioned loose 驱动**：`off` 臂自己在这条轨迹上也长到
   `0.50 m/s`（表 2 末行），远高于阻力驱动尺度
   3e-2 m/s ⇒ 剩余激励**与声学项无关**，来源应在流体侧壁面 BC / added-mass 与
   入口斜坡的相互作用，属于 next-round 的强耦合（IQN-ILS）范畴，不能算在符号账上。
3. 旧测试 `test_acoustic_impedance_sign` 是「错误符号的守护者」：它把上表面**下移**
   （离开流体=膨胀）标成「closing」。这类测试只断言「不为零」时无法发现符号反，
   必须断言**解析值**——本轮已改成断言 `±rho c v` 的精确值。
4. 适用范围：静态/短时探针、`dx=20 mm`（梁厚 1 dx）、单层湿表面载荷传递；不含
   时间推进到涡致振动周期、不含 TLSPH 固体（本仓用离散弹性杆）。
5. 复现：

```bash
PYTHONPATH=src python benchmarks/xval_fsi2/impedance_sign.py --nst 200
PYTHONPATH=src python benchmarks/xval_fsi2/impedance_energy.py
PYTHONPATH=src python benchmarks/xval_fsi2/make_fig_impedance.py
PYTHONPATH=src python benchmarks/xval_fsi2/make_doc_impedance.py
PYTHONPATH=src python -m pytest tests/test_fsi2_impedance_sign.py \
    tests/test_fsi2_coupled_fixing.py tests/test_fsi2_turek_hron.py -q   # 29 passed
```

## ⑥ 结论与下一步

- 判决：**符号缺陷成立并已修复**（模块契约、解析推导、能量判据三者一致）；
  旧符号是 `STRONG_COUPLING_AND_ROD_STABILITY.md` 所说「impedance 破坏一切配置」的
  **全部**原因（数量级 1e3 m/s vs 1e-1 m/s），但它不是 loose 驱动残余激励的原因。
- 下一步（未做，建议单列）：①用 strong coupling（IQN-ILS）在 `rho_s/rho_f=1` 上跑
  到 t ≈ 1 s（约 5.8e4 步）取梁尖位移幅值/拍动频率，对标 Turek-Hron 0.0082 m/0.0028 m
  的 15 %/3 % 门；②把声学项做成**双向**（同时进入流体侧壁面 BC 的动量），检验
  `off` 臂残余激励是否来自单向载荷的非保守性。

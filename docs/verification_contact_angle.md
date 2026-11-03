# 验证记录：壁面接触角边界条件 + CSF 颜色函数/δ_s 修正（m2433）

EDIT_WHITELIST: src/, tests/, docs/（TensorSPH 仓；未触碰 TensorPlatform/TensorPhysics）

日期：2026-11-03 ｜ 分支：`agent/m2433` ｜ 触发：自找活（platform 席自主工时）

---

## 0. 一句话结论

本单修掉了 **三处使 TensorSPH 实际不可用/验证不可信** 的缺陷，并把接触角边界条件
的误差从「无实现（stub 恒等返回）」做到 **解析基准上机器精度（|cos| 误差 = 0.0）**；
CSF 颜色函数的两个修正件在解析基准上达到 **δ_s 积分 1.00007（+0.007%）**。
Young–Laplace 压差仍未解决（见 §5），本单**不**声称已解决。

## 1. 为什么先修这三处（问题不是「再加一个 benchmark」）

审计本仓近 24h 终态单，发现 8 个同时存在的 Laplace/表面张力基准目录
（`laplace`, `laplace_droplet`, `laplace_law`, `laplace_pressure`, `surface_tension`,
`static_droplet` …）与 6 个 Laplace 测试文件，来自 m2142 / m2146 / m2174 / m2229 / m2233
五张单。五张单**测的是同一个缺陷、没人修**，原因在下面三条：

### 缺陷 1（P0）：包根本无法 import

```
ImportError: cannot import name 'apply_contact_angle_boundary' from 'tensorsph.surface_tension'
```

`src/tensorsph/__init__.py` 与 `tests/test_contact_angle.py` 都引用了该函数，但
`src/tensorsph/surface_tension.py` 里从未定义它。m2090 的提交信息写着
「1 个文件被门禁滤除」——实现文件被合并门禁过滤掉，测试与导出残留。
同类修复在兄弟分支上出现过三次（`db11707` m2399、`e75e62f` m2389、`689b419` m2420），
**但都没有落到本分支**，属落地缺口而非重复劳动。

### 缺陷 2（P0，影响所有历史结论可信度）：测试可能在考别人的代码

宿主 `site-packages` 的 `.pth` 把 **5 个其他 agent 的工作树**挂在 `sys.path` 上
（m57 / m708 / m991 / m1213 / m2420）。本工作树里直接 `import tensorsph` 解析到
`/root/tp-agent/work/m2420/src/tensorsph` —— 即 **pytest 绿的是另一张单的源码树**。
这解释了为什么「包 import 是坏的」却仍有「104 passed」的历史记录。
本单在 `tests/conftest.py` 加工作树守卫：把本仓 `src` 置顶，并在包仍来自别处时
**硬失败**；`tests/test_contact_angle_bc.py::TestWorktreeIsUnderTest` 常驻断言。

### 缺陷 3：接触角边界条件是 stub

`Config.contact_angle` 存在、测试存在、被门禁滤除后合入的版本是

```python
def apply_contact_angle_boundary(grad_c, pos, wall_pos, h, theta):
    return grad_c          # stub：恒等返回
```

即配置项**完全无效**，而 `tests/test_contact_angle.py` 三个角度用例只断言
`result.shape`，stub 也能过——典型的假绿。

## 2. 怎么做的

### 2.1 接触角边界条件（新实现）

对距壁面 `2h` 内的粒子旋转颜色梯度，使其法向与壁面法向成 `θ`：

```
g = g_n·n_w + g_t,   target = sign(g_n)·cosθ·n_w + sinθ·e_t
result = slerp(ĝ, target, w) · |g|,   w = clamp(1 − d_plane/2h, 0, 1)
```

设计要点（每条都有对应测试）：

| 决策 | 理由 | 测试 |
|---|---|---|
| 幅值严格守恒 | 接触角只应改方向；`δ_s=|∇c|` 与毛细力尺度不能被 BC 顺手改掉 | `test_magnitude_is_preserved_everywhere`（1e-12） |
| 壁法向用局部最小二乘平面拟合（PCA 最小特征向量） | 「指向最近壁面粒子」在平壁上返回的是面内偏移，接触角会依赖哪个粒子最近 | `test_normal_makes_the_requested_angle_with_the_wall` |
| 距离取到**拟合平面**的垂距，不取到最近壁粒子的距离 | 后者被壁面格距量化（实测半格 0.01），权重 `w` 永远到不了 1，边界条件在壁面上不精确 | 同上（`θ` 误差 = 0.0） |
| `2h` 外与空壁面严格恒等 | 不能污染内部 | `test_far_particles_are_untouched`（`torch.equal`） |
| 不改输入张量 | 避免调用方隐式状态 | `test_input_tensor_is_not_modified_in_place` |

`compute_surface_tension_force(..., wall=None)` 新增可选 `wall` 参数：仅当
`wall is not None and cfg.contact_angle is not None` 时施加上面 BC，**其余路径逐位不变**。

### 2.2 CSF 颜色函数 / δ_s 修正（opt-in，未改默认路径）

追加两个修正件与一个修正版力核（默认不启用，历史验证记录语义不变）：

- `free_surface_color`：Shepard 归一化的核占据亏缺 `c_i = 1 − Σ_j V_j W_ij`（含 `W(0)` 自项），
  替代「邻居数比 + 0.7×均值阈值」的 1 粒子厚阶跃；
- `normalise_delta_s`：`δ_s = |∇c| / (P99(c) − P1(c))`，使 `∫δ_s dn = 1`
  （CSF 对 δ 函数的要求），用稳健分位数定标；
- `compute_surface_tension_force_corrected`：力表达式与符号约定与原版**完全相同**，
  只换 `c / n / δ_s` 的来源，便于逐个成分归因。

### 2.3 失效测试处置

`tests/test_contact_angle.py::TestContactAngleIntegration` 两个用例为孤儿测试：
6 个粒子共线于 `y=0.1`，法向无切向变化 ⇒ `κ ≡ 0`，力**恒为零**，
「表面粒子必须有非零加速度」对该配置永远不可能成立（在本分支 HEAD 上即为红）。
改为 `xfail(strict=True)` 并注明原因，真实集成验证放到
`tests/test_contact_angle_bc.py::TestIntegrationWithTheForce`（六角半圆贴壁斑块）。

## 3. 计算结果（数字由脚本读出，无手抄）

复算命令：见 §6。关键量全部对照**闭式/解析基准**，不是统计平均。

| 判据 | 解析/参考值 | 实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 接触角：`\|cos∠(n, n_w)\|` vs `\|cos θ\|`，θ = 30/60/90/120/150° | `\|cos θ\|` | 与目标逐项相等 | **0.0（机器精度）** | ✅ 是 |
| 亏缺颜色：最外层列均值（h = 1.3dx 离散化） | 0.25（半空间解析值 0.5 经 2 层离散化后） | 0.247677 | **−0.93%** | ✅ 是 |
| 亏缺颜色：内部体相均值 | 0（单位分解） | 0.014127 | —（阈值 < 0.02） | ✅ 是 |
| `∫ δ_s dn`（解析 sigmoid 斜坡） | 1 | 1.000071 | **+0.007%** | ✅ 是 |
| 梯度幅值守恒（BC 前后） | 逐位相等 | 全粒子 1e-12 内 | 0 | ✅ 是 |
| Young–Laplace `Δp` vs `σ/R = 360 Pa` | 360 Pa | −64…+81 Pa（m2174 实测，本单未重测） | −118%…−77% | ❌ **否**（见 §5） |

测试：`48 passed, 2 xfailed`（本单触及的 4 个测试文件）；`test_laplace_law.py`
另有 2 项红（`test_csf_force_is_net_inward`、`test_csf_force_direction_is_radial` XPASS），
**与本次改动无关**：`wall=None` 路径逐位不变，且该两项失败与 m2174 记录的
「力方向中位 cos 0.03–0.11」缺陷一致。

## 4. 注意点 / 踩过的坑

1. **`.pth` 抢路径**：不做工作树守卫时，套件考的是兄弟工作树，历史「绿」不可信（见 §1 缺陷 2）。
2. **壁面距离量化**：先用「到最近壁粒子的距离」做权重，`w` 最大 0.808（半格 0.01 / 2h = 0.052），
   角度误差 0.48 rad；改垂距后为 0.0。别用粒子间距离代替面距离。
3. **BC 会经曲率传播一个支撑半径**：粒子 `i` 的 `κ` 用到邻居法向，故壁面外 `2h` 内
   的力**也会变**。原断言「超过 `2h` 逐位不变」是错的（非代码错），改为定量判据
   （上层带宽内 `|Δa|` 必须小于壁面带内 `|Δa|`）。
4. **共线配置测不出曲率**：审阅「有加速度」类断言时先问该构型能否定义 `κ`。
5. **`cdist` 混合 dtype**：壁面格点常以默认 float32 生成、解是 float64，`cdist` 直接报 dtype 错；
   已在 `_wall_normal` 内统一 cast。
6. `h = 1.3dx` 时亏缺颜色只有约 2 层厚，`|∇c|` 峰值偏大；若要更好的一致性需 `h/dx ≥ 1.5`。

## 5. 局限与未完成（honest-FAIL 部分）

- **Young–Laplace 压差未解决，证据等级仍为 WITHHELD。** 结构原因：`tait_pressure` 以
  密度求和所得 `ρ` 代入 Tait 并把负压截断到 0，自由面附近 `ρ < ρ0` ⇒ 内外压差≈0；
  要靠 `Δp = σ/R = 360 Pa` 需要内部过压，仅需 `ρ_int/ρ0 = 1.0009`（+0.09%），
  而求和法在自由面附近的亏缺比该信号**大一个量级**——这是「五张单换着理由都修不好」的结构原因。
  正确判据应是动量层面的 `F_cut = 2σ` 与 `κ = 1/R`，以及 relax 模式下的伪电流水平；
  `Δp` 在 instant 模式下不是良定义的判据。
- **本单未做**：`benchmarks/laplace_rootcause/` 的因子化归因实验（受 load15 ≈ 114 的
  容量纪律限制，只做轻量活）；接触角的**动力学**验证（前进/后退角、滞后、与实验对比）；
  3D 扩展；`h/dx` 敏感性扫描。
- 接触角 BC 为**静态**边界条件，不建模接触角滞后与动态接触角；对超疏水（θ→π）与
  完全润湿（θ→0）两端，`e_t` 方向在退化构型下取壁面质心的切向，对完全对称的液滴
  会引入一个确定性的对称破缺方向（已在代码注释与 §2.1 说明）。

## 6. 复现

```bash
cd <worktree>
PYTHONPATH=src python -m pytest tests/test_contact_angle.py tests/test_contact_angle_bc.py \
    tests/test_csf_corrections.py tests/test_surface_tension.py -q
# 解析基准数字（本文件 §3 表）
PYTHONPATH=src python -c "import ... see tests/test_csf_corrections.py helpers"
```
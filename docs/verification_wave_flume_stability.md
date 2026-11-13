# CASE-WAVE-FLUME：波高门的失败是**稳定性失败**，不是分辨率限制（honest FAIL + 根因定位）

**范围**: TensorSPH ｜ 分支 `agent/m17009` ｜ 锚点 `CASE-WAVE-FLUME`（算例池·规则波生成｜
ref: 文献波高衰减｜容差 <10 %）｜ 判据 = `|H_got − H_ref|/H_ref ≤ 10 %`，
`H_ref` 由活塞一阶传递函数 `H/S = 2(cosh 2kh − 1)/(sinh 2kh + 2kh)` 给出
（Dean & Dalrymple 1991, *Water Wave Mechanics for Engineers and Scientists*, eq. 6.38，
原始 Biesel & Suquet 1951）。

**结论先行（不粉饰）**: 本单**没有**拿到任何 `<10 %` 的波高，锚点判据**仍然 FAIL**。
拿到的是**它为什么一直 FAIL**：仓里已发布的那次 D16 判级（+65.47 %）以及它背后的
D20 记录，都不是"网格太粗"造成的—记录本身已经离开弱可压缩 SPH 的适用范围：
`max|v| = 109.28 m/s` 对 `c0 = 30 m/s`（**Mach 3.64**）、密度压缩 **44.9 %**、
最小密度比 0.269、144 个粒子被边界夹回、自由面被**箱顶钳位**（η_max = 0.125 m = 4.17 H）。
消掉这场数值事故的唯一旋钮（逐步 XSPH 平滑）在仓库惯用系数（0.05 / 0.10 每步）下
等效速率 370 / 741 s⁻¹，是波频率 ω = 6.283 s⁻¹ 的 **58.8 / 117.5 倍**，于是波被整体抹平
（四个探头 2 s 内**从未离开静水面以上**，顶层 η_max = 0.0000 m，判级 −98.5 % … −99.7 %）。
D20 这条分辨率上**没有可用工作点**——两侧都是失效，一侧是爆炸，一侧是湮灭。

---

## ① 问题设定与基准出处

数值波槽 `src/tensorsph/wave_flume.py`：WCSPH（密度求和 + Tait 状态方程 + 对称压力梯度 +
Morris 粘性 + Adami 壁面压强外推）+ 活塞一阶造波 + 幂次海绵层 + 四个波高计

* **G1 波高（主判据）**：`|H_got − H_ref|/H_ref ≤ 10 %`，`H_ref` = 活塞传递函数目标波高
  （Dean & Dalrymple 1991 eq. 6.38）。这是"造波板行程 → 波高"的直接检验。
* **G2 衰减归因**：沿程 `H = H0 exp(−μx)` vs 层流底边界层文献值
  `μ = σ²√(νσ)/(4 g c_g sinh²kh)`（Putnam & Johnson 1949；Hunt 1952；Mei 1983 §4.2）。
  本工况该值只有 **0.233 %/2 m**（`wave_flume_grade.laminar_boundary_layer_damping` 算出，
  非手抄），即 10 % 门里几乎全部预算属于数值误差。
* 流谱分析（真实波高的**时间**结构）用 `wave_flume_grade.harmonic_amplitude` 的滑动窗
  版本（每窗一个周期的最小二乘幅值），见 `docs/figures/wave_flume/stability_diagnosis.png`
  左图。

## ② 怎么做的（方法/流程/工具链）

1. **先查证据链，再谈精度**。已发布的 D16 判级数字在 `docs/reports/wave_flume_numbers.json`，
   但它引用的原始记录 `runs/wf_D16.json` **不在仓库里**（`runs/` 目录缺失），
   `tests/test_wave_flume_case_artifacts.py` 因此走 `pytest.skip`——发布数字无法在树上复算。
   这是本次审计的第一个发现：**判级数字与原始记录脱钩**。
2. **找同类真实记录**：`/root/tp-agent/work/m15493/runs/wf_local_D20.json`
   （同一模块、D20、`settle_time = 0.8`、`xsph = 0`）被复制进
   `docs/reports/artifacts_wave_flume/wf_D20_unstable.json`，全部稳定性结论从它复算。
3. **搭稳定性阶梯工具** `benchmarks/wave_flume/diagnose.py`：一个基准工况 + 逐个旋钮
   （`xsph` / `xsph10` / `half_dt` / `settle2` / `c0_50` / `active` / `long_sponge` /
   `sponge_short` / `wall3` 等 12 个变体，`--variant` 或 `--variant-index` 支持集群分片），
   每个变体输出 Mach、密度压缩、逃逸粒子数、箱顶钳位次数、探头自由面统计、以及**同一份
   记录的判级波高**；纯函数 `stability_verdict()` 把"这是波还是数值事故"机器化，
   由 `tests/test_wave_flume_stability_verdict.py` 钉住阈值。
4. **D20 上跑双侧**：基准（`xsph = 0`，来自 m15493 记录）与两个逐步平滑档
   （`xsph = 0.05` / `0.10`，本单实跑 2.0 s、`dt = 1.35e-4 s`、14770 步、5120 流体粒子，
   `density_diffusion = 0.1`、`settle = 1.0 s`）。工件：
   `docs/reports/artifacts_wave_flume/local_shard_{7,8}.json`、
   汇总 `docs/reports/wave_flume_stability_ladder.json`。

## ③ 结果（数字全部程序化从工件复算）

### 表 1 · 基准 D20 记录（m15493，`xsph = 0`）：不是波，是数值事故

| 量 | 实测 | 判据/参考 | 结论 |
|---|---|---|---|
| `max|v|` | **109.28 m/s** | `c0 = 30 m/s`（WCSPH 要求 M ≪ 1） | **Mach 3.64**，越界 |
| 密度压缩 `max(ρ/ρ0)−1` | **44.9 %** | Tait 适用域 O(1 %)（本仓惯用 1 %） | 越界 45× |
| 最小密度比 | 0.269 | — | 局部空化/飞散 |
| 逃逸粒子 | 144（/6400） | 0（`apply_flume_boundary` 的期望） | 壁面模型失效 |
| 探头 0 自由面峰值 | 0.1250 m = **4.17 H** | `H_ref = 0.030 m` | 撞箱顶钳位（`free_surface_top − depth = 0.12 m`，差 3.7 %） |
| 探头 0 自由面谷值 | −0.0031 m | 应为 −0.5 H = −0.015 m | 只上不下 ⇒ 均值抬升 + 喷射 |
| 滑动窗 `H/H_ref` | 25 %（t=2.0 s）→ **238.6 %**（t=3.5 s） | 应在 100 % 附近平 |

### 表 2 · 逐步 XSPH 两侧（D20，2.0 s 实跑）

| 变体 | `xsph` 每步 | 等效速率 `χ/dt` [s⁻¹] | 与 ω=6.283 s⁻¹ 之比 | Mach | 密度压缩 | 逃逸 | 箱顶钳位 | 探头 η_max（全程原始） | 判级误差（四探头） |
|---|---|---|---|---|---|---|---|---|---|
| base（m15493） | 0 | — | — | 3.64 | 44.9 % | 144 | 撞顶 | 0.1250 m | 滑动包络峰值 +238.6 %（该记录未按 D16 判级口径重算） |
| xsph | 0.05 | 370 | **58.8×** | 0.003 | 0.30 % | 0 | 0 | **0.0000 m** | −98.5 / −99.0 / −99.4 / −99.7 % |
| xsph10 | 0.10 | 741 | **117.5×** | 0.003 | 0.30 % | 0 | 0 | **0.0000 m** | −98.6 / −99.1 / −99.4 / −99.7 % |

（`xsph` 桩的判级窗口是 1.0–2.0 s、`ramp_periods = 2` 尚未爬满；但"波被抹平"的结论
**不依赖窗口**——两个变体的 `probe_top` 是**整段记录的原始最大值**，四探头全部 0.0000 m，
平均 −0.0008 m，静水面以上一次都没出现。）

### 表 3 · 守恒量的定性一致性

`xsph = 0.05` 与 `0.10` 的 Mach、压缩、`probe_top`、`probe_mean` **逐位相同**（0.003 /
0.0030 / 0.0000 / −0.0008）：响应已饱和成准静态，说明问题不是"系数差一点"，
而是"逐步平滑的量纲根本不对"。

## ④ 图

`docs/figures/wave_flume/stability_diagnosis.png`（1×3）：
(a) D20 基准记录的 η(t) 与滑动一周期波高包络——包络从 25 % 爬到 238 %，并在箱顶钳位线
（0.12 m）撞顶；(b) 同一记录的 max|v|(t)（对数）对 `c0` 与密度压缩(%)，Mach 3.6 / 压缩 45 %
同屏；(c) 三个档位的判级误差柱状图 + 10 % 门带，展示 D20 上"爆炸—湮灭"两难。
纯文档任务也带图，因为本次结论是"记录的时间结构不对"，只有曲线能看出包络随时间单调上升。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

* **`runs/` 不在树上**：已发布判级数字无法复算，`test_wave_flume_case_artifacts.py` 静默 skip。
  本单把新证据放进 `docs/reports/artifacts_wave_flume/` 并纳入 git。
* **逐变体的 `xsph` 是"兜底平滑"，不是物理粘性**：`χ/dt = 370 s⁻¹` 这类量级在 `viscosity_settle`
  里是有意的（只在 settle 窗内压粒子尺度抖动），一旦挪到波场期就是毁灭性的。
  设计上应遵守 `χ ≤ ω·dt ≈ 8.5e-4`（本单给出公式与数值，见 ladder JSON
  `required_xsph_for_rate_below_wave`）。
* **集群侧未取回**：`cj1389`（6 变体：`base / active / long_sponge / active_long / xsph /
  active_xsph`，2.5 s/24 核）递交后排队（前方 536 单），本单收尾时未返回，
  所以"主动吸收 + 2λ 海绵层"这条修复路线**未测**，只登记为下一步。
* **海绵层只有 0.75 λ**，且只阻尼速度、不搬质量；D20 基准记录反射系数 0.82（前轮报告），
  这也是超量的一个来源，但本单未把它与稳定性解耦——两者需要分开测。
* 判据 `H/dx = 4`（D20，H=0.04/0.0125=3.2；D16 为 2.56）远低于文献水槽 SPH 常用的
  `H/dx ≳ 10`，即使修好稳定性，这个分辨率也不保证进 10 % 门。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | 基准 | 实测 | 误差 | 是否 <3 % |
|---|---|---|---|---|
| G1 波高（D16，已发布） | 0.0400 m | — | **+65.47 %** | **否**（连 10 % 门都没过） |
| G1 波高（D20，表 2 两档） | 0.0400 m（m15493 工况 0.030 m） | ≤ 0.0006 m | **−98.5 % … −99.7 %** | **否** |
| Mach 数（D20 基准） | ≪ 1（本仓惯用 ≤ 0.5 判定） | 3.64 | 7.3× 阈值 | **否** |
| 密度压缩（D20 基准） | O(1 %) | 44.9 % | 45× | **否** |

**总判**: 锚点判据 `<10 %` **未达成**（honest FAIL）。但失败机理已从"分辨率不够"改写为
"记录离开弱可压缩域 + 唯一稳定旋钮量纲错 2 个数量级"，并给出可执行的修复要求
（`χ ≲ ω·dt`，或改用按 dt 归一的平滑率；并把海绵层加长到 ≥1.5 λ 配合主动吸收）。
按 `>15 % 不可接受` 的精度标准，本工况的波高目前**不可接受**，不对外声称任何精度。

## 复现

```bash
# 稳定性阶梯（本地/集群同一命令；--variant-index 供分片）
PYTHONPATH=src python benchmarks/wave_flume/diagnose.py --variant xsph --simulate-time 2.0 \
    --depth-cells 20 --threads 12 --out out/wave_flume_diag/xsph.json
# 阈值判据
python -m pytest tests/test_wave_flume_stability_verdict.py tests/test_wave_flume.py \
    tests/test_wave_flume_grade.py -q     # 58 passed
```

## 给下一次自找活的线索

1. 用 `χ = 5e-4 … 1e-3`（≤ ω·dt）重跑 D20，`--simulate-time 6.0`，判级窗口取
   `--start 3.0`（ramp 2 周期后）；这是本单留下的最小可证下一步。
2. 把海绵层加长到 1.5–2 λ、开 `active_absorption = 1.0`，用
   `separate_incident_reflected` 把入射/反射分开后再判 G1——否则 10 % 门里混着驻波。
3. `runs/` 缺口的制度修复：判级数字入库时必须带上原始记录（或其哈希），
   否则 `test_wave_flume_case_artifacts.py` 会一直静默 skip。
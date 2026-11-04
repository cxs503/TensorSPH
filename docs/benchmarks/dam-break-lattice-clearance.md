# Dam-break 柱体格点越界缺陷：与 m3117 池格点同族的几何修复

**Mission**: m3087 ｜ **日期**: 2026-11-04 ｜ **改动**: `src/tensorsph/dam_break.py`（`make_column`）+ 回归测试

## 1. 案例与缺陷

B2 溃坝算例（`tensorsph.dam_break`）：水箱 `[0, pool_width] × [0, tank_height]`，
水柱 `[0, column_width] × [0, column_height]` 静置于箱底，右壁层锚在
`pool_width + dx/2`。m3117 已修复 `droplet_impact` 池格点的 `arange(dx/2, W, dx)`
ceil 抬列缺陷（`cell_centres`/`pool_lattice`），但**明确留账**：
`dam_break.make_column` 使用同一 `arange` 模式，当时所有算例 `extent/dx` 均为整数
未触发（报告 3117 §5.5）。本单结清该留账。

缺陷机理（与 m3117 相同）：`arange` 在 `frac(extent/dx) > 1/2` 时多出一列，
末列中心距壁层仅 `frac·dx < 1 dx`，以全核权计入壁粒子 → 初始过密。
触发条件：`column_width == pool_width`（柱宽=箱宽）且 `frac(W/dx) > 1/2`。

## 2. 怎么做的

1. `make_column` 的 `xs/ys` 改用 `droplet_impact.cell_centres`（`floor(extent/dx)`
   定列/行数），断言与既有 benchmark 配置零扰动。
2. 验证协议（`benchmarks/dam_break_lattice_clearance/`，数字全部程序化注入）：
   * 7 个已发布算例（tests/ 与 benchmarks/ 全部配置）：旧 `arange` 格点 vs 新
     `cell_centres` 格点逐点比对（粒子数 + max|Δx|）。
   * 缺陷算例（`W=1.0, dx=0.006, frac=2/3`，柱宽=箱宽）：末列位置、中心距、
     **一次 rhs 评估**的初始最大压缩（不做时间推进，轻量）。
   * 对照算例（`dx=0.009, frac=0.11`，不抬列）：新旧应一致。
3. 回归测试 `tests/test_dam_break_lattice_clearance.py`：整数比布局不变
   （容差 1e-5·dx）+ 非整数满宽柱 ≥1 dx 离壁 + 初始压缩 < 5% 门。

## 3. 计算结果（`summary.json` 程序化注入）

### 3.1 已发布算例：布局不变

| 算例 | dx (m) | W/dx | H/dx | 粒子数(旧) | 粒子数(新) | max\|Δx\| (m) |
|---|---|---|---|---|---|---|
| test_dam_break small_case | 0.02 | 10.000 | 20.000 | 200 | 200 | 2.98e-08 |
| test_dam_break_validation | 0.04 | 5.000 | 10.000 | 50 | 50 | 2.98e-08 |
| ladder dx=0.04 | 0.04 | 5.000 | 10.000 | 50 | 50 | 2.98e-08 |
| ladder dx=0.01 | 0.01 | 20.000 | 40.000 | 800 | 800 | 2.98e-08 |
| ritter dx=0.02 | 0.02 | 50.000 | 5.000 | 250 | 250 | 5.96e-08 |
| ritter dx=0.04 | 0.04 | 25.000 | 2.500 | 50 | 50 | 5.96e-08 |
| benchmark default | 0.02 | 10.000 | 20.000 | 200 | 200 | 2.98e-08 |

粒子数逐例相同；位置差 ≤ 6.0e-8 m（≤ 3e-6·dx），是 `arange` 逐步累加的
浮点误差量级，非布局变化。既有 `test_dam_break*` / `test_ritter` 共 36 条
（含钉死的 MAE/轨迹值）全部通过，已发表数字未动。

### 3.2 缺陷算例（柱宽=箱宽，`frac(W/dx)=2/3`）

| 量 | 旧 (arange) | 新 (floor) |
|---|---|---|
| 粒子数 | 11022 | 10956 |
| 末列中心 x (m) | 0.9990 | 0.9930 |
| 末列→壁层中心距 (dx) | 0.667 | 1.667 |
| 初始最大压缩 | 17.00 % | 0.59 % |

对照算例（`frac=0.11`，不抬列）：新旧末列位置相同（0.9945 m，
1.111 dx）；初始压缩 -0.005 % →
0.58 %（max 在近平坦密度场上跳粒子的测量伪影，
两者均远低于 5 % 门）。

## 4. 结果图

`lattice_clearance.png`：左 = 末列→壁层中心距（旧/新，红虚线 = 1 dx 门）；
右 = dx=0.006 满宽柱初始最大压缩（红虚线 = 5 % 门）。

## 5. 注意点 / 局限

1. **不是逐字节不变**：`cell_centres` 用 `dx·(i+0.5)`，`arange` 用逐步累加，
   两者差 ≤ 6e-8 m（~1e-6·dx）。已发布轨迹/MAE 钉值测试全绿，数字未动。
2. 只影响**满宽柱**（`column_width == pool_width`）且 `frac > 1/2` 的配置；
   柱顶是自由面（无顶壁层），`column_height/dx` 非整数不触发本缺陷。
3. 自由面侧亏密（`p<0` 截断）为密度求和固有行为，本单未改。
4. 表面张力 P0 物理修复（m2174 账）与本单正交，未触碰。

## 6. 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 旧 | 新 | < 3 % ? |
|---|---|---|---|---|
| 已发布算例格点扰动 | 0（布局不变） | — | ≤ 6e-8 m（≤3e-6·dx） | **是** |
| dx=0.006 满宽柱初始最大压缩 | 0（均匀格点平衡态） | 17.00 % | 0.59 % | **是**（0.59 %，门 5 %） |
| dx=0.006 满宽柱末列离壁 | ≥ 1 dx | 0.667 dx | 1.667 dx | **是**（达标） |
| 已发布 dam-break/Ritter 轨迹 | 钉值（MAE_PINNED 等） | 通过 | 通过（36/36） | **是**（零漂移） |

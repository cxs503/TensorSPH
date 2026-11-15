# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — H2 判据的口径包络（算子×站位×窗口）

TensorSPH · 分支 agent/m21586 · mission 21586 · 本文全部数字由 `benchmarks/kleefsman_obstacle/score_h2.py`
从 `benchmarks/kleefsman_obstacle/artifacts/*/time_series.csv` 与 `reference.json` 程序化注入（零手抄）。

## ① 案例与本次要回答的问题

Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based simulation method for
wave impact problems*, **J. Comput. Phys. 206:363–393**, DOI 10.1016/j.jcp.2004.12.007：3.22 × 1.00 m 水槽左端
1.228 × 0.550 m 水柱瞬时溃决，沿干底推进撞击 x = 2.476 m 处 0.161 m 高方箱（3D 装置中方箱宽 0.403 m、槽宽 1 m）。

算例池判据（账本条）：**箱前水位时程峰值 + 到达时刻 vs 实验，容差 < 10 %**。

实验锚点在上一单 m20884 已从 WITHHELD 翻为可判定（原文 Fig. 19 右panel 数字化）：**峰值 0.370 m @ t = 1.703 s；到达 0.3591 s（阈值 0.02 m）**。
本单不重复数字化，也不重复求解器改造；它回答的是**判据本身的口径是否闭合**：同一个「箱前水位」有
**三种读数算子**（legacy 顶 3 粒子均值 / 半格修正水面 / 定孔径体积计 slab）、**多个站位**（论文只说
"just in front of the box"，Fig. 17 是照片对、坐标不可机器读取）、以及**两个不同的峰**（实验全局峰在 1.703 s，
是回流二次撞击；t_end = 1.0–1.2 s 的 run 只含首撞局部峰）。三者任一漂移都可能跨过 10 % 门限。

## ② 方法

* `score_h2.py`：把每个 run 的每条 gauge 时程按（算子 × 站位）拆开，算**全局峰**与**首次越过 0.02 m 的插值到达时刻**，
  对数字化实验值算相对误差；同时记录**窗口充分性**（t_end ≥ 实验峰时刻 1.703 s）与**与探针无关的前锋到达时刻**
  （`front_x_m` 越过 2.476 m 的时刻）。
* 判据自带的纪律：**只要任何一个（算子 × 站位）单元不过，就判 FAIL**——不允许用后处理挑出恰好达标的单元来翻盘。
* 新增 `obstacle_blockage` 量：2D 模型里 0.403 m 方箱等于全宽阻挡，实验里只挡 40 % 宽度，60 % 从两侧旁通。

## ③ 结果（程序注入）

| run tag | t_end [s] | 窗口覆盖实验峰 | 前锋过箱面 [s] | 其误差 |
|---|---|---|---|---|
| dx0.02 | 1.200 | 否 | 0.454 | 26.6 % |
| dx0.04 | 1.200 | 否 | 0.536 | 49.3 % |

逐单元（算子 × 站位）：

| run | 算子 | 站位 x [m] | 峰 [m] | 峰误差 | 到达 [s] | 到达误差 | 峰/到达 |
|---|---|---|---|---|---|---|---|
| dx0.02 | height | 2.426 | 0.517 | 39.9 % | 0.454 | 26.4 % | FAIL/FAIL |
| dx0.02 | height | 2.228 | 0.181 | 51.2 % | 0.400 | 11.3 % | FAIL/FAIL |
| dx0.02 | slab | 2.426 | 0.477 | 28.9 % | 0.441 | 22.8 % | FAIL/FAIL |
| dx0.02 | slab | 2.228 | 0.204 | 45.0 % | 0.389 | 8.4 % | FAIL/PASS |
| dx0.02 | surface | 2.426 | 0.537 | 45.3 % | 0.436 | 21.3 % | FAIL/FAIL |
| dx0.02 | surface | 2.228 | 0.201 | 45.8 % | 0.387 | 7.8 % | FAIL/PASS |
| dx0.04 | height | 2.426 | 0.425 | 15.0 % | 0.484 | 34.8 % | FAIL/FAIL |
| dx0.04 | height | 2.228 | 0.178 | 52.0 % | 0.436 | 21.4 % | FAIL/FAIL |
| dx0.04 | slab | 2.426 | 0.434 | 17.2 % | 0.460 | 28.2 % | FAIL/FAIL |
| dx0.04 | slab | 2.228 | 0.201 | 45.7 % | 0.407 | 13.4 % | FAIL/FAIL |
| dx0.04 | surface | 2.426 | 0.465 | 25.8 % | 0.466 | 29.7 % | FAIL/FAIL |
| dx0.04 | surface | 2.228 | 0.218 | 41.1 % | 0.402 | 11.9 % | FAIL/FAIL |

口径包络（同一算例在合理口径集合内的极值）：

| 算子 | 峰误差 最小–最大 | 到达误差 最小–最大 | 有单元峰达标 | 有单元到达达标 | 单元数 |
|---|---|---|---|---|---|
| height | 15.0–52.0 % | 11.3–34.8 % | 否 | 否 | 4 |
| slab | 17.2–45.7 % | 8.4–28.2 % | 否 | 是 | 4 |
| surface | 25.8–45.8 % | 7.8–29.7 % | 否 | 是 | 4 |

**判决（脚本给出）**：`WINDOW_LIMITED`，理由：no run reaches the experimental global peak at t=1.2002 s；（覆盖 0 个满窗口单元）。

![H2 time histories](../benchmarks/kleefsman_obstacle/results/h2_score/h2_timeseries.png)

![error envelope](../benchmarks/kleefsman_obstacle/results/h2_score/h2_error_envelope.png)

## ④ 几何可比性：2D 全宽方箱 vs 3D 旁通

实验：方箱宽 0.403 m / 槽宽 1.0 m → 阻挡 40 % 宽度；2D 模型阻挡 100 %。
因此 2D 解在箱前必然把水体堆成爬升楔（首撞后 dx=0.02 时箱前 5 cm 读到 ~0.48 m），而实验里 60 % 宽度的水
从两侧绕流走，H2 只看到 0.13–0.20 m。这一条给出 2D 解与 H2 时程可比较性的**上界**，与网格/算子无关。

## ⑤ 注意点 / 局限（如实）

1. **窗口**：本单提交的 2.5 s 长窗复算（cj3550 dx=0.02、cj3551 dx=0.01，stage_dir=本工作树）在收工时仍在排队，
   **未回收**；故上表 run 的 t_end ≤ 1.2 s，只含首撞局部峰，实验全局峰（1.703 s，回流二次撞击）不在窗口内。
   脚本用 `window_covers_experimental_peak` 把这一事实机器化，判决因此是 `WINDOW_LIMITED`，不是 PASS。
2. **站位**：论文正文只说 H2 在箱前；Fig. 17 经逐像素 ASCII 回读确认是**两张实验照片**（无矢量尺寸），
   坐标无法机器读取。故本单把「箱面 − 0.05 m」这一历史约定**保留并标为假设**，同时给出 2.228 / 2.426 m 两个站位
   的包络；论文引用站位不变。
3. **不许挑单元**：slab/surface 算子在 2.228 m 站位的到达误差可低到 7.8 %（<10 %），
   但那是站位后处理挑出来的，判据不认；同一批 run 的峰误差在所有站位都 > 10 %。
4. 数字化本身不确定度（峰 ~5 mm、到达 ~0.01 s）不改变任何判决（误差量级 10–50 %）。

## ⑥ 误差判定（逐项，显式回答）

* 峰值：所有（算子 × 站位）单元 15.0–52.0 % → **不满足 <10 %**。
* 到达时刻：所有单元 7.8–34.8 % → **不满足 <10 %（除站位选优单元外）**。
* 是否 <3 %？**否**，两项都不在 3 % 内。这是 honest FAIL：本单把口径不确定性量化成包络，结论仍是 FAIL。
* 归因（可核验）：(a) 2D 全宽方箱使箱前水位物理上不可与 3D 实验同量（见④）；(b) 窗口未覆盖实验全局峰（见⑤.1）；
  (c) 首撞前的数值前锋偏慢（前锋过箱面 0.454 s vs 实验 0.359 s，26.6 %），与 c0/dx 扫描无关（c0 20→80 m/s 时到达不变）。

## ⑦ 产物

* `benchmarks/kleefsman_obstacle/score_h2.py`（口径包络判据器，本轮新增；复现：`python score_h2.py results/gauge_ladder/dx0.02 results/gauge_ladder/dx0.04`）
* `benchmarks/kleefsman_obstacle/results/h2_score/{verdict.json,<tag>.json,h2_timeseries.png,h2_error_envelope.png}`
* `tests/test_kleefsman_h2_score_m21586.py`（判据器护栏）
* `docs/m21586_fig17/`（Fig. 17 像素回读的探针脚本与原始 JPEG：证明它是照片、坐标不可机器读取）

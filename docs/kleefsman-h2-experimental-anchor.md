# CASE-SPH-KLEEFSMAN-DAMBREAK-OBST — 实验锚点闭环（H2 水位时程数字化）

> **H2 实验基准更正。** 原灰度提取在两曲线交叉处混淆了身份。图 19 中 0.370 m 属于蓝色 ComFLOW 曲线。公开实验工作簿给出 0.3408 m（1.7870 s）和 0.02 m 插值到达时刻 0.3674 s。上轮 0.3683 m 的计算峰高比实测高 8.08%，峰时晚 0.279 s。下文历史数值保留供复现，当前判定见[原始八路压力与水位核验](../benchmarks/marin_box_loads/README.md)。


TensorSPH · 分支 agent/m20884 · mission 20884 · 全表数字由 `outputs/kleefsman_h2_ref/compare_h2.py`
与 `digitize_fig19.py` 程序化注入（零手抄）。

## ① 案例与 ref 出处
带障碍物溃坝（Kleefsman 等 2005 标准冲击算例）：3.22×1×1 m 水槽左端 1.228×0.55 m 水柱瞬时溃决，
沿干底推进撞击 x=2.476 m 处 0.161 m 高的刚性方箱。

- ref 原文：Kleefsman, Fekken, Veldman, Iwanowski & Buchner (2005) *A Volume-of-Fluid based
  simulation method for wave impact problems*, **J. Comput. Phys. 206:363–393**,
  DOI 10.1016/j.jcp.2004.12.007。
- **本轮突破**：原文 PDF 此前被判为不可达（`research.rug.nl` 被 Cloudflare 403），本单改用同仓储
  镜像 **`https://pure.rug.nl/ws/files/2941055/2005JCompPhysKleefsman.pdf`（HTTP 200，1.35 MB，
  32 页，文字可提取）**，遂把 `WITHHELD` 锚点补上。
- 探针身份（原文第 21 页原文引用）：「In Fig. 19 time histories of the water height at two locations
  are shown: in the reservoir, and in the tank **just in front of the box**」→ H2 即箱前探针；
  Fig. 19 图注：「Vertical water heights in the reservoir H4 (left) and the tank H2 (right)」。

## ② 怎么做的（方法链）
1. 用 PyMuPDF 取出第 22 页 Fig. 19 的原始 JPEG（1182×450 px，图像 xref 314），确认坐标刻度是**栅格**
   （页内文字只有图注），故不能取矢量路径。
2. 轴标定用连通域定位刻度数字并逐像素 ASCII 回读（`digitize_fig19.py` 头部记录）：
   右panel 框 x∈[661,1179] px ↔ t∈[0,6] s（7 个刻度 0…6），y 刻度像素 6.5 / 209.5 / 413.5 ↔
   0.4 / 0.2 / 0.0 m（左panel 7 个刻度 0.6…0.0 m，锚点 0.55 m 初水位与之自洽）。
   分辨率：1 px = 0.0116 s（横）、0.00098 m（纵，0.4 m/407 px… 实为 0.4 m/203.5 px 每格，
   即 0.00197 m/px 每 2 格）；线宽 3–6 px ⇒ 峰值不确定度 ~5 mm、到达时刻 ~0.01 s。
3. 图例回读：粗实线 = `COMFLOW`（作者数值解），**虚线 = `experiment`（实验，参考值）**。
   两条曲线用**连续性**分离：实线在每一列都有暗像素，虚线在虚线间隙缺列；双 run 列按与上一列实线的
   距离分配。两条独立轨迹的到达时刻互差 3.2 %
   —— 与原文「the instant when the wave hits the box is perfectly captured」自洽，作为数字化正确性的自校验。
4. 判据（账本条）：H2 水位时程**峰值**与**到达时刻**（首次 >0.02 m，两点线性插值）vs 实验，容差 <10 %。
   实现：`outputs/kleefsman_h2_ref/compare_h2.py`。

## ③ 结果
实验锚点（数字化）：**峰值 0.370 m @ t=1.70 s；到达时刻 0.359 s**。
（COMFLOW 自算同图：峰值 0.336 m @ 1.77 s，
到达 0.371 s。）

| run tag | t_end [s] | 峰值 [m] | 峰值误差 | 到达 [s] | 到达误差 | 峰值/到达判定 |
|---|---|---|---|---|---|---|
| dx0.02 | 1.20 | 0.517 | 39.9 % | 0.454 | 26.4 % | FAIL / FAIL |
| dx0.01 | 1.00 | 0.377 | 2.0 % | 0.444 | 23.7 % | PASS / FAIL |
| dx0.04 | 1.20 | 0.425 | 15.0 % | 0.484 | 34.8 % | FAIL / FAIL |

![digitised H2](../outputs/kleefsman_h2_ref/h2_fig19.png)

## ④ 误差判定（逐项，显式回答）
- 到达时刻：最大误差 34.8 %（dx0.02），**不满足 <10 %** → 判 FAIL。
- 峰值：误差 2.0–39.9 %，**不满足 <10 %** → 判 FAIL。
- 是否 <3 %？**否**，且不在 10 % 门限内。这是 **honest FAIL**：本单把锚点从 WITHHELD 变成可判定，
  判据本身未通过。
- 归因（可核验、非托词）：(a) 探针口径 —— 代码用贴壁 slab 探针，箱前水花/爬升使 0.08 m 口径内
  峰值系统性偏高；(b) 上游探针位置按箱面 −0.05 m 取，H2 的真实 x 在 Fig. 17 里是栅格图、本节点无法
  机器读取；(c) dx=0.02 的数值前锋偏慢 + 人工粘性；(d) 现有 run 只到 t_end=1.0–1.2 s，而实验全局
  峰值在 1.70 s（回流水再撞），故表内峰值是**窗口内局部峰**，与实验全局峰不是同一量。
- **待回收**：2.5 s 从 t=0 重算（覆盖 1.70 s 全局峰）已递交集群 `cj3027`（repo TensorSPH, ref main,
  `run.py --dx 0.02 --t-end 2.5 --tag dx0.02_t250`），排队中未回收；回收到后 `compare_h2.py dx0.02_t250`
  即可把「全局峰 vs 实验」这一格补成可判定的数字。在此之前，峰值的比较口径受限，已如实标注。

## ⑤ 注意点 / 局限
- 数字化源于 **2 倍栅格**（1 px≈0.012 s），峰值的 mm 级不确定度不改变 10 % 判决（误差量级 20–40 %）。
- 图例样式依赖 ASCII 回读（粗实线 vs 虚线段），已在 `digitize_fig19.py` 里留了可复查的输出路径。
- H2 的 x 坐标未从 Fig. 17 机器读取（栅格），沿用代码的「箱面 −0.05 m」约定；若真实 H2 更靠上游，
  到达时刻误差会被部分抵消/放大，需后续用双探针敏感性给出区间。
- 不得用本次数字去粉饰既有报告：旧文档里的 0.696 m/0.517 m 峰值来自不同探针口径，不可与实验直接比。

## ⑥ 产物
- `outputs/kleefsman_h2_ref/digitize_fig19.py`（数字化，含轴标定与曲线分离）
- `outputs/kleefsman_h2_ref/h2_fig19.csv|.json|.png`（380 点实验序列 + 双曲线覆盖图）
- `outputs/kleefsman_h2_ref/reference_h2.json`（锚点 + 出处 + 不确定度 + 交叉校验）
- `outputs/kleefsman_h2_ref/compare_h2.py` + `comparison.json`（判据实现与判决）
- `benchmarks/kleefsman_obstacle/reference.json`：`experimental_values.status` WITHHELD → **ANCHORED**
- `tests/test_kleefsman_h2_anchor.py`（锚点结构、数字化自校验、判据函数的护栏）

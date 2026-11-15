# CASE-SPH-WEDGE-ENTRY-2D — 可判性缺口收口（m21953）

**锚点** `CASE-SPH-WEDGE-ENTRY-2D`（`feature_ledger` 状态 `open`，标题「算例池·二维楔形体入水砰击压力」）
**验收** 楔形体表面峰值砰击压力系数 vs Zhao & Faltinsen (1993) JFM 246:593（DOI 10.1017/s002211209300028x），β=30° 恒速入水，容差 <15 %
**仓/分支** TensorSPH ｜ `agent/m21953` ｜ 前序 m21677（footprint 估计器）、m21742（参考侧分账）、m21468（阶梯判据）、m21430（文献锚复核）

## ① 案例介绍

二维刚性对称楔，deadrise 半角 β=30°，恒速 V=1 m/s 入水，Fr→∞、We→∞（自相似）、无粘 + Monaghan 人工粘性 α=0.03；
弱可压 WCSPH（Tait γ=7，c₀/V=20，B=ρ₀c₀²/γ），核 h=1.3dx，Courant 0.35，观测窗止于湿半宽 c_end=50 mm，
自相似坐标 τ=|x|/c(t)，湿宽系数 C=π/(2 tanβ)=2.7206990。

被评量 = 壁面压力系数 C_p = p/q（q=½ρV²）的**有限面积平均峰值** C_p,Φ：在宽 Φ=8 mm 的滑窗内取 C_p 均值的最大值
（m21677 预声明；相当于一支 8 mm 压阻传感器的读数，而不是不可测的点极大值）。

## ② 怎么做的

1. **本单认领的实质增量**：前序各轮（m20161/m21742）把判决卡在「只有 1 个可判网格」，
   缺的正是 `dx = 1 mm` 与 `dx = 0.5 mm` 两档的 footprint 工件——而集群 `fetch` 只回递交态 tarball（m21005 §③），
   于是这两档虽递交成功、数字却回不到工作树。本单先修**通道**，再用通道补数据：
2. **工件摘要通道（新工具 `artifact_digest.py`）**：作业把 `ensemble_<tag>_c0v20.json` 压成一份
   只含 ledger 所需字段的 JSON 打到 stdout（`DIGEST_BEGIN … DIGEST_END`），写作侧 `--restore` 还原成
   最小工件；还原件的 `restored_from_digest` 记录源文件 sha256 与主机名，可追溯。
   通道的**无损性由测试证明**，不是声明：`tests/test_wedge_artifact_digest.py` 对每份已提交工件做
   digest→restore→`footprint_ledger.grid_record`，要求 `obs_cp / synth_cp / bootstrap / residual` **逐位相等**（8 项，本地全绿）。
3. **补网格：递交成功，读数未取回（honest）**：以 `run_ladder_shard.py`（同一算例、同一 61 帧自相似系综、
   同一预声明 Φ=8 mm）递交 `dx = 1 mm`（cj3802，8 核）与 `dx = 0.5 mm`（cj3803，16 核）。
   本单写作时两作业已排队 22 min 仍 `queued`（集群 73 作业在飞、全席饱和），**两档读数未取回**；
   台账把它们记在 `missing_grids` 里，**不插值、不用相邻档顶替、不改判决规则**。判据文件、阈值、判决规则一字未改。
   作为补偿性证据，本单另做**确定性复核**：`dx = 2 mm` 用同一命令在本地 8 线程重跑（69 s 墙钟），
   footprint 四行数值与 bootstrap 与已提交工件**逐位相同**（steps = 587 一致），
   说明链上随机性（帧 bootstrap 种子）已固定、运行环境不是不确定性来源；
   本地墙钟硬上限 120 s（`run_shell` 超时），故 `dx ≤ 1 mm` 档**不可能**在本机产出，必须走集群。
4. **加一条与验收口径一致的副读数**（不改判决）：`footprint_ledger.py` 新增 `designated_readout`，
   把任务指定的验收对象（Z&F 1993 COM 点极大 6.92）先乘闭式有限面积比 R(ε) 再比，
   这样"传感器口径"不再由求解器买单（m21742 已量化：R = 0.9453，恒定 −5.5 %）。

## ③ 计算结果

**主判据（规则不变）**：残差 = C_p,Φ(SPH) / Wagner 场过同一 footprint 算子 − 1；可判 = 帧 bootstrap 半宽 ≤ 15 %。

| grid | dx [mm] | footprint [cells] | C_p,Φ (SPH) | 同算子参考 | 残差 r | U | 可判 | 副读数 r (vs COM×R) |
|---|---|---|---|---|---|---|---|---|
| `dx4mm` | 4.00 | 2 | 11.869 | 6.888 | +72.3 % | 30.7 % | 否 | +81.4 % |
| `dx2mm` | 2.00 | 4 | 5.999 | 7.406 | -19.0 % | 14.8 % | 是 | -8.3 % |

缺失档位：['dx1mm', 'dx0.5mm']

**判决（预声明规则，未改）**：`NOT_DECIDABLE` —— only 1 judgeable grid(s) (U > 15%)

**副读数（口径与验收一致，不进判决）**：

| 可判 grid | vs COM×R 残差 |
|---|---|
| `dx2mm` | -8.3 % |

* `inside_gate` = **True**，`brackets_target` = **False**，max|r| = +8.3 %
* 说明：single-signed residuals: no bracket

## ④ 结果图

![figA](artifacts/case_sph_wedge_entry_2d_footprint/figures/figA_dual_reference.png) — 同一批 SPH 读数（误差棒 = 61 帧 bootstrap 5–95 %）对四个参照水平：同算子 Wagner（虚线，主判据分母）、
Z&F COM×R（点划线，验收口径分母）、Wagner 点极大、Z&F COM 点极大。

![figB](artifacts/case_sph_wedge_entry_2d_footprint/figures/figB_dual_residual.png) — 两个残差族对照 15 % 门限：实线 = 主判据（同算子 Wagner），虚线 = 副读数（COM×R）；
每点标注两个残差与该档 bootstrap 半宽的可判性。

## ⑤ 注意点（坑/局限）

* **两个分母回答两个不同问题**：主判据问「SPH 相对同一理想化的解析解差多少」，副读数问「相对任务指定的
  Z&F 复合解、且口径已对齐时差多少」。二者不可混读；判决只由主判据给。
* **参考族分歧 +21.4 %**（COM 6.92 vs Wagner 8.40 点值）比 15 % 门还宽，这是本案分辨率的硬下限（m21742）。
* **弱可压噪声地板**：c₀/V=20 时 0.5 % 的密度误差就值约 4 个 C_p 单位；各档 `frame_scan.max_rho_ratio`
  若超出 1 % 带（Monaghan 1994），该档读数带着"形式假设被违背"的限定，本单照实登记。
* **作业→工件通道仍是间接的**：digest 通道保证无损，但需要作业把 digest 打到 stdout；若 stdout 也不可取回，
  该档只能记 `MISSING`，不得插值、不得用相邻档顶替。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）

| 量 | 参考 | 结果 | 误差 | < 3 % ? |
|---|---|---|---|---|
| 同算子残差 r（dx=2 mm） | Wagner 场过同一 footprint 算子 = 7.406 | 5.999 | -19.0 % | **否** |
| 副读数（dx=2 mm） | Z&F COM × R = 6.542 | 5.999 | -8.3 % | **否** |
| 同算子残差 r（dx=4 mm） | 6.888 | 11.869 | +72.3 % | 否（该档 U=31 % 不可判） |
| 参考族分歧（分辨率硬下限） | Wagner 点极大 8.402 | COM 点极大 6.92 | +21.4 % | **否（>15 % 门）** |
| 确定性：dx=2 mm 本地重跑 | 已提交工件（steps=587） | 4 行 footprint + bootstrap | 逐位相同（0.0 %） | **是** |

## ⑦ 复现

```bash
# 通道无损性（本地，无 SPH 运行）
python -m pytest tests/test_wedge_artifact_digest.py -q
# 补一档（集群）：作业内产出工件并打 digest 到 stdout
python benchmarks/case_sph_wedge_entry_2d/run_ladder_shard.py --shard 2 --of 4 --threads 8 \
    --out docs/artifacts/case_sph_wedge_entry_2d_footprint
python benchmarks/case_sph_wedge_entry_2d/artifact_digest.py \
    docs/artifacts/case_sph_wedge_entry_2d_footprint/ensemble_dx1mm_c0v20.json
# 还原 + 重算台账 + 出图 + 本文档
python benchmarks/case_sph_wedge_entry_2d/artifact_digest.py <digest.json> --restore \
    docs/artifacts/case_sph_wedge_entry_2d_footprint/ensemble_dx1mm_c0v20.json
python benchmarks/case_sph_wedge_entry_2d/footprint_ledger.py
python scripts/make_wedge_m21953_doc.py
```

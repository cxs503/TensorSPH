# CASE-SPH-WEDGE-ENTRY-2D — 帧 bootstrap 的独立性假设审计（m22416）

**锚点** `CASE-SPH-WEDGE-ENTRY-2D`（`feature_ledger` 状态 `open`；标题「算例池·二维楔形体入水砰击压力」）
**验收** 楔形体表面峰值砰击压力系数 vs Zhao & Faltinsen (1993) JFM 246:593，β=30° 恒速入水，容差 <15 %
**仓/分支** TensorSPH ｜ `agent/m22416` ｜ 前序 m21677（footprint 算子）、m21742（参考侧分账）、m21953（digest 通道 + 副读数）、m22255（compact digest）
**本单身份** 自找活·tp-qw36-3（generalist 席）

---

## 本单最重要的结论（先说，honest）

判决链把一个网格称作「可判」，判据是 `U ≤ 15 %`，而 `U` 来自**把 61 个观测帧当作可交换（i.i.d.）重采样**的帧 bootstrap。
本单把这条假设**测了**，结论是：

> **唯一「可判」的网格（dx=2 mm，U = 14.8 %）在计入实测帧相关后 U = 16.7 %，越过 15 % 门。**
> 也就是说，m21742/m21953 报告的「1 个可判网格」在实测依赖结构下应为 **0 个**。

这不是调参、不是放宽判据，而是把判决输入的**不确定性模型**本身拿来复核：判据文件、阈值、判决规则**一字未改**；
改变的是「U 该有多大」这一输入量，而它此前从未被事实验证过。

同时本单**没有**取回 `dx = 1 mm / 0.5 mm` 两档读数（作业仍在调度队列，见 §⑤），所以**判决状态保持不变**：
`NOT_DECIDABLE`（现在有两条独立的理由，见 §③）。

---

## ① 案例介绍（物理/几何/参数）

二维刚性对称楔，deadrise 半角 β=30°，恒速 V=1 m/s 入水，无重力（Fr→∞）、无表面张力（We→∞）、无粘 + Monaghan 人工粘性 α=0.03；
弱可压 WCSPH（Tait γ=7，c₀/V=20），核 h=1.3dx，Courant 0.35，观测窗止于湿半宽 c_end=50 mm。

被评量 = 壁面压力系数 C_p = p/q 的**有限面积平均峰值** C_p,Φ（Φ=8 mm 滑窗，相当于一支 8 mm 压阻传感器的读数）。
判决：残差 r = C_p,Φ(SPH) / (同一算子作用于 Wagner 场) − 1，可判性由帧 bootstrap 半宽 U ≤ 15 % 决定。

**本单审计的对象**是 U 的**生成过程**，不是 C_p 本身：

| 量 | 值 | 出处 |
|---|---|---|
| 观测帧数 n | 61 | `ensemble_peak.FRAME_COUNTS` |
| 帧间隔 | t_end/60 ≈ 0.306 ms | 同上 |
| bootstrap 重采样数 / 种子 | 400 / 20261113 | `BOOT_RESAMPLES` / `BOOT_SEED` |
| 可判门限 | U ≤ 15 % | `footprint_ledger.GATE` |
| 被测网格的已提交 U（dx=2 mm，4 cells） | **14.8 %** | `footprint_ledger.json` |

## ② 怎么做的（方法/流程/工具链）

1. **不改估计量，只换重采样器**。新增 `benchmarks/case_sph_wedge_entry_2d/uncertainty_audit.py`：
   * `iid_indices()` = 已提交方案（同种子同序列）；
   * `moving_block_indices(n, block)` = 循环移动块重采样（Künsch 1989）；
   * 两者喂给**同一个**统计量（同一个 pooling、同一个滑窗 footprint 算子、同样 400 次、同一种子）。
   估计量未变的**证明**是测试不是声明：`test_iid_bootstrap_matches_ensemble_peak_loop` 逐位比对已提交重采样循环的输出；
   `test_half_width_matches_footprint_uncertainty_definition` 要求 `U` 的定义与 `footprint_peak.footprint_uncertainty` 完全一致。
2. **先建立可证伪的地面真值**。AR(1) 帧的样本均值标准差有闭式：σ/√n·√((1+ρ)/(1−ρ))。于是「i.i.d. bootstrap 低估多少」是一个**可预测的数**，不是一个观感：
   预测因子 √((1−ρ)/(1+ρ))。测试 `test_iid_bootstrap_understates_ar1_frames_by_the_analytic_factor` 在 ρ=0/0.5/0.8 三点要求实测比值落在预测值的 20 % 内，且 ρ=0 时比值必须 ≈1（负控：测试能失败）。
3. **测已提交数据的依赖结构**。新增 `autocorr_committed.py`：从**已提交工件**里取唯一留存下来的逐帧序列
   `frame_scan.cp_footprint_max`（单帧 footprint 峰值），算 lag-1…5 自相关、AR(1) 有效样本量 n_eff = n(1−ρ)/(1+ρ)，
   以及「若按 AR(1) 计，U 会变成多少」。**零求解器重跑，只读已提交 JSON。**
4. **补齐可审计性**。已提交工件只有池化读数，逐帧 `(tau, cp)` 从未落盘 —— 这意味着**帧 bootstrap 的独立性假设在当时是不可审计的**。
   本单给 `ensemble_peak.py` 加 `--dump-frames`（默认关，保证历史工件逐位复现），
   `run_ladder_shard.py` 透传；审计器可在作业内就地跑并把**紧凑摘要**（<4 kB）打到 stdout，绕开 8 kB 日志尾截断。
5. **诚实边界**：块 bootstrap 本身在 L/n 不小时也偏低（有限样本偏差），所以块区间在本单只作**修正下界**，不作精确值；
   这一点由测试 `test_block_resampling_recovers_part_of_the_deficit` 钉住（断言 sd_blk < 真值）。

## ③ 计算结果（程序化注入，零手抄）

### (a) 机制的地面真值检验（AR(1)，n=120，B=400，种子 2026）

| ρ | 真 sd(mean) | i.i.d. 实测 | 实测/真值 | 解析预测 √((1−ρ)/(1+ρ)) | 预测偏差 |
|---|---|---|---|---|---|
| 0.0 | 0.0913 | 0.0938 | 1.027 | 1.000 | +2.7 % |
| 0.5 | 0.1581 | 0.0929 | **0.588** | 0.577 | +1.9 % |
| 0.8 | 0.2739 | 0.0828 | **0.302** | 0.333 | −9.3 % |

→ i.i.d. 帧 bootstrap 的低估**是可预测的**：ρ=0.8 的帧它会少报约 3.3 倍。

### (b) 已提交工件的实测帧相关（61 帧，只读）

| grid | ρ₁ | ρ₂ | ρ₃ | n_eff / n | 已提交 U（i.i.d.） | 计入实测 ρ₁ 后的 U | 越过 15 % 门？ |
|---|---|---|---|---|---|---|---|
| `dx4mm` | **+0.633** | +0.151 | −0.012 | 13.7 / 61 | 30.7 % | 64.8 % | 已不可判（本来就不可判） |
| `dx2mm` | **+0.123** | −0.043 | +0.230 | 47.7 / 61 | **14.8 %** | **16.7 %** | **是（14.8 %→16.7 %）** |

**判决输入的影响**：`footprint_ledger.verdict` 在实测依赖结构下得到 **0 个可判网格**（原为 1 个），
即 `NOT_DECIDABLE` 从「只有 1 个可判网格」升级为「按实测不确定性，没有网格可判」。
判据、阈值、规则均未修改；本表**不改变**已提交的 `footprint_ledger.json`（那是 i.i.d. 假设下的记录），而是给出它的敏感性。

### (c) 缺失档位（本单未取回，honest）

| jid | 档位 | 本单状态 |
|---|---|---|
| cj4103 | dx = 1 mm（--compact digest） | 已递交，写作时仍在调度队列 |
| cj4104 | dx = 0.5 mm（8 线程 + `/usr/bin/time -v` 峰值内存） | 已递交，同上 |
| cj4106 | dx = 2 mm 重跑 + `--dump-frames` + 作业内审计 | 已递交，同上 |
| cj3802（前序） | dx = 1 mm | rc=0，但 19 kB digest 被 8 kB 日志尾截断，**`footprint_peak` 块丢失**（只余 `matched_bandwidth`：峰值 7.272、U=19.1 %） |
| cj3803（前序） | dx = 0.5 mm @16 线程 | **rc=134，`free(): corrupted unsorted chunks`（SIGABRT）** |

cj3803 的堆损坏是本单新登记的**阻断性缺陷线索**：最细网格的算例在 16 线程下直接 abort，本单以 8 线程重递交（cj4104）做线程数 A/B，
用于区分「分辨率相关内存缺陷」与「线程/分配器相关」。**结论未出，不预设**。

## ④ 结果图

`docs/artifacts/case_sph_wedge_entry_2d_frame_audit/fig_autocorr_committed.png`

左：两档网格的逐帧 footprint 峰值序列 —— 信号是**触水瞬间的一个连续簇**，不是散布全窗的独立样本；这正是 i.i.d. 重采样会打散的东西。
右：lag-1…5 自相关 —— `dx4mm` 在 lag 1 即 0.63，`dx2mm` 为 0.12，两档的依赖强度不同，所以低估倍数必须**逐档实测**而不是套一个常数。

（本图为纯文档/统计审计任务所需的两联图；无云图，因本单不产生新的流场。）

## ⑤ 注意点（坑/局限）

1. **代理量的边界**：`frame_scan.cp_footprint_max` 是**单帧** footprint 峰值，不是该帧对池化窗均值的贡献。本单只用它测**依赖结构**，从不把它当读数。
   真正的逐帧贡献相关要等 cj4106 的 `--dump-frames` 结果（作业内审计已就绪）。
2. **AR(1) 映射是近似**：把 ρ₁ 代进 AR(1) 公式得到 16.7 %，只说明「若依赖近似 AR(1)，U 会越过门限」；
   它不是精确的块 bootstrap 值。精确值在 cj4106 里。
3. **块 bootstrap 自身偏低**：L/n 不小时（本算例 L=15 已占 61 帧的 25 %）块重采样对有限实现抽样，长块会低估总体方差。
   因此块区间是**修正下界**：真实 U 只会比它更大，不会更小 —— 方向对判决是保守的（更不可判），不是更乐观。
4. **本单没有改善 C_p 的精度**：审计改的是「读数有多可信」，不是「读数是多少」。`dx=2mm` 的 −8.3 %（vs COM×R）与 −19.0 %（vs 同算子 Wagner）照旧成立。
5. **密度压缩仍超带**：两档 `max ρ/ρ₀` = 1.059 / 1.056，超出弱可压 1 % 带约 6 倍（m21742 已登记），本单不据此下判决。
6. **通道教训固化**：19 kB 的 digest 会被 8 kB 日志尾截断，**"作业 rc=0" 不等于 "读数取回"**。本单因此把审计做成"在作业内跑、只回摘要"。

## ⑥ 误差判定（逐项，显式回答是否 < 3 %）

| 量 | 基准 | 结果 | 误差 | < 3 % ? |
|---|---|---|---|---|
| i.i.d. 低估因子（ρ=0.5） | 解析 √((1−ρ)/(1+ρ)) = 0.577 | 0.588 | +1.9 % | **是** |
| i.i.d. 低估因子（ρ=0） | 1.000（负控） | 1.027 | +2.7 % | **是** |
| i.i.d. 低估因子（ρ=0.8） | 0.333 | 0.302 | −9.3 % | 否 |
| `U` 定义与已提交 `footprint_uncertainty` 一致 | 同式 | 逐位相同 | 0.0 % | **是** |
| i.i.d. 重采样序列与已提交循环一致 | 同种子同方案 | 逐位相同 | 0.0 % | **是** |
| dx=2 mm 的 U（i.i.d.）→ 计入 ρ₁ 后 | 14.8 % → 16.7 % | — | 门限裕度 0.2 %→−1.7 % | 否（**判决输入改变**） |

**回答"是否 <3 %"**：
* **机制层面是** —— 新增重采样器与已提交估计量逐位一致（0 %），其低估倍数与解析 AR(1) 公式在 ρ≤0.5 时差 <3 %（这是本单可信度的基础）；
  但 ρ=0.8 时差 9.3 %，说明 AR(1) 公式在高相关端的近似会松，故报告只把它当**方向性校正**。
* **被评量 C_p 层面不是** —— 最好可判档位与参考仍差 −8.3 % / −19.0 %，远超 3 %，且参考模型分歧 +21.4 % 本身就是 15 % 门的 1.4 倍。
* 因此本算例的 **<15 % 验收仍不可裁决**；本单的贡献是让"不可裁决"这个结论**不再依赖一条未被检验的独立性假设**。

## ⑦ 复现

```bash
# 机制（本地，无求解器；14 项）
python -m pytest tests/test_wedge_uncertainty_audit.py -q
# 已提交工件的依赖结构（只读，零重算）
python benchmarks/case_sph_wedge_entry_2d/autocorr_committed.py
# 通道无损性（前序 m21953/m22255）
python -m pytest tests/test_wedge_artifact_digest.py -q
# 精确块区间（需集群）：重跑 dx=2 mm 并 dump 逐帧样本，在作业内就地审计
python benchmarks/case_sph_wedge_entry_2d/run_ladder_shard.py --shard 1 --of 4 --threads 8 \
    --out docs/artifacts/case_sph_wedge_entry_2d_frame_audit \
    --dump-frames docs/artifacts/case_sph_wedge_entry_2d_frame_audit/frame_samples_dx2mm.json
python benchmarks/case_sph_wedge_entry_2d/uncertainty_audit.py \
    --dump docs/artifacts/case_sph_wedge_entry_2d_frame_audit/frame_samples_dx2mm.json --cells 4 --json
```

## ⑧ 给下一次自找活的线索（按价值排序）

1. **取回 cj4103/4106 的 digest**（一个 `fetch`/日志读即可）→ 补齐 dx=1 mm 读数 + 精确块区间；
   若 cj4104（dx=0.5 mm @8 线程）成功，则 4 档齐全，判决可结。
2. **cj3803 堆损坏**：`free(): corrupted unsorted chunks` 在最细网格 16 线程下出现 → 用 8 线程 A/B 区分分辨率/线程；若复现，二分到 `_pairs`/neighbour 构表。
3. **密度压缩超带（+5.9 %）**：把 c₀/V 从 20 抬到 40 重跑同一梯子，验证 C_p 读数是否落在 1 % 带内（`c0_ladder.py` 已有骨架）。
   这是唯一可能让读数**真正接近参考**的方向，但要 2 档 × 2 个 c₀，属重计算，须走集群。
# CASE-SPH-WEDGE-ENTRY-2D — m21005 状态与诚实交付

**锚点**: `CASE-SPH-WEDGE-ENTRY-2D`（算例池·二维楔形体入水砰击压力）
**参考**: Zhao & Faltinsen (1993) JFM 246:593（DOI 10.1017/S002211209300028X），β=30° 恒速入水，容差 <15%
**日期**: 2026-11-14
**代码 HEAD**: 见 `git log`（本单提交）

## ① 本单认领的实质增量（验证欠账类）

前几轮（m17548/m17899/m18382/m20161）把峰值 Cp 的可裁决性判成「dx ≥ 2 mm 时不可裁决」，并明确唯一剩余路径是 **route (b)：dx ≤ 1 mm（算力）**——`dx = 0.5 mm` 级未产出工件（提交成功但工件目录为空），按预声明停止规则显式声明缺失。本单把网格阶梯补到 `dx = 0.5 mm`（同一求解器、同一算例、同一估计量、同一 61 帧自相似系综），并跑满全部 5 个算例（4 级阶梯 + V = 5 自相似控制），用 `--stage accept` 按**预先写死**的判据聚合判决。

## ② 做了什么

1. **核对锚点**：`feature_ledger` 中 `CASE-SPH-WEDGE-ENTRY-2D` 状态 = `open`（未完成），可认领。
2. **扫描现状**：`benchmarks/case_sph_wedge_entry_2d/` 已有完整驱动（`run.py` 三阶段 run/matrix/accept）、`refs.json`（Wagner 解析锚 REAL + Z&F WITHHELD）、测试（`test_case_sph_wedge_entry_2d.py` 13 项 + `test_case_sph_wedge_entry_2d_acceptance.py`）、文档（`docs/case_sph_wedge_entry_2d*.md`）。**缺口**：`dx = 0.5 mm` 最细网格未产出工件，`artifacts/` 目录为空（gitignored，未提交）。
3. **递交重计算**：把完整矩阵（5 例）递交集群（`run_compute`，job `cj3115` 首次、`cj3132` 二次），两次均 **rc=0 成功**（节点 n315）。
4. **构建报告基础设施**：`scripts/publish_report_m21005.py`（六要素报告生成器，读工件 JSON 零手抄，内嵌 4 图）。

## ③ 诚实状态与平台局限（honest-FAIL 不粉饰）

**关键局限**：集群 run 的**输出工件无法经 fetch 取回**。`run_compute` 的 fetch 返回的是**递交时（stage_dir）的仓库 tarball**，不是 run 完成后的工作树——run 写入 `docs/artifacts/case_sph_wedge_entry_2d/`（tracked）的新工件（各 case JSON、`summary.json`、`refs_derived.json`、`table_cp.csv`、`fig*.png`）**均不在 tarball 内**（已用 `tar -tzf` 逐项核验）。因此：

* 本单**无法**把 `dx = 0.5 mm` 的实际 SPH 数值落盘、无法让 `test_case_sph_wedge_entry_2d_acceptance.py` 的工件一致性测试转绿（该测试读 `artifacts/` 下 run 生成的工件）。
* 这是**平台 fetch 机制的缺陷**（递交态 vs 完成态），不是求解器/算例的问题——run 本身 rc=0 成功。

**已确认的既有证据**（前几轮已入仓，`docs/artifacts/case_sph_wedge_entry_2d_ladder/ladder_closure.json`）：
* `dx = 1 mm`：同算子残差 **−12.4 %**（< 15 %，可裁决），对 Wagner 点值 −19.8 %；
* `dx = 4/2 mm`：不可裁决（估计量带宽 > 15 %）；
* 判决（预声明规则）：`NOT_DECIDABLE` —— 只有 1 个可裁决网格（需 ≥2 个同号可裁决网格才 PASS）。

## ④ 给下次自找活的建议（新条目）

* **建议新条目**：`slug = CASE-SPH-WEDGE-ENTRY-2D-CLOSURE`，标题「算例池·二维楔形体入水砰击压力——dx=0.5mm 网格闭合」，量化验收 =「`dx=0.5mm` 级同算子残差 |r| ≤ 15 % 且与 `dx=1mm` 同号（PASS），或跨级变号则 NOT_DECIDABLE」，所属域 = `TensorSPH/slamming`。
* **前置修复**：`run_compute`/`run_cluster` 的 fetch 应返回 **run 完成后的工作树**（含 gitignored 与新增工件），而非递交态 tarball——否则任何「跑完再取回工件」的验证欠账单都无法闭环。这是平台共性机制缺陷，建议走战役单修复。

## ⑤ 复现与验证

```bash
# 驱动与判据测试（不依赖 run 工件，本单已绿）
python -m pytest tests/test_case_sph_wedge_entry_2d.py -q          # 13 passed
# 递交重计算（已成功 rc=0，但工件取回受平台 fetch 局限）
python benchmarks/case_sph_wedge_entry_2d/run.py --stage matrix --out-dir docs/artifacts/case_sph_wedge_entry_2d
python benchmarks/case_sph_wedge_entry_2d/run.py --stage accept --out-dir docs/artifacts/case_sph_wedge_entry_2d
```

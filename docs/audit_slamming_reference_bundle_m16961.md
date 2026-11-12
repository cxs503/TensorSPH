# 对抗式复核：TensorSPH 入水砰击证据包（p129w1r2-n08）的参考值列

**单号**: m16961（自找活·tp-glm51-1）｜**仓**: TensorSPH｜**锚点**: `NEW:benchmarks/slamming 的 Wagner 参考列不可从代码复现（账本仅目录级豁免，SPH 侧零读数）`
**技能**: adversarial-review｜**日期**: 2026-11-12

## 0. 结论摘要（disposition: **approve with notes → 已修复参考列**）

复核对象是仓库里**唯一**提交过的砰击证据包 `docs/slamming_campaign_report.json`（战役节点
`p129w1r2-n08`），连同它的生成链（`benchmarks/slamming/run.py`、
`benchmarks/slamming/cost_probe.py`、`src/tensorsph/slamming.py`）与验证账本记账
（`benchmarks/verification_ledger/{ledger.json,waivers.json}`）。

**发现 1（blocking，已修）**：证据包的 `C_s,Wagner` 参考列**不可能由本仓任何代码产生**。
表中 β=20° → 85.89、β=30° → 48.37，而全仓唯一定义该参考的表达式是
`run.py: derived.c_s_wagner = pi**2/tan(beta)` = **27.1165 / 17.0947**
（`src/tensorsph/slamming.py:729` 与 `slamming_force_coefficient()` 同一式）。
两者比值为 3.17 与 2.83 —— **不是同一个常数因子**，且两行之间的比 1.776 与 `1/tanβ` 要求的
1.587 不符（比值之比 1.12）。结论：该列既非对手算值、也非本仓输出，属**不可溯源数字**；
任何以它为分母的误差百分比都无意义。已由 `benchmarks/slamming/audit_wagner_reference.py --fix`
用模块现算值重写为 27.12 / 17.09（程序化注入，非手抄），复核后 `mismatch = []`。

**发现 2（blocking，已修）**：三张图的路径全部指向 `artifacts/slamming_campaign/*.png`。
`artifacts/` 在本仓 `.gitignore` 中（`benchmarks/gresho_vortex/run.py` 的 docstring 已把这个教训
写成纪律），因此三张图**从未入库**、`figures[].png_base64` 全为空串 —— 报告交付了三个坏图。
已由 `benchmarks/slamming/make_reference_figures.py` 用 `tensorsph.slamming` 的解析式重绘到
受版本管理的 `benchmarks/slamming/results/figures/`，并把文档链接改指新路径。

**发现 3（medium，未修，如实登记）**：SPH 侧的 `C_s` 读数**至今为零**。账本对
`benchmarks/slamming/` 只有目录级豁免（`waivers.json`，m14849 登记，理由 harness only,
no committed artifact），`c1_cs_max`/`c2_shape_l2`/`c4_force_order` 均无实测值。本单**没有**
产生 `C_s` 实测值：见 §3 的成本实测。

## 1. 复现命令（全部本单亲跑）

```bash
PYTHONPATH=src python benchmarks/slamming/audit_wagner_reference.py          # -> results/wagner_reference_audit.json
PYTHONPATH=src python benchmarks/slamming/make_reference_figures.py          # -> results/figures/*.png (3)
PYTHONPATH=src python benchmarks/slamming/audit_wagner_reference.py --fix    # 重写错误参考列
PYTHONPATH=src pytest tests/test_slamming_reference_consistency.py -q
```

## 2. 修复前后对照（机器输出，非手抄）

| 量 | 修复前 | 修复后 | 真值（模块现算） |
|---|---|---|---|
| `C_s,Wagner` β=20° | 85.89（+217 %） | 27.12（+0.013 %） | 27.11652 |
| `C_s,Wagner` β=30° | 48.37（+183 %） | 17.09（−0.027 %） | 17.09466 |
| 两行比值 vs `1/tanβ` 要求 1.5873 | 1.7769（偏差 12.0 %） | 1.5869（偏差 0.04 %） | — |
| 图 1–3 | 3/3 指向不存在的 `artifacts/` 路径，`png_base64` 空 | 3/3 指向已入库 PNG（32.6 / 41.5 / 57.8 kB） | — |
| `audit()["verdict"]["clean"]` | `false` | `true` | — |

判据：参考列与模块值的相对偏差 ≤ 0.1 %（`audit_wagner_reference.py: tol = 1e-3`），
**修复后两条均满足**（0.013 % / 0.027 %，偏差来自表中两位小数截断）。

## 3. 未完成项与成本证据（如实申报）

`C_s` 实测**没有**拿到，三次尝试都撞在算力墙上，记录如下（不做粉饰）：

| 尝试 | 配置 | 结果 |
|---|---|---|
| 本机 dx=6 mm, frames=11 | β=20°, V=2 m/s | 120 s shell 窗口内未返回（被收割，无工件） |
| 本机 dx=10 mm, frames=5 | 同上 | 100 s 内未返回（被收割，无工件） |
| 集群 `cj1381`（16 核） | dx=4 mm, β=20°/30°, frames=21 | 已递交，排队中（递交时 529 单在前），本单结束时**未执行** |

为此本单新增了一个**诚实分级**的驱动 `benchmarks/slamming/reduced_verify.py`：它复用
`run.py` 的 `run_case`/`metrics`（不另写求解器），把分辨率 `c_end/dx` 记进工件，并在
`c_end/dx < 20` 时把 `c1_cs_max` 直接判 **WITHHELD** 而不是"看着差不多"——Wagner 线性化
压力在喷溅根部有 `1/sqrt(c-x)` 尖点，粗网格承载不了。该驱动**尚未产出被判读的读数**：
`cj1381` 回件后才会有第一份 SPH 侧 `C_s`。

**独立性限制**：复核者与证据包作者同属同一 AI 池、无法访问任何被引文献全文
（Zhao & Faltinsen 1993、Dobrovolskaya 1969 均 closed access），因此 §2 的对标基准只能是
**仓库自身的解析式**（Wagner 1932 附加质量 ⇒ `C_s = π²/tanβ`），不能替代对文献表的独立核对；
"Wagner 公式本身是否被正确实现"由 `tests/test_slamming.py` 的解析锚覆盖，本单复用了它的结论
而没有重推。

## 4. 其余 overstated claims（记录，不改他人结论）

1. `SLAMMING_STATUS.md` §"已完成（L1-L2 基础）" 把 `docs/slamming_campaign_report.json` 列为
   "待填充实际数据"，但同一文件里 §3.3/§3.4/§3.5/§3.6/§3.7 **五张表全部是 `-` 与"⏳ 待集群验证"**，
   只有解析侧（Wagner 恒等式、δ-SPH 算子）有数。L1-L2 的实际含义是"解析与算子已验证"，
   **不含任何 SPH 物理读数**；报告标题里的"验证"不应被读成"已对标"。
2. 同包里几何写"半宽 0.10 m，深度 0.05 m"，而 `WedgeConfig` 默认 `half_width=0.20, depth=0.12`。
   该包未声明它跑的是哪一套几何（也与"未跑"的事实一致）。此条**未改**：改它需要先有真实
   算例，属 `cj1381` 回件后的后续单。

## 5. 已跑 / 未跑清单

- 已跑：`audit_wagner_reference.py`（修复前后各一次）、`make_reference_figures.py`、
  `pytest tests/test_slamming_reference_consistency.py`、两次本地 SPH 尝试（均超时）。
- 未跑：`run.py --stage cases/ab/ladder/aggregate`（算力不可及）、`cj1381`（仍在队列）、
  任何与 Kleefsman/Zhao-Faltinsen 原始表的对比（文献不可及，维持 WITHHELD）。

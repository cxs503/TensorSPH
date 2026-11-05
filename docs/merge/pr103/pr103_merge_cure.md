# PR #103（agent/m3163）x main 的治愈合并记录

**任务**：TensorSPH PR #103 对 main 变脏（`mergeable=false`）→ 三方合并、逐文件解冲、核心测试不低于 main 基线、把成果回写分支。
**执行分支**：`agent/m3877`（本文件所在分支）；**治愈后的合并提交**：`ddda891`（第一父 = `b7af006` = `agent/m3163` 旧 SHA，第二父 = `ba07a10` = `origin/main`）。
**旧/新 SHA**：`agent/m3163` 旧 `b7af0066f065dedc63ed2d8729a3259d6b8e94f1` → 新 `ddda891`（本机提交；直推被安全策略拦截，`agent/m3877` 的 tip 就是该合并提交，从 `agent/m3163` 快进即可发布）。

## ① 案例介绍

冲突不是代码，而是 **Young–Laplace 压差分辨率阶梯的三份实测工件**：

`benchmarks/laplace_law/convergence/relax_dx{0.0025,0.0020,0.0015}.json`

物理算例（两侧设置完全相同）：半径 R = 20 mm 的水滴（ρ₀ = 1000 kg/m³、σ = 7.2 N/m、c₀ = 20 m/s、μ = 0.01 Pa·s、γ = 7 的 Tait 状态方程、三次样条核 h = 1.3 dx、六方晶格），从静止均匀质量出发做毛细松弛，读出压差并对比解析 Δp = 2σ/R = 360 Pa。

| 参数 | dx = 2.50 mm | dx = 2.00 mm | dx = 1.50 mm |
|---|--:|--:|--:|
| 粒子数 N | 228 | 362 | 838 |
| branch 协议步数 / t/t_σ | 684 / 1.0004 | 855 / 1.0004 | 1140 / 1.0003 |
| main 协议步数 / t/t_σ | 900 / 1.3162 | 900 / 1.0530 | 900 / 0.7897 |

两侧各自重跑了同一算例但**协议不同**：main 用固定 900 步（三个分辨率落在 0.79 / 1.05 / 1.32 t_σ，比较的是三个不同毛细相位），branch 用固定物理时刻 t/t_σ = 1.0（这正是 PR #103 要修的方法学缺陷）。

## ② 怎么做的

1. 在独立工作树 `/root/tp-agent/work/m3877_merge` 检出 `agent/m3163`，`git merge origin/main` → 三个 JSON `CONFLICT (content)`，其余 52 个文件自动合并（main 侧新增 `benchmarks/{ci_lint_gate,ci_red_clearance,conservation,init_consistency,poiseuille}`、`src/tensorsph/lattice_consistency.py` 等；branch 侧新增 `run_ladder.py`、`make_ladder_doc.py`、`docs/verification_laplace_resolution_ladder.md`、`tests/test_laplace_resolution_ladder.py`，两侧文件集不相交，无一被覆盖）。
2. **数据工件不能做标量合并**（两套物理跑动），因此按「语义并集 + 零静默丢失」解冲，脚本 `docs/merge/pr103/union_merge.py`：
   - 顶层保留 **branch 协议**（它是 schema 超集：含末态 `field` 快照与 absolute 读数，阶梯、其测试与 `docs/verification_laplace_resolution_ladder.md` 都由它渲染）；
   - main 侧全部数字逐字保存在 `provenance.superseded_protocol_run`（setup / reference / laplace / spurious_current / curvature_final / history），并写明为何不把固定步数协议作为阶梯协议。
3. 脚本随后做**机器化无丢失证明**：把 base/main/branch 三方 JSON 展平成叶子路径集合，逐个断言（a）branch 每个叶子在原路径且值未变；（b）main 每个叶子在 preserved 块中以**完全相同的值**存在。任一不满足即 `LOST` 并非零退出。结果见 `docs/merge/pr103/union_audit.json`。
4. 另发现 branch 的阶梯图守卫红：`benchmarks/laplace_law/laplace_resolution_ladder.png` 在 `b7af006` 中缺失（该提交注明"1 个文件被门禁滤除"）。用 `make_ladder_doc.py` 从工件重渲染（文档文本与提交版**逐字节一致**，说明只是补图），随合并提交入库。
5. 工具链：git 三方合并 + 自写并集/审计脚本、pytest 9.2.0（`-n 24` xdist）、ruff（E/F/W）、matplotlib（Agg）。图中每个数字由 `docs/merge/pr103/{extract_ladder_compare.py,make_report_figures.py}` 从工件注入，零手抄。

## ③ 计算结果

冲突文件清单与解法：

| 冲突文件 | main 侧 | branch 侧 | 解法 |
|---|---|---|---|
| `convergence/relax_dx0.0025.json` | 900 步重跑（130 叶子：setup/laplace/spurious/curvature/history） | t/t_σ=1.0 重跑 + `field`（1033 叶子） | 顶层 branch（1033）+ `provenance.superseded_protocol_run` 保 main（130）= 1167 叶子 |
| `convergence/relax_dx0.0020.json` | 同上 | 1589 叶子 | 1589 + 130 = 1723 |
| `convergence/relax_dx0.0015.json` | 同上 | 2734 叶子 | 2734 + 130 = 2868 |

合并后两套协议的读数（`ladder_compare.json`，Δp 参考 360 Pa）：

| dx [mm] | branch bandA 误差 | branch absolute 误差 | branch v_max/v_ref | main bandA 误差 |
|--:|--:|--:|--:|--:|
| 2.50 | −84.8 % | −33.7 % | 0.0219 | −86.4 % |
| 2.00 | −89.8 % | −18.9 % | 0.0263 | −106.6 % |
| 1.50 | −80.4 % | −21.2 % | 0.0254 | −112.3 % |

图（`pr103_protocol_compare.png`）：(a) 两个协议读出的毛细相位——main 的固定 900 步在三个分辨率上散在 0.79/1.05/1.32 t_σ，branch 恒为 1.0；(b) 同一算例两套跑动的 bandA 误差并列，**没有任何一侧的数字被覆盖或平均**。
图（`pr103_union_coverage.png`）：并集覆盖率——合并后叶子数 = branch 叶子 + main 叶子，丢失 0。

测试证据（同一容器、同一命令、`-n 24`）：

| 树 | 结果 |
|---|---|
| `origin/main` 基线（`ba07a10`，独立工作树） | **2 failed, 363 passed, 8 xfailed**（290.7 s） |
| 合并后（`ddda891`） | **2 failed, 372 passed, 8 xfailed**（225.3 s） |

两侧失败的**是同样两个**用例：`tests/test_oscillating_droplet.py::test_recorded_mode2_frequency_deficiency_is_reproduced` 与 `::test_oscillation_frequency_gate_is_not_met`——main 于 2026-11-04 自己把它们设成 `xfail(strict=True)`（mode-2 拟合对架构敏感：x86_64 CI +8.64 % vs loong64 −31.78 %），本容器上它们 XPASS 于是 strict 判失败。**基线红，合并后同样红，且不是本 PR 引入**；合并后通过数 363 → 372（+9，来自 branch 的阶梯测试），失败集合不变。
定向复验：`pytest tests/test_laplace_resolution_ladder.py tests/test_laplace_law.py` → **18 passed, 1 xfailed**（含「文档 = 工件逐字节重渲染」与「图存在且 >20 KB」两条守卫）。

## ④ 结果图

- `docs/merge/pr103/pr103_protocol_compare.png`：(a) 协议相位对照（t/t_σ vs dx，两协议 + 1.0 参考线）；(b) 两协议 bandA 误差并列柱状图 + 15 %/3 % 判定线。
- `docs/merge/pr103/pr103_union_coverage.png`：三份冲突 JSON 的叶子覆盖（branch / main / merged），直观看"零丢失"。

## ⑤ 注意点

- **数据工件没有"标量三方合并"这回事**：两侧是两次独立物理跑动，机械 overlay（`--ours`/`--theirs`）会静默丢掉一侧全部数字——这正是本单禁止的做法。
- main 的 `benchmarks/ci_red_clearance/make_doc.py` 曾用这三份 JSON 的 `steps`/`t_over_t_sigma` 作为"红①已清"的证据（当时读到 900 步）。合并后顶层回到 t/t_σ=1.0 协议，**main 的 900 步数字仍在 preserved 块里可审计**；若重跑该文档生成器，"after"列会取到 branch 协议的数——这是已知的口径差异，需在下次重渲染时明确注明，本文不改动 main 的历史报告。
- `benchmarks/laplace_law/` 目录本身有 32 条**既有** ruff 违规（E501 等），与本合并无关；新增的本目录脚本已过 `ruff check`（E/F/W）。
- 直推 `origin` 被安全策略拦截（符合平台纪律）：治愈后的合并提交停在本地 `ddda891`，`agent/m3877` 的 tip 即该提交，**从 `agent/m3163` 快进发布即可**（`ddda891` 的第一父就是 `b7af006`，无需 force）。
- JSON 工件里新增了 `provenance` 键，渲染器不读它，故阶梯文档仍逐字节一致；若未来有消费者严格校验顶层键集合，需一并更新。

## ⑥ 误差判定（关键量 vs 基准）

| 关键量（本单判据） | 基准 | 实测 | 是否达标 |
|---|---|---|---|
| 冲突两侧叶子静默丢失数 | 0 | **0**（1167/1723/2868 叶子中，main 130×3 + branch 1033/1589/2734 全部存活，值未改） | 是（100 % 守恒，远超 3 % 口径） |
| 测试失败数（合并后 vs main 基线） | ≤ 基线 2 | **2（同一对用例）** | 是（差 0） |
| 通过用例数（合并后 vs 基线） | ≥ 363 | **372** | 是（+9） |
| 阶梯文档 vs 工件重渲染 | 逐字节一致 | 一致 | 是 |
| 阶梯图守卫 | ≥ 20 000 B | 274 733 B | 是 |

**是否 < 3 %？**——本单是合并/集成任务，唯一可量化的"误差"是**内容丢失率与测试回归率**，两者均为 **0 %（< 3 %）**。物理量方面：合并后阶梯仍是诚实 FAIL（absolute 读数 −18.9 … −33.7 %，bandA −80 … −90 %），**没有任何一侧被用来"修饰"结果**；这些数只用于证明合并未改动测量，不构成物理精度改善声明。
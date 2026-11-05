# PR #87（agent/m2919）→ main 冲突治愈报告

**结论先行**：4 个文件是 add/add 冲突，两侧都是**同一基准的独立实现**（不是同一脚本的两个版本）。
机械 overlay 会静默丢掉一整侧的实现与报告，因此本次按「语义并集」处理：两套实现、两套测试、
两份报告数字全部留档，且用测试证明合并后不降级、并已抓到一处**非文本冲突**（`Config.csf_curvature`
默认值从 `mls` 漂到 `fit_quadratic`），若不显式钉住，分支验证过的物理配置会被静默改写。

| 项 | 值 |
|---|---|
| main（合并目标） | `ba07a1066bcb2be5046dc63ae0179273b9f02ee7`（Merge #104 via merge-governor） |
| PR 分支 `agent/m2919` 旧 SHA | `72f966fd6795fb0fb0b4c183bd86077f56d128a4`（m2919: matured mission deliverables） |
| 三方合并 base | `5800532752d8e0f9104e06c412f6af6797b1452f`（Merge PR #53） |
| 分叉度 | main 领先 40、分支领先 1（`git rev-list --left-right --count`） |
| 冲突文件 | 4（全部 `AA` add/add） |
| 分支干净并入文件 | `benchmarks/capillary_wave/validation.py`（575 行，语法校验通过） |
| 合并后测试 | 25 passed / 2 failed（与 main 基线同样的 2 个既有失败）/ 27 collected |

---

## ① 冲突清单与逐文件解法

| # | 文件 | main 侧 | 分支侧 | 解法（语义并集） | 防静默丢失的证据 |
|---|---|---|---|---|---|
| 1 | `benchmarks/oscillating_droplet/run.py` | 567 行（elliptical_drop / analytic_omega / peak_decay / studies / plot / doc 注入） | 474 行（lobed_droplet / drop_frequency_2d·3d / 两级网格 fit / mode-2·3 阶梯） | `run.py` 取 main 原文；分支实现**整文件保留**为 `run_m2919.py`，加 12 行 provenance 注释说明来源与运行方式 | `git diff origin/main -- run.py` 为空；`run.py` 与 main 逐字节相同 |
| 2 | `benchmarks/oscillating_droplet/validation_report.json` | 135 行（单算例摘要） | 8934 行（4 算例 + 完整 time history + 场快照） | main 的报告留在原路径；分支报告保留为 `validation_report_m2919.json` | 两份 JSON 均 `json.loads` 通过；行数 135 / 8934 与两侧原值一致 |
| 3 | `tests/test_oscillating_droplet.py` | 258 行 / 12 项（曲率展开复推、估计量、拟合器、真实松弛门） | 152 行 / 19 项（解析分支、拟合器基线+噪声负控、模态对称性、求解器回归） | 单文件并集：main 套件原文在下，分支套件追加，import 绑到 `run_m2919`（两套 API 不同，不能互相顶替） | AST 比对：main 17 项 + 分支 19 项 = 合并 31 顶层项，**缺失 0**；无重名定义 |
| 4 | `docs/verification_oscillating_droplet.md` | 131 行（正文，对应 `run.py`） | 90 行（分支报告，含公式陷阱/曲率噪声/for-else bug 判例） | main 正文保留，分支文档作为「附 A」整节保留（标题降级一级，工件路径改指 `*_m2919`，无内容删减） | 附 A 逐段对应分支原文；仅 4 处路径替换（见 `docs/merge_pr87/` 日志） |

### 为什么不是「取一侧」

两侧 `run.py` 的公开 API 不兼容（`run_case` 签名与返回结构、`ellipsoidal`/`lobed` 几何、解析式函数名、
`main()` 输出格式全不同），合进同一文件必须给其中一侧改名，等于把「谁取代谁」这个判断藏在重命名里。
保留为两个模块后：main 的文档/测试/报告路径不变，分支的可复现入口 `run_m2919.py` 也仍然可跑。

为避免两个 benchmark 互相覆盖产物，保留副本的三处输出名改为 `*_m2919`
（`validation_report_m2919.json`、`oscillating_droplet_m2919.png`、`oscillating_droplet_m2919_decay.png`）。

---

## ② 非文本冲突：`Config.csf_curvature` 默认值漂移（本次最重要发现）

分支的 `run_case` 只构造 `Config(...)`，**没有显式指定曲率方案**。而 main 在同一位置把默认值改了：

```text
git diff 5800532 origin/main -- src/tensorsph/droplet_impact.py
-    csf_curvature: str = "mls"
+    csf_curvature: str = "fit_quadratic"
```

于是「文本合并没有冲突」的表象下，分支的物理配置被静默换掉了。同一算例
（`R=20 mm, dx=2.5 mm, n=2, ε=0.05, 233 粒子, 1 个解析周期`）实测：

| 曲率方案 | ω 实测 [rad/s] | Lamb 2-D 基准 [rad/s] | 误差 | 拟合残差 | 分支 `TestSolverRegression` 的 <30 % 守卫 |
|---|---|---|---|---|---|
| `fit_quadratic`（main 新默认，**未钉住**） | 28.154 | 73.485 | **−61.69 %** | 0.179 | **失败**（`assert 61.6875 < 30.0`） |
| `mls`（分支基线方案，**已钉住**） | 64.033 | 73.485 | **−12.86 %** | 0.054 | 通过 |

处理：在 `run_m2919.py` 里把 `csf_curvature` 提升为 `run_case` 参数并默认 `"mls"`（附注释说明
「main 默认值在本基准基线之后发生漂移，本基准的数字只对它调参时的方案有意义」），
并在 `case` 记录里回写实际使用的方案（原代码已记录 `cfg.csf_curvature`，现在它会如实反映选择）。

**未做的归因（如实声明）**：分支历史报告里 `mode2` 记录值 28.1538 与本机未钉住时的 28.154 数值相同，
但分支当时用的是旧内核（`csf_consistent.py` 在 main 上 +39 行），因此**不能**断定历史数字产生于哪种方案；
本节只陈述本次可控实验：同一棵树、同一算例，两个方案差 48.8 个百分点。

---

## ③ 测试证据（程序化记录，日志在 `docs/merge_pr87/logs/`）

| 树 | 命令 | collected | passed | failed | 用时 |
|---|---|---|---|---|---|
| main 基线（`/tmp/base38`，`ba07a10`） | `pytest tests/test_oscillating_droplet.py -v` | 12 | 10 | 2 | 221.98 s |
| 合并结果（本树） | 同上 | 27 | **25** | 2 | 169.65 s |
| 分支 tip（`/tmp/br38`，`72f966f`） | `pytest ... -k TestSolverRegression -v` | 2 | 2 | 0 | 40.56 s |

- **不降级**：合并后通过数 25 ≥ 基线 10，且基线通过的 10 项在合并树内全部仍通过；新增的 15 项分支测试全绿。
- **同样的 2 个失败是既有的**：`test_recorded_mode2_frequency_deficiency_is_reproduced`、
  `test_oscillation_frequency_gate_is_not_met`，两侧都因 `xfail(strict=True)` 在本机 **XPASS** 而计为 FAILED
  （main 自己的注释：mode-2 拟合架构脆弱，x86_64 +8.64 % vs loong64 −31.78 %，等重新定基线）。
  合并既未引入也未修复它们。
- 分支的两条求解器回归在合并树里默认跳过（`@pytest.mark.slow`，与 main 把仿真重型用例标记为 slow 的约定一致，
  单条 ~20 s），需要时 `pytest tests/test_oscillating_droplet.py -m slow` 运行。

质量门：`ruff check artifacts/ tests/ benchmarks/oscillating_droplet` 全通过（分支原文件本身有 1 处
E501 + 2 处 W292，是它从未进 main、未过 lint 门的历史欠账，本次一并在保留副本里修掉，语义不变）。

---

## ④ 分支内的物理结论（两侧都维持 honest-FAIL，未因合并而美化）

分支报告的 4 个算例（`validation_report_m2919.json`，程序化注入）：

| 算例 | n | dx [mm] | 粒子数 | ω_Lamb | ω 实测 | 误差 | 残差 | 稳定性 |
|---|---|---|---|---|---|---|---|---|
| mode2 | 2 | 1.5 | 639 | 73.485 | 28.154 | −61.69 % | 0.105 | 粒子出箱 @step 778 (0.44 T) |
| mode3 | 3 | 1.5 | 648 | 146.969 | 56.308 | −61.69 % | 0.416 | 粒子出箱 @step 1280 (1.46 T) |
| coarse | 2 | 2.0 | 365 | 73.485 | 62.111 | −15.48 % | 0.202 | 粒子出箱 @step 1200 (0.91 T) |
| no_sigma（负控） | 2 | 1.5 | 639 | — | 38.313 | — | 0.034 | 稳定 |

main 正文的基准算例是 −43.5 %（且自述那一轮已发散）。**两侧都是 FAIL，均未进入 15 % 门，更没有进入 3 % 门**；
动态（振荡）表面张力的证据等级维持 **WITHHELD**。合并只保证两侧证据都在、都可复现，不改变这个判定。

---

## ⑤ 注意点 / 局限

1. **不要用「谁的 run.py 更新」来决定取舍**：两侧是并行实现，行数多的那份（main 567 行）不等于覆盖了分支的
   mode-3 阶梯与噪声负控，反之亦然。判据是「两侧内容都在 + 两侧测试都跑」。
2. **默认值也是合并面**：`Config` 默认值、`dt` 判据、`csf_scheme` 之类的隐式契约不会产生 git 冲突，
   但会改变结果。本单靠「分支自己的回归测试在合并树上变红」才暴露出来——**合并后必须跑分支测试，不能只跑 main 的**。
3. **未推送到 GitHub**：`origin` 是 `https://github.com/cxs503/TensorSPH.git`，本环境无凭据且平台纪律禁止自推；
   合并提交已落在本工作树分支上（SHA 见 commit / `finish`），平台推送后 PR #87 的 diff 即包含 main 最新内容、
   `merge_governor` 可复验。**这一步（推回 `agent/m2919`）是唯一未按原单执行的动作，原因是权限与纪律，不是遗漏。**
4. **未跑全仓测试**：只跑了 `tests/test_oscillating_droplet.py` 的全量（基线 vs 合并）以及合并触及文件的 lint；
   改动范围限于 `benchmarks/oscillating_droplet/`、`tests/test_oscillating_droplet.py`、`docs/`，未触及
   `src/tensorsph/`，故未做全仓回归（本机单文件仿真用例已占 170–220 s，全仓会超预算）。
5. **`fit_damped_oscillation` 报告的 `gamma_1_s` 为 null**：分支 `measured` 字典里没有这个键（原实现只在
   `summary_row` 里用 `gamma_measured`），保留副本未改这一点，仅记录，未修补。

## ⑥ 复现命令

```bash
# 合并冲突复现
git merge --no-ff --no-commit origin/agent/m2919     # 4 个 AA 冲突

# 语义并集（本单的构造脚本，幂等，从 docs/merge_pr87/sides/ 的两侧原文重建）
python3 docs/merge_pr87/build_merge.py

# 测试
PYTHONPATH=src python3 -m pytest tests/test_oscillating_droplet.py -v
PYTHONPATH=src python3 -m pytest tests/test_oscillating_droplet.py -m slow -v   # 分支两条求解器回归
python3 -m ruff check tests/ benchmarks/oscillating_droplet artifacts/

# 曲率方案对照（28.154 vs 64.033 rad/s）
PYTHONPATH=src python3 docs/merge_pr87/probe_omega.py   # 钉住 mls 后的一档
python3 docs/merge_pr87/make_figures.py             # 图 + evidence.json
```
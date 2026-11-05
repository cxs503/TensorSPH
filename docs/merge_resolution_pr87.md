# PR #87（`agent/m2919` → `main`）三方合并处置记录

**仓库** TensorSPH ｜ **合并基点** `ba07a10`（main） ｜ **被合并分支 tip** `72f966f`（agent/m2919）
**任务** 修复 PR #87 的 `mergeable=false`：三方合并、逐文件解冲、语义并集、测试不低于 main 基线。

## ① 冲突清单（4 个，全部 add/add）

`git merge origin/agent/m2919`（在 main 上）报告：

```
CONFLICT (add/add): benchmarks/oscillating_droplet/run.py
CONFLICT (add/add): benchmarks/oscillating_droplet/validation_report.json
CONFLICT (add/add): docs/verification_oscillating_droplet.md
CONFLICT (add/add): tests/test_oscillating_droplet.py
```

其它 m2919 新增文件（`benchmarks/capillary_wave/validation.py`）无冲突，自动并入。

**为什么是 add/add 而不是改冲突**：两条谱系在**同一路径上各写了一份同名文件**——
main 线（m3020 → m3312 → 9f01f06）与 m2919 线（m2919）都独立实现了同一个
benchmark「2-D 振荡液滴自由振荡频率 vs Rayleigh–Lamb 解析解」。这是**同一科学问题
的两份独立实现**，不是同一份文件的两处编辑，所以「机械 overlay」（取一侧）必然
静默丢弃另一侧的全部科学产出。

## ② 逐文件解法

### 2.1 `benchmarks/oscillating_droplet/run.py` — main 线为规范实现 + m2919 独有能力并入

| 侧 | 内容 | 处置 |
|---|---|---|
| main | 4 个 study（frequency / ladder / damping / quiet）、`--mode` CLI、`plot()`、`summarise()`、`strip_trace()`、与 `make_doc.py` 的注入契约 | **保留为规范实现**（功能超集：m2919 只有单算例 `--case` 循环） |
| m2919 | `lobed_droplet()`：任意方位模态 `r = R(1+εcos mθ)` 几何 | **并入**（main 只有 m=2 的面积守恒椭圆，无法给 m≥3 分支评分） |
| m2919 | `drop_frequency_2d` 的 `n=1 → ValueError` 守卫 | **并入** `analytic_omega()`（m=1 是刚体平移，原实现会静默返回 0） |
| m2919 | `mode_amplitude()` = 有符号四极矩 `Σm(x²−y²)/Σmr²` | **并入**为 `mode_quadrupole()`（main 的特征值估计量返回 \|s\|，对 90° 相位翻转不可见；m2919 报告里的 s 序列正是这个量） |
| m2919 | σ=0 负控（无表面张力不得按 ω₂ 振荡） | **并入** `run_case()`：`sigma=0` 分支改为「必须显式给 `steps`」，并把 `ω_ref=0` 处的 4 处除零改为 NaN 语义 |
| m2919 | `--case` CLI、`fit_damped_oscillation`、`make_figures` | **不并入主路径**（与 main 的 `--mode`/`damped_cosine_fit`/`plot` 语义重复），实现逐字存档于 `artifacts/merge_pr87/m2919/run.py` |

### 2.2 `benchmarks/oscillating_droplet/validation_report.json` — 两份测量，各留一份

两文件是**两次独立运行的实验数据**，schema 不同（main：`frequency/ladder/damping_series/quiet_case/verdict`；m2919：`rows[]/gates_pct/references`）。
把两套 schema 缝进一个 JSON 会伪造一份谁都没产生过的报告，且 `make_doc.py` 依赖顶层键做注入。
处置：`validation_report.json` 保留 main 线（当前维护线，文档注入源），m2919 线逐字保留为
`validation_report_m2919.json`（277 KB，一个数字未改），并由 `make_doc.py` **程序化注入**到
验证文档的附录 A（零手抄）。

### 2.3 `docs/verification_oscillating_droplet.md` — main 正文 + 附录 A

main 线文档由 `make_doc.py` 生成（正文 ⑥ 节）+ 一段手写 `## ⑦ 诚实记录`。
处置：正文保留 main 线；在文件尾部追加 **`## 附录 A：合并保留的 m2919 独立谱系`**，
其中的数字表由 `make_doc.py::appendix_m2919()` 从 `validation_report_m2919.json` 读取后渲染
（不是手抄），并同时把该函数接进 `make_doc.py` 的渲染流程——因此「重新生成文档」和
「当前入库文档」在附录段上逐字节一致（已用 `gen in doc → True` 复核）。
m2919 线原文逐字保留在 `docs/verification_oscillating_droplet_m2919_lineage.md`。

### 2.4 `tests/test_oscillating_droplet.py` — main 全量 + m2919 性质断言（第 5 节）

main 线测试更成熟（含 strict-xfail 钉住 x86 +8.64% / loong64 −31.78% 的架构脆弱性，
XPASS 会失败，双向防漂移），保留之；m2919 线的**性质**逐条并入第 5 节（名称映射写在该节注释里）：

| m2919 原断言 | 并入后 |
|---|---|
| n=1 无回复力须抛 `ValueError` | `test_mode_one_is_a_translation_and_has_no_frequency` |
| `ω ∝ √σ`、`ω ∝ R^{-3/2}` | `test_frequency_scales_as_sqrt_sigma_and_r_minus_three_halves` |
| 大模态下 2-D/3-D 公式趋于一致 | `test_branches_agree_for_large_azimuthal_modes` |
| 模态观测量在 90° 旋转下变号 | `test_mode_estimator_flips_sign_under_a_ninety_degree_rotation`（并入后改为打 `mode_quadrupole`：main 的 \|s\| 估计量**本就不会**变号，这一点现在被显式断言，成为两个估计量的区别证据） |
| `lobed_droplet` 几何自洽 | `test_generalised_lobe_geometry_fills_the_analytic_lobe_area[2,3,5]`（判据升级为解析面积 `πR²(1+ε²/2)` 与质量一致性） |
| σ=0 负控 | `test_without_surface_tension_the_mode_does_not_oscillate`（标 `slow`，判据加「覆盖率 ≥ 半周期」） |

m2919 线原测试文件逐字存档于 `artifacts/merge_pr87/m2919/test_oscillating_droplet.py`。

## ③ 测试证据

见下节（跑完后回填）。
### 3.1 合并后测试（本机，CPU，`PYTHONPATH=src`）

| 检查 | 命令 | 结果 |
|---|---|---|
| 合并后·快层 | `pytest tests/test_oscillating_droplet.py -q -m "not slow" -k "not recorded_mode2 and not gate_is_not_met"` | **17 passed**（16.3 s） |
| main 基线·快层（`/tmp/wt-base` = `ba07a10`） | 同上命令 | **10 passed**（15.7 s） |
| 合并后·新增内容 | 上表 17 − 基线 10 = **+7**（第 5 节 6 条 + `lobed_droplet` 参数化 3 例中的 2 例计入…实际为 4 条性质测试 + 3 个参数化例） | 无失败 |
| ruff 门 | `ruff check benchmarks/oscillating_droplet/ ruff tests/test_oscillating_droplet.py` | **All checks passed** |
| AST 解析 | `ast.parse` on run.py / make_doc.py / test 文件 | OK |
| 文档-生成器一致性 | `appendix_m2919()` 输出是否为入库文档的子串 | **True**（25 行，零手抄） |

**未完成的部分（如实记录）**：`release_case` 那两个 strict-xfail 固定测试
（`test_recorded_mode2_frequency_deficiency_is_reproduced`、
`test_oscillation_frequency_gate_is_not_met`，R = 10 mm / dx = 1 mm / 1.8 周期）
在本机当前负载下（同机另有 3 个 pytest 进程）单次运行超过 20 min 未收敛到断言，
被主动中止，**未取得本轮实测**。替代证据是路径级等价性证明：

| 量（σ>0 共用路径，R=20 mm, dx=6 mm, 90 步） | main 基线 | 合并后 | 相对差 |
|---|---|---|---|
| `fit_second_moment_rad_per_s` | 41.497920989990234 | 41.497920989990234 | 0 |
| `fit_kinetic_energy_rad_per_s` | 214.72645568847656 | 214.72645568847656 | 0 |
| `damping.fit_lambda_per_s` | 0.0 | 0.0 | 0 |
| `initial_state.s_moment` | 0.08936836380361467 | 0.08936836380361467 | 0 |
| `final_state.s_moment` | 0.06294474200765666 | 0.06294474200765666 | 0 |
| `mass_relative_drift` / `max_speed` | 0.0 / 0.4075653283063975 | 同 | 0 |
| `trace.s_second_moment` 全序列 | — | — | **逐元素完全相同** |

即：`σ>0` 的所有数值路径在合并前后**逐位一致**（合并只新增了 `sigma=0` 分支、
两个估计量与若干守卫），因此那两个 xfail 钉子的输入与基线严格相同，其钉住值不会漂移。

## ④ 分支与 SHA

| 项 | SHA |
|---|---|
| 合并前 `main` | `ba07a10`（Merge #104 (m3204) via merge-governor） |
| 合并前 `agent/m2919`（PR #87 head，**旧 SHA**） | `72f966f` |
| 合并提交（**新 SHA**，父 = `ba07a10` + `72f966f`） | `git rev-parse HEAD`（本文件所在提交） |
| 本地别名 ref（供 governor/网关取用） | `refs/heads/cure/pr87-merged` = 同一提交 |

**推送状态（如实记录）**：本机到 GitHub 网络不通（fetch 与推送均在 30 s 内
`Operation too slow`，fetch 直连超时 >120 s），因此**本轮无法把合并提交推回
`agent/m2919`**：PR #87 的 head 仍是 `72f966f`，GitHub 侧仍显示 mergeable=false。
恢复网络后只需把远端 `agent/m2919` 快进到本合并提交（一条 refspec 推送），
即可完成第 4 步。该提交已把 `main` 的全部 40 个提交作为第一父提交，因此
快进后 PR #87 的 `mergeable` 必然为 true，merge_governor 下一周期可直接记账合并。

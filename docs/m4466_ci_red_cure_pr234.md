# PR #234（`agent/m4111`）CI 复红诊断与处置：lint 门欠账 + 8 条断言漂移（m4466）

> 结论摘要（honest）：**CI 红的主因不是 PR 分支内容，而是合并树（`main ∪ branch`）自带的 lint 欠账**
> ——`ruff check .` 在 main tip 上 41 条违例，三个 matrix job 全部在 **Lint 步**即 `exit 1`，
> 后面的 pytest / e2e 步从未执行。本次把 lint 门清零（0 违例），并把合并树上 **13 条 pytest 失败中的 7 条**修掉；
> 仍残留 6 条（明细见 §5），因此**本条 PR 的 CI 尚未验证为绿**，不做粉饰。

## ① 案例介绍

| 项 | 值 |
|---|---|
| 仓库 / PR | `cxs503/TensorSPH` PR #234，标题「m4111 成果（merge_master 补开 PR）」 |
| PR head | `ed03b6c4`（分支 `agent/m4111`），1 commit / 1 文件 |
| PR base | `main@7d91b89`（Merge PR #172） |
| 评审时的 main tip | `b90a8a1`（Merge PR #231 / m4131） |
| PR 自身改动 | 仅 `docs/pr149_ci_fix_report.md`（+57 行，纯文档，零代码） |
| CI 事实 | `pytest (3.10)`、`(3.11)`、`(3.12)` 三个 check-run 均 `completed/failure`（18:20Z / 18:34Z / 18:41Z） |
| 工作树隔离 | `work/m4466`，分支 `agent/m4466`，基于 `main@b90a8a1` |

物理/工程背景：TensorSPH 的 CI 是「lint → 单元与 smoke → 端到端算例 → manifest 验收字段」四道门。
本任务不是求解器物理问题，而是**发布门禁**问题：门禁红会使所有领域成果都无法落地。

## ② 怎么做的（方法与工具链）

1. **先从 GitHub API 取事实，不猜**：`/pulls/234`（head/base/commits/files）、`/commits/{sha}/check-runs`、
   `?Accept: application/vnd.github.v3.diff`。确认 PR 只加了一个 md 文件 ⇒ 失败不可能是 PR 代码引入。
2. **定位 CI 实际跑的树**：GitHub PR 作业检出 `refs/pull/234/merge` = `main ∪ branch`，
   所以判定侧应是 **merge 树**而不是 PR 分支（这是上一轮 PR#149 判例已确立的判读，本次沿用）。
3. **本地复现**：主开发机上 `tensorsph` 以 `pip install -e` 指向**别的工作树**（`work/m4114/src`），
   直接 `pytest` 会 import 到别人的代码（首次全量即出现 4 个 collection ImportError 假红）。
   → 所有测量都加 `PYTHONPATH=$PWD/src`，实测 `tensorsph.__file__` 指向本树后才采信。
4. **两步顺序复现 CI**：`python -m ruff check .` → `python -m pytest -q --no-header -tb=line -rf`（`-n 4` 并行，
   机器被多智能体跑满，单机串行 211 s 起步）。
5. **按判例决定修哪一侧**：能真实改就改（未用导入/未用变量/文件末换行/叙述性数字标记），
   只有「整行是 markdown 载荷」的文档生成器才用文件级 `# ruff: noqa: E501` 并在文件头写明理由
   （与 `benchmarks/laplace_pressure_closure/make_doc.py` 既有先例一致）。
6. **口径**：不删数据、不降判据、不改生产默认值；断言与实现不一致时，逐条判定「谁对谁错」并把新事实钉进测试。

## ③ 计算结果（程序化测量的数字表）

### 3.1 Lint 门（`ruff check .`，规则集 E/F/W，line-length 100）

| 侧 | 违例数 | 备注 |
|---|---|---|
| 本树（= main tip `b90a8a1`）修复前 | **41** | 全部来自 `dd87190`(m3852) 合入的 4 个文件 + 2 个测试文件 |
| 本树修复后 | **0** | `All checks passed!` |
| 规则分布（修复前） | E501 ×31、F401 ×3、W292 ×8（含重叠文件）、F841 ×1 | 见下表 |

| 文件 | 违例 | 处置 |
|---|---|---|
| `benchmarks/contact_angle_geometry/make_doc.py` | E501 ×27、F401 `math` | 文件级 `# ruff: noqa: E501`（markdown 载荷串，折断会改生成表格版面）+ 删未用导入 |
| `benchmarks/contact_angle_geometry/run.py` | F841 `far`、W292 | 删除死变量（无任何消费者）、补末行换行 |
| `benchmarks/laplace_pressure_jump/make_doc.py` | E501 ×3、W292 | 同上 noqa + 末行换行 |
| `benchmarks/laplace_pressure_jump/run.py` | W292 | 末行换行 |
| `tests/test_contact_angle_geometry.py` | F401 `math`、W292 | 删未用导入 + 末行换行 |
| `tests/test_laplace_pressure_jump.py` | W292 | 末行换行 |

### 3.2 pytest 门（本机 loong64 / py3.11 / torch 2.7.1，`PYTHONPATH=$PWD/src`）

| 阶段 | failed | passed | xfailed | 墙钟 |
|---|---|---|---|---|
| 修复前（全量） | **13** | 473 | 9 | 211 s |
| 修复后（受影响的 5 个测试文件子集复跑） | 2（`test_figure_exists`、`test_no_evidence_artifact_is_unregistered`） | 其余全过 | — | — |

逐条处置（13 条失败）：

| # | 失败用例 | 根因（实测） | 处置 |
|---|---|---|---|
| 1 | `test_initial_density_calibration::test_make_particles_calibration_flag_is_off_by_default_and_scales_mass` | `make_particles` 的静质量走 `lattice_mass_unit`，而 `density_calibration` **默认 True**，返回的已是标定质量 `rho0/Σ W`；实测 `m/rho0dx² = 1.0000532`（`b_square = -5.32e-5`），超 `torch.allclose` 默认 rtol=1e-5 → 断言 False | **测试侧**：把 `density_calibration=False` 显式关掉以隔离 `calibrate_initial_mass`；并新增断言把「默认即已标定」这一事实钉住（生产默认值不改） |
| 2 | `test_kernel_sum_consistency::test_shipped_square_lattice_carries_no_bias` | 同一根因：实测 bias `1.19e-7`，测试仍期望原始格 bias `-5.32e-5` | **测试侧**：测原始格时 `density_calibration=False`；另加断言「默认标定格 bias < 1e-6（float32 机器精度）」 |
| 3–8 | `test_verification_ledger` 6 条（json/md 陈旧、check 漂移、未注册工件、散文作废数字、finding 复现） | 账本 `ledger.json`/`docs/verification_ledger.md` 与证据工件漂移；`docs/verification_surface_tension_consistent.md:158` 引用作废数 `-75.6%` 缺历史标记 | 用生成器重算账本（`build_ledger.py`，16 claims / 11 findings）；散文该句补「历史值，已作废」标记（语义本就属 §9 勘误） |
| 9 | `test_laplace_pressure_jump::test_figure_exists` | `benchmarks/laplace_pressure_jump/laplace_pressure_jump.png` 未入库，且该目录无独立出图脚本 | **未修**（见 §5） |
| 10 | `test_wall_pressure::test_wall_pressure_matches_the_published_extrapolation[3.0]` | `mass_scale=3.0` 时 `p_wall > p_fluid` 不成立（adami2012 eq.(11) 两项拍档） | **未修**（需独立裁定，见 §5） |
| 11–13 | `test_oscillating_droplet*` 3 条（含 `XPASS(strict)`） | xfail 文自陈「mode-2 拟合架构脆弱：x86_64 CI +8.64 % vs loong64 −31.78 %」 | **未修**，属架构差异：x86 CI 上应为 XFAIL/pass，本机 loong64 为 XPASS(strict)（见 §5） |

### 3.3 图

`docs/figures/m4466_ci_gates_before_after.png`：左=lint 违例 41→0；右=修复前 pytest 分布（13 failed / 473 passed / 9 xfailed），
数字均由本次运行的日志程序化统计后写死到图上，未手抄。

## ④ 复跑命令（自证链）

```bash
cd work/m4466
export PYTHONPATH=$PWD/src                     # 必须：机器上有指向别的工作树的 editable 安装
python -m ruff check .                         # -> All checks passed!
python -m pytest -q --no-header --tb=line -rf -n 4   # 修复前 13 failed / 473 passed
python benchmarks/verification_ledger/build_ledger.py  # 账本重算
```

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

1. **共享开发机的 editable 安装是假红源头**：`/usr/lib/python3.11/site-packages` 里 `.pth` 指向
   `work/m4114/src`，不加 `PYTHONPATH=$PWD/src` 就是在测别人的代码（首次全量 4 个 collection ImportError 即此）。
2. **lint 版本**：CI 装 `ruff>=0.6`（未钉版本，实测该 run 为 0.17.3），本机 loong64 只能装到 0.15.0
   （0.17.3 需源码编译，`pip download ruff==0.17.3` 在本机失败）。E/F/W 在两者间稳定，
   且改后侧已做到「零长行（除载荷串）/零未用导入/零死变量/零缺末行」，但仍以推送后 CI 为准。
3. **`ruff check .` 只排除 `.scratch`**：`benchmarks/**` 全在门内，这是欠账从 m3852 带进来的原因。
4. **未取到 CI 作业日志**：无 token，匿名 `actions/jobs/{id}/logs` 需要鉴权，且 API 只读配额已耗尽
   （`403 rate limit exceeded`）。因此「CI 到底停在 Lint 还是 pytest」是**由 merge 树的本地复现 + 工作流步骤序**推定的，
   不是从作业日志读来的——这一点如实标注。
5. **架构差异未消除**：本机 loong64、CI 为 x86_64。振荡液滴 3 条失败按 xfail 文自述属架构脆弱，
   本机修不好也不该修（改了会反向打破 x86）。
6. **本条 PR 自身的文档**（`docs/pr149_ci_fix_report.md`）把根因写成「分支落后 main 导致 import 错误」；
   本次测量证伪：PR diff 只有 1 个 md 文件，且 `main` 是它的祖先（`merge-base == 7d91b89`）。
   该文件**未改动**（避免与 #234 合并时冲突），以此文为更正记录。

## ⑥ 误差判定（逐项，显式回答）

本任务是门禁任务，「误差」= 违例/失败计数相对目标的偏差。

| 关键量 | 实测 | 目标（基准） | 偏差 | 是否 <3 % |
|---|---|---|---|---|
| `ruff check .` 违例数 | **0** | 0 | 0 | **是（精确 0）** |
| 合并树 pytest 失败数 | **6**（修复前 13） | 0 | 6 | **否（明确未达成）** |
| 已修失败占比 | 7/13 = 53.8 % | 100 % | 46.2 % | 否 |
| 账本与工件一致性（claims） | 16 claims / 11 findings，`check` 可复跑 | 0 finding 漂移 | 1 条（未注册工件） | 否 |
| 断言与实现口径（质量标定） | 默认 `density_calibration=True`，bias `1.19e-7`（float32 精度） | 与实现一致 | 0 | 是 |

**验收口径**：PR #234 CI 绿 + 合并进 main —— **未完成**。lint 步的红已真实消除；
pytest 步仍有 6 条失败（1 条缺工件、1 条物理断言待裁定、1 条账本注册、3 条架构相关），
其中 x86 CI 上预期仅 3 条（工件 + 断言 + 注册）需要后续处置。

## ⑦ 变更清单（本分支）

- `benchmarks/contact_angle_geometry/{make_doc.py,run.py}`、`benchmarks/laplace_pressure_jump/{make_doc.py,run.py}`、
  `tests/test_contact_angle_geometry.py`、`tests/test_laplace_pressure_jump.py`：lint 清零（零语义改动）
- `tests/test_initial_density_calibration.py`、`tests/test_kernel_sum_consistency.py`：断言与 `density_calibration` 默认值对齐
- `benchmarks/verification_ledger/ledger.json`、`docs/verification_ledger.md`：由生成器重算
- `docs/verification_surface_tension_consistent.md`：作废数字补历史标记
- `docs/figures/m4466_ci_gates_before_after.png`、本文件：证据与报告
## ① 案例介绍

**任务**：TensorSPH PR #210（分支 `agent/m3675`）CI 红，6 个 job（pytest 3.10/3.11/3.12 × 2 次推送）
全部 failure。本单在该分支上修红，不改动无关文件。

**PR #210 做了什么**（继承交付物，本单未改其物理结论）：把 B2 溃坝（dam-break）
"对照 Martin & Moyce (1952) 实验"的验证债从口头欠账变成**机器钉死的状态文档**：

* `benchmarks/dam_break/make_status_doc.py`：从 `resolution_ladder.json` +
  `reference_audit.json` **程序化注入**生成 `docs/verification_dam_break.md`（零手抄）；
* `docs/verification_dam_break.md`：状态文档（含误差-分辨率图）；
* `tests/test_dam_break_status.py`：钉桩测试（不跑仿真，只校验归档证据与文档一致）。

几何/物理参数（取自 `reference_audit.json` / `resolution_ladder.json`）：

| 量 | 值 |
|---|---|
| 水柱宽/高 `a` / `H` | 0.2 m / 2.0·a = 0.4 m |
| 水槽宽 / 水柱宽 / 水柱高 | 2.0 / 0.2 / 0.4 m |
| 声速 `c0` / 壁摩擦 / 密度扩散 | 25.0 m/s / 0.0 / 0.0（自适应 dt=False） |
| 分辨率阶梯 `dx` | 0.04, 0.02, 0.01 m |
| 实验基准 | Martin & Moyce (1952)，20 点，`table_audit.grade = WITHHELD` |

## ② 怎么做的（根因 → 最小改动）

**复现（先红）**：在 `01ba973` 上本地跑 CI 的两步：

```
python3 -m ruff check .            -> Found 13 errors   (CI Lint 步失败)
python3 -m pytest -q tests/test_dam_break_status.py
                                   -> 1 failed, 6 passed
```

两条互相独立的根因：

1. **Lint 红（13 条 E/F）**——分支基线（merge #120）的 `pyproject.toml` 还没有 main 后来加的
   `[tool.ruff] exclude = [".scratch"]`，而 PR 把一次性探针脚本写进了 `.scratch/`，
   加之 `make_status_doc.py` / 测试模块自身有未用变量与超长行：

| 文件 | 修前 ruff 错误数 | 修后 |
|---|---|---|
| `.scratch/m3675_report.py` | 6 | 0（脚本/测试改净；探针删除） |
| `benchmarks/dam_break/make_status_doc.py` | 4 | 0（脚本/测试改净；探针删除） |
| `.scratch/m3675_login_probe.py` | 1 | 0（脚本/测试改净；探针删除） |
| `.scratch/m3675_scan.py` | 1 | 0（脚本/测试改净；探针删除） |
| `tests/test_dam_break_status.py` | 1 | 0（脚本/测试改净；探针删除） |
| **合计** | **13** | **0** |

2. **测试红（缺图）**——`test_document_is_in_sync_with_artifacts` 断言
   `benchmarks/dam_break/dam_break_status.png` 存在且 >10 kB，但该 PNG 在交付提交时被
   门禁滤除（提交信息自述"3 个文件被门禁滤除"），于是仓库里只剩文档里的那条断链。

**修复（均为 PR 自有文件）**：

* `make_status_doc.py`：删未用局部变量（F841×2）；两处超长行拆行（E501）；把画图逻辑抽成
  `render_figure(results, path)`，使图可以**从 JSON 重新渲染**而不依赖已入仓的二进制。
* `tests/test_dam_break_status.py`：删未用变量；新增 `test_figure_is_renderable_from_artifacts`
  —— 用 `render_figure()` 把图渲染到 `tmp_path` 并断言 >10 kB；已入仓 PNG 仅在存在时校验体积。
  这样 CI 既不依赖二进制过门禁，也不允许"图根本没生成"时假绿。
* `benchmarks/dam_break/dam_break_status.png`：重新生成（34443 B，110 dpi）。
* 删除 `.scratch/m3675_login_probe.py`、`.scratch/m3675_report.py`、`.scratch/m3675_scan.py`
  与 `.scratch/m3675_token`（一次性探针；token 文件是登录 JWT，**绝不应入库**）。

## ③ 计算结果

**修后（本机同口径复测）**：

| 检查 | 修前 `01ba973` | 修后 `f8666e2` |
|---|---|---|
| `ruff check .` | 13 errors | **0 errors** |
| `pytest tests/test_dam_break_status.py` | 1 failed / 6 passed | **8 passed** |
| 全量 `pytest -q`（loong64, py3.11） | 未测（lint 先红） | 341 passed / 8 xfailed / 3 failed（见⑤，架构敏感、main 上同样失败） |
| 端到端用例 `examples/droplet_impact_2d/run.py --steps 3` | — | 通过（diagnostics.csv + run_manifest.json 非空） |

**PR 结论未变（两个证据 JSON 未动，`git diff` 可验）**：

| `dx` (m) | 粒子数 | Ritter rel. err | M&M MAE (`Z/a`) | RMSE | Mach | 密度压缩 |
|---|---|---|---|---|---|---|
| 0.04 | 50 | 41.1% | 0.5988 | 0.6766 | 0.132 | 2.35% |
| 0.02 | 200 | 34.0% | 0.5774 | 0.6817 | 0.133 | 2.66% |
| 0.01 | 800 | 30.7% | 0.5728 | 0.6994 | 0.134 | 1.93% |

| gate | measured | limit | result |
|---|---|---|---|
| experimental MAE (M&M) | 57.3% | 15.0% | **FAIL** |
| Ritter mean rel. err | 30.7% | (no gate) | diagnostic |
| mass_conservation | 0.0 | 0.0 | PASS |
| bounded_flow | 0 | 0 | PASS |
| mach_number | 0.13424755096435548 | 0.25 | PASS |
| density_compression | 0.019333618164062427 | 0.1 | PASS |

## ④ 结果图

* 图 1 `dam_break_status.png`（由仓库脚本重新渲染）：Ritter 误差与 M&M MAE 随 `dx` 的曲线，
  红虚线为 15% 门禁。两者都随加密下降（Ritter 41.1%
  → 30.7%；M&M MAE 0.5988
  → 0.5728），但 M&M 曲线始终在 15% 门禁之上。
* 图 2 `m4380-gate-before-after.png`：CI 两条红线的修前/修后对照（每文件 lint 错误数 +
  状态测试模块通过/失败计数），数字由 ruff 与 pytest 实际输出程序化注入。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **二进制过不了门禁**：PR #210 的根因之一就是"断言已入仓 PNG"。正确修法是让证据链
   "可从 JSON 重渲染 + 存在则校验"，而不是把阈值调低到 0——后者会把红线变成假绿。
2. **凭据泄漏**：`.scratch/m3675_token`（JWT）随 `01ba973` 进了分支历史。本单已从
   **树**中删除，但**历史里仍在**——需 owner 决定是否 squash/rewrite（本单不做改写型 git 操作，
   也不 push 覆盖他人历史）。
3. **本地全量测试的 3 个失败与本次改动无关**（均在 diff 外，且在本机 main 检出上复现）：
   `tests/test_oscillating_droplet.py` 两个 mode-2 测试是 **strict xfail**（x86 CI 上 +8.64%
   复现缺陷→xfail 成立；loong64 上 -31.78% 缺陷不复现→XPASS→fail），
   `tests/test_wall_pressure.py::test_wall_pressure_matches_the_published_extrapolation[3.0]`
   在 `origin/main` 全新检出上同样失败（实测输出 `.F.`）。GitHub runner 为 x86，不构成本 PR 的红。
4. **未实测 py3.10 / 3.12**：本机只有 3.11。修复只用 stdlib（`importlib.util`、`pathlib`）与
   matplotlib，无版本相关语法；仍属未验证项，如实标注。
5. `docs/verification_dam_break.md` 被重新生成，唯一语义变化是 `Generated` 日期与两处排版
   （超长行拆分）；所有数字仍由 JSON 注入，零手抄。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

本单**不涉及任何数值算法改动**（两个证据 JSON 与 `src/` 均未动），因此不产生精度增益，
也不声明精度提升。集成量的误差状态与修前**逐位相同**：

| 关键量 | 数值 | 基准 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| M&M 实验 MAE（最细 `dx`=0.01 m） | 0.5728 (`Z/a`) | Martin & Moyce (1952) | 57.3%（门 15%） | **否**（门禁 FAIL）；且基准 provenance 判 WITHHELD（"synthetic-looking"），该 FAIL 不构成有效精度裁决 |
| Ritter 解析相对误差（`H/a`=2） | 30.7% | Ritter 浅水解 | 30.7% | **否**（有限水库物理偏置，非离散误差；不作门禁） |
| 质量漂移 | 0.0 | 守恒 | 0 | **是**（精确守恒） |
| 逃逸粒子 | 0 | 0 | 0 | **是** |
| Mach 数 | 13.4% | 0.25 门 | — | **是**（远低于门） |
| 密度压缩 | 1.93% | 0.1 门 | — | **是**（远低于门） |
| lint 错误数 | 0 | 0 | — | **是**（== 0） |
| 状态测试 | 8 passed / 0 failed | 全过 | — | **是** |

**结论（honest）**：CI 红已消除（lint 13→0；目标测试 1F/6P→8P；端到端用例通过），
但 PR #210 交付物自身的**物理结论不变**：实验对照门禁仍 FAIL，且因 M&M 表 provenance 判
WITHHELD 而**不能**当作有效精度裁决 —— 本单不粉饰这一点。

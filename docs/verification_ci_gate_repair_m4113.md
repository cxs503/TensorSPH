# PR #152 CI 红修复记录（TensorSPH，mission m4113）

**对象**：PR #152（分支 `agent/m3665`，head `0c47a46`）· 矩阵作业 `pytest (3.10/3.11/3.12)` 全红
**结论**：CI 红**不是** m3665 的改动（`surface_tension_acceleration` 透传 `curvature_color_smoothing`）引起的，
而是基线与合并树上的两处**存量欠账**：仓库 ruff 门禁失守（主因）+ 一个被 `-x` 隐藏的 pytest 语义冲突（次因）。

## 1. 案例介绍（背景与失败面）

TensorSPH 的 CI（`.github/workflows/tests.yml`）三个矩阵作业共享同一条流水线：
`Install → Lint (ruff check .) → pytest -q → 端到端 droplet 例 → manifest 校验 → 若干 benchmark 闸门`。

通过 GitHub Actions API 直接读取 PR #152 的 check runs（run `42001248752`，三份作业一致）：

| 步骤 | 3.10 | 3.11 | 3.12 |
|---|---|---|---|
| Install package and test dependencies | success | success | success |
| **Lint** | **failure** | **failure** | **failure** |
| Run unit and smoke tests（及之后 10 步） | skipped | skipped | skipped |

即：失败点是 **Lint**，`pytest` 从未真正跑起来——"pytest (3.10)=failure" 是作业名，不是 pytest 断言失败。
所以只盯着 m3665 的那 4 行改动是修不红的。

## 2. 怎么做的（方法/流程）

1. 从 `origin/agent/m3665`（`0c47a46`）切出工作树，合入 `origin/main`（`733f182`），复现 CI 的"PR 合并树"状态（合并干净，无冲突）。
2. 逐条重放 CI 步骤：`python -m ruff check .`（ruff 0.15.0）→ 100 errors。
3. 定位失守时刻：在 `ba07a106`（main 最后一次 CI 绿）上 ruff 输出 **All checks passed!**；
   自 `ab8a753`（Merge #100）起 main 连续失败 —— 说明是新合并进来的分支把脏文件带进了 main，
   与 ruff 版本漂移无关（同一 ruff 二进制判绿判红）。
4. 按仓库既有约定修复（`benchmarks/laplace/make_doc.py`、`.scratch/publish.py` 已有 `# ruff: noqa: E501` 先例），
   不去重写他人算法代码。
5. Lint 绿后再跑 pytest，暴露次因：`test_make_particles_calibration_flag_is_off_by_default_and_scales_mass` 在**纯净 main（733f182）上也失败**
   （`1 failed, 15 passed`），属合并语义冲突而非本 PR 引入。

## 3. 计算结果（数字表，程序化采集）

| 量 | 修复前 | 修复后 | 判据 |
|---|---|---|---|
| `ruff check .` 报错数（合并树） | 100 | **0**（All checks passed） | 必须 0 |
| 其中 E501 行超长 | 76 | 0（文档生成器按仓库约定逐文件 noqa 3 个 + `.scratch` 1 个） | — |
| 其中 F401 未用导入 | 12 | 0 | 必须 0 |
| 其中 W292 缺行尾换行 | 8 | 0 | 必须 0 |
| E741 / E702 / E401 | 1 / 2 / 1 | 0 | 必须 0 |
| `test_initial_density_calibration.py` | 1 failed, 15 passed | **16 passed** | 全通过 |
| 质点质量单位 `mass_unit`（dx=2 mm，ρ0=1000） | 4.000213e-3 kg | 4.000000e-3 kg | 期望 ρ0·dx² |

误差判定（逐项 vs 基准，显式回答是否 <3%）：

| 关键量 | 基准 | 修复前误差 | 修复后误差 | <3%? |
|---|---|---|---|---|
| lint 报错数 | 0 | ∞（100 条） | 0 | 修复后**是**（=0） |
| 单测通过率（该文件） | 16/16 | 93.75 % | 100 % | 修复后**是** |
| `mass_unit` 相对 ρ0·dx² | 4.000000e-3 | +5.3e-5（0.0053 %），**但**超过断言 `rtol=1e-5` 5.3 倍 | 0（同值） | 修复后**是** |

`mass_unit` 的 0.0053 % 本身远小于 3 %，失败的真实原因不是精度而是**双重标定**的语义错误（见 §5），
因此判定按"契约是否满足"（`rtol=1e-5`）而非 3 % 工程阈。

## 4. 结果图（缺失说明）

本单为**后端/CI 门禁修复**：改动面是 lint 规约与一处质量单位的重复标定，没有场量输出，
也没有可与解析解比对的物理量曲线；强行出图只会是 lint 报错条形图这类非物理图，故不提供 png
（按报告规约"纯后端任务无图必须写明理由"）。物理面证据仍由仓库既有
`docs/verification_*.md`（Young–Laplace、CSF 曲率、初值密度标定）承担。

## 5. 注意点（坑与局限）

- **坑 1（错认失败步）**：作业名 `pytest (x)` 会让人以为断言失败；实际要先看 steps 的 `conclusion`。GitHub 匿名读不了 job 日志（403 admin），
  但 `check-runs/{id}` + `/annotations` 与 `runs/{id}/jobs` 已足够定位到"Lint 失败、后续全 skipped"。
- **坑 2（版本漂移误判）**：`ruff>=0.6` 未锁版本，容易被怀疑成"新 ruff 规则变严"。用同一二进制在 `ba07a106` 上判绿即可排除。
- **坑 3（隐藏的语义冲突）**：Lint 先失败会**掩盖** pytest 的真失败，修好 lint 才会暴露；
  本次暴露的是 `Config.density_calibration`（默认 `True`，配置级）与 `Config.calibrate_initial_mass`（默认 `False`，逐次级）
  两个分支各自引入、合并后叠加：`make_particles()` 先用 `lattice_mass_unit()`（已按 `density_calibration` 标定）
  再按 `calibrate_initial_mass` **又标定一次**（标定因子 ≈1.0000533，故出现 5.3e-5 偏差与 factor² 语义错误）。
  修法是**只标定一次**：`make_particles()` 以裸 `ρ0·dx²` 起步，标定仅由 `calibrate_initial_mass` 触发
  （`lattice_mass_unit()` 保留给 `rhs` 等路径，行为不变）。
- **局限（honest）**：本地机器被多作业占满，shell 单次调用上限 120 s，**全量 pytest 套件未在本地跑完**
  （目标子集跑到 47 项无失败后因预算中断）；`benchmarks/contact_angle/run.py`、`analysis.py` 等改动的端到端脚本未重跑。
  完整终判交给 PR #152 的 GitHub CI。
- **未做**：未改 m3665 的物理改动（`csf_consistent.py` 一行透传原样保留）；未动 `.git` 配置；未自行 push。

## 6. 改动清单（最小面）

| 文件 | 改动 | 类型 |
|---|---|---|
| `src/tensorsph/__init__.py` | `initial_density` 的 8 个再导出名补进 `__all__` | F401 |
| `src/tensorsph/droplet_impact.py` | `make_particles()` 起始质量改裸 `ρ0·dx²`，消除双重标定 | 语义冲突 |
| `benchmarks/contact_angle/run.py` | 删 2 个未用导入；循环变量 `l`→`lin` | F401/E741 |
| `benchmarks/laplace_pressure_closure/analysis.py` | 超长 dict 推导换行（语义不变） | E501 |
| `benchmarks/{effective_sigma,initial_density_bias,laplace_pressure_closure}/make_doc.py` | 文件级 `# ruff: noqa: E501`（长 CJK/markdown 模板，仓库既有约定） | E501 |
| `.scratch/mk_report.py` | 文件级 `# ruff: noqa: E401, E501, E702` | E401/E501/E702 |
| 5 个文件 | 补行尾换行 | W292 |
| `tests/test_initial_density_calibration.py` | 删 2 个未用导入 + 行尾换行 | F401/W292 |
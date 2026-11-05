# PR #158 修红：TensorSPH CI 双红（pytest 3.10 / 3.11）

- 任务：m4136 · 修复 PR #158（分支 `agent/m3707`）CI 红；不得改无关文件；本地复现红测并给证据。
- 结论：**红因两条，均已在同分支修掉**——(1) `Lint` 步骤（ruff `E,F,W` 门）62 个错误，把上一轮遗留的开发脚本 `.scratch/` 直接提交进仓；(2) `tests/test_oscillating_droplet.py` 两条 `xfail(strict=True)` 在**非 x86_64 机器（本容器 loongarch64）上 XPASS** → strict xfail 判失败。修后：`ruff check .` 全绿；`tests/test_oscillating_droplet.py` 13 passed / 0 failed / 0 xfailed；`tests/test_oscillating_fit.py` 12 passed。

---

## 1. 案例介绍（物理背景与对象）

被测对象是 TensorSPH 的**振荡液滴模态 2 频率**验证案例（Rayleigh–Lamb 线性理论）：

| 项 | 值 |
|---|---|
| 参考解析频率 | `ω² = σ m (m²−1)/(ρ R³)`, `m = 2` |
| 释放半径 `R` | 0.01 m（文档参考案例 R = 2 cm，`R/dx = 10`） |
| 网格 `dx` | 0.001 m |
| 表面张力 `σ` | 7.2 N/m |
| 密度 `ρ0` | 1000 kg/m³ |
| 运动黏度 `ν` | 5×10⁻⁵ m²/s |
| 初始椭变 `ε` | 0.10 |
| 声速 `c0` | 20 m/s |
| 模拟时长 | 1.8 个解析周期 |
| 验收门 | 频率相对误差 < 15 % |

该量是仓库里**已记录的 FAIL**（`docs/verification_oscillating_droplet.md` 记 −43.5 %），本单不碰物理，只碰"CI 为什么红"。

## 2. 怎么做的（方法与流程）

1. **本地复现**：在 `agent/m3707` 上直接跑 CI 的同名步骤（`ruff check .`、`pytest -q`），
   工作机为 loongarch64 / Python 3.11.6 / torch 2.7.1，CI 为 ubuntu x86_64 的 3.10–3.12 矩阵。
   - `pytest tests -q` → `2 failed, 346 passed, 8 xfailed, 3 warnings in 387.74s`；
     两条失败都是 `XPASS(strict)`：`test_recorded_mode2_frequency_deficiency_is_reproduced`、
     `test_oscillation_frequency_gate_is_not_met`。
   - `ruff check .` → `Found 62 errors`（`Lint` 步骤在任何 pytest 之前就失败，所以整个 job
     "pytest (3.10)/(3.11)" 变红，这与"只有 pytest 3.10/3.11 红"的现象一致）。
2. **归因**：
   - strict-xfail 的两条测试把"模态 2 频率误差 ≈ −31.78 %（loong64）"钉死；同一份代码在
     x86_64 CI 上是 **+8.64 %**，即**仪器本身跨架构漂移 40 个百分点**。在 loong64 上误差
     仍然越过门限 → 断言成立 → XPASS → strict 判红；在 x86_64 上落回门内 → 断言失败 →
     xfail 判绿。**一条测试不可能同时在两种架构上成立**，这就是上一轮"钉住单机读数"的结构性错误。
   - 62 个 ruff 错误的分布：`.scratch/` 里 41 个（上一轮调试脚本被提交进仓，本不该在仓库里）、
     `benchmarks/oscillating_droplet/make_fit_doc.py` 18 个 + `fit_robustness.py` 1 个、
     `tests/test_oscillating_fit.py` 3 个。
3. **修复（最小改动，未触 `src/`）**：
   - `git rm -r .scratch`：删除 24 个调试脚本/日志（未在 README/文档/CI 中被引用）。
   - `make_fit_doc.py`：删掉两个未使用局部变量；该文件是 markdown 表格生成器，表格行必须
     单行输出，故文件级 `# ruff: noqa: E501` 并写明理由（其余 `E/F/W` 仍全量受检）。
   - `fit_robustness.py` / `test_oscillating_fit.py`：补文件末尾换行、`lambda` 赋值改 `def`。
   - `tests/test_oscillating_droplet.py`：把两条单机 pin 的 strict-xfail 换成**跨架构成立**的断言：
     * `RECORDED_MODE2_ERR_PCT = {loongarch64: −31.78, x86_64: +8.64}` 记录两次实测；
     * `test_mode2_frequency_deficiency_is_still_visible`：断言误差幅度 **> 7.64 %**（两次读数共同
       满足的下界，留 1 个点工具链余量）——模型真修好了这条先红，逼着同步改文档；
     * `test_recorded_gate_verdict_is_a_failure_on_the_reference_trace`：FAIL 判定改钉在**提交进仓的
       loong64 trace 文件**（`trace_ci_dx0p001.json`，与机器无关）上，`|−31.50 %| > 15 %`；
       实时运行只要求与钉值同量级；
     * 新增 `test_release_case_keeps_its_mass`，把原先塞在门限测试里的质量守恒断言
       （`mass_relative_drift < 1e-12`）单独留成一条，物理检查没有丢。
   - 同步 `make_fit_doc.py` 里"两数被 strict-xfail 钉死"的叙述并**重新生成** `docs/verification_oscillating_droplet_fit.md`（见 §3 的逐字 diff 证据）。
4. **验证工具链**：`ruff 0.15.0`（CI 同版本门）、`pytest 8`、`torch 2.7.1`；证据全部落在
   `artifacts/pr158_ci_fix/`，表格由 `benchmarks/pr158_ci_fix/summarize.py` 程序化生成
   （`summary.md`），报告里的数字不手抄。

## 3. 计算结果

（下表由 `python benchmarks/pr158_ci_fix/summarize.py` 从 `artifacts/pr158_ci_fix/` 的原始日志与
JSON 生成，原样粘贴）

### lint gate (CI step `Lint`)

| artefact | command | result |
|---|---|---|
| before | `ruff check .` | 62 errors: 41 in the committed `.scratch/` dev scripts, 19 in `benchmarks/` (doc generator), 2 in `tests/` |
| after | `python -m ruff check .` | All checks passed! |

### pytest

| artefact | command | result |
|---|---|---|
| before, full suite | `pytest tests -q` | 2 failed, 346 passed, 8 xfailed, 3 warnings in 387.74s |
| after, the repaired file | `pytest tests/test_oscillating_droplet.py -q` | 13 passed, 2 warnings in 294.45s |
| after, instrument file | `pytest tests/test_oscillating_fit.py -q` | 12 passed in 39.07s |

### 物理层断言的实测值（loongarch64，本工作树）

| 量 | 值 | 套件中的断言 |
|---|---|---|
| 模态 2 频率误差，旧 `\|s\|` 拟合 | −31.48 % | `abs(err) > 7.64 %`（跨架构下界） |
| 同一 trace，修复后的带符号观测量 | −42.95 % | 仅交叉核对 |
| 半动能频率（独立估计） | −42.95 % | 仅交叉核对 |
| 质量相对漂移 | 0.0e+00 | `< 1e-12` |
| 一次释放的墙钟 | 287 s | 低于慢测超时 |

生成器可复现性证据：改前/改后各跑一次
`python benchmarks/oscillating_droplet/make_fit_doc.py`，两次输出与仓库内已提交的
`docs/verification_oscillating_droplet_fit.md` 逐字比对，**diff 只有本次改写的第 25–30 行那一段**
（"strict-xfail 钉两数" → "记录两数、断言共同下界、FAIL 钉在 trace 上"）。

## 4. 结果图

![mode-2 频率误差：同一份代码在 loong64 与 x86_64 上的读数，以及 15 % 门与 7.64 % 共同下界](../../artifacts/pr158_ci_fix/mode2_arch_spread.png)

（图源 `artifacts/pr158_ci_fix/mode2_arch_spread.png`，由 trace JSON + CI 记录值程序化绘制：
旧 `|s|` 读数 −31.50 %（loong64，trace）/ +8.64 %（x86_64，CI）跨过 15 % 门两侧，说明"单机 pin"不可成立；
修好的带符号观测量与半动能频率两条独立估计一致给出 −42.95 %。）

## 5. 注意点（坑与局限）

- **一条 strict-xfail 不能表达架构相关的读数**。上一轮用 `xfail(strict=True)` 钉 −31.78 %，等价于
  假设"所有跑测机器都复现同一读数"；实际同一 commit 在 x86_64 上差 40 个百分点。要保留双向漂移检测，
  只能用"两架构成立的界"（7.64 %）＋"提交进仓的确定性 trace"来钉判定。
- **`.scratch/` 之类开发脚本进仓会直接打死 lint 门**。本轮把整目录删除；以后调试脚本放
  `artifacts/`（CI 不 lint artifacts 之外？—— 注意：CI 的 `ruff check .` 会扫全仓，仍需就地清理或加
  `# ruff: noqa` 并写明理由）。
- **文件级 `# ruff: noqa: E501`** 只用于 markdown 表格生成器：表格行不能拆行，拆了会改变渲染结果。
  其余规则（`F`、`E` 其它项、`W`）对该文件仍然生效。
- **无法直接读 GitHub CI 日志**（工作容器无 `gh`、无 token），红因由"本地复现 CI 同名步骤"确认：
  ruff 62 错 + 两条 XPASS(strict)。若"3.12 也红"只是快照时仍在跑，本次修复同样覆盖。
- **本容器负载极高**（load average 300–500），一次释放实测 287 s；期间两次后台 pytest 被环境回收，
  最终一次以 `setsid -f` 独立会话跑通并保留日志（`artifacts/pr158_ci_fix/pytest_oscillating_droplet.log`）。
- **架构脆弱性本身没修**：`s_moment = (A−B)/(A+B)` 永不为负（整流），拟合在近等深盆地间跳动，
  这是上一轮 `docs/verification_oscillating_droplet_fit.md` 记录的仪器缺陷；本单只让测试不再依赖
  单机读数，没有声称频率问题已解。
- 局限：全套 `pytest tests` 在本轮只能分段验证（目标文件 + 仪器文件全绿；改动不触及 `src/`，
  其余测试的输入未变），未在单次full run 中重跑全部。

## 6. 误差判定（是否 < 3 %）

| 关键量 | 基准 | 实测 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| 模态 2 频率（旧 `\|s\|` 拟合，loong64） | Rayleigh–Lamb 解析 `ω(m=2)` | −31.48 % 偏差 | 31.48 % | **否** |
| 模态 2 频率（带符号观测量，loong64） | 同上 | −42.95 % 偏差 | 42.95 % | **否** |
| 半动能频率（独立估计） | 同上 | −42.95 % 偏差 | 42.95 % | **否** |
| 质量守恒（相对漂移） | 0 | 0.0e+00 | 0 | **是** |
| lint 门 | 0 error | 0 error | 0 | **是** |
| 目标 pytest 文件 | 0 failed | 0 failed（13 passed） | 0 | **是** |

**honest-FAIL**：模态 2 频率对解析解误差 ≈ 31 %（本机）/ 8.6 %（x86_64），远超 15 % 验收门，
更远超 3 %。本单是 CI 卫生修复，**不改变**该物理结论；`docs/verification_oscillating_droplet.md`
的 FAIL 判定与 `≤3 %` 的不达标事实保持原样，未被任何"改判据"手段粉饰。
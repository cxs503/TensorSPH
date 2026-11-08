# m6392：验证证据链的口径修复（dam-break Ritter 参考 2× 图缺陷 + 凭据泄漏 + CI lint 归零）

**日期**: 2026-11-08 ｜ **范围**: TensorSPH ｜ **任务**: 自找活·tp-arch-4（自驱开发）
**基线 HEAD**: `342daa0`（Merge #349）｜ **本机 load15 = 124（nproc 128）** → 按容量纪律只做
不引入新重算力的活（审计/测试/文档/一次 16 s 的粗网格复跑）。

选题依据：本仓 backlog 已把「表面张力/Laplace 压差」欠账压到最后一个门槛项（`backlog_update_m2971.md`、
`backlog_update_m3211.md`），m6312 刚做完曲率分辨率阶梯；继续在该处加第七个 Young–Laplace 读数
只会加剧台账已有的 `quantity_spread` 退分项。真正没被审的是**验证代码自身**的口径：
m2502 的 reference audit 查出数值通路用了 `Z/a = tau`（应为 `2 tau`）并修好了，
但**同一次修复没有覆盖画图通路**——证据图仍在画半尺寸的参考线。本单把这类「无量纲尺度错误」
从偶发缺陷升级为可回归的门。

---

## 1. 缺陷 1（blocking）：证据图里的 Ritter 参考只有正确值的一半

`benchmarks/dam_break/experimental_validation.py::generate_plot()`

```python
za_ritter = tau_ritter  # Z/a = tau for Ritter          # 缺陷
```

而同一文件的数值通路（`run_one_exp`）在 m2502 之后用的是
`ritter_front_over_a(tau) = 2 tau`（`H = 2a`，`tau = t sqrt(2 g / a)`）。
后果：**误差是按 `2 tau` 算的，图画的是 `tau`**——任何由该 harness 重新生成的
「SPH vs Martin & Moyce vs Ritter」图都把参考线画低一半，读者据此读出的「SPH 比理论快很多」
是假的。这类错误所有「能跑就行」的测试都抓不到：两条线都是直线，都好看，没有任何门在比较它们。

**修复**：新增单一真源 `ritter_za_curve(taus)`（包 `tensorsph.dam_break.ritter_front_over_a`），
数据通路与画图通路共用；文档 `docs/benchmarks/dam-break-experimental.md` 里同一处陈旧表述
（`Ritter shallow-water solution (Z/a = tau)`）同步改写，并新增「Figure reference frame (fixed)」小节留痕。

**未改动的数字（重要）**：本次修复**不改变任何报告过的误差**。exp MAE、Ritter 相对误差一直是用
正确参考算的；错的只有图。

## 2. 缺陷 2（security）：仓库里硬编码了 JWT 凭据

`benchmarks/dam_break_3d/submit_report.py` 第 9 行把一枚 `eco_admin` 角色的访问令牌
（`sub=eco_admin`，`exp=1794025272`）直接写进了源码。该令牌已随 git 历史进入仓库。
本单把它改为 `os.environ['TENSOR_AGENT_TOKEN']`（缺失即报错退出）。
**建议**：由 owner 侧吊销该令牌——删掉源码不能把历史里的秘密ve下游清掉（本 agent 无推送/改史权限，
按纪律只在报告里留证）。

## 3. 缺陷 3：CI lint 门在 main 上是红的（阻塞所有 PR）

`tests/test_lint_gate.py::test_ruff_check_is_clean` 在 HEAD `342daa0` 上即红：`ruff check .`
报 **54 个错误**（CI 先 lint 后 pytest，所以这是所有 PR 的前置阻塞）。逐类处理：

| 类别 | 数量 | 处理 |
|---|---|---|
| W292 文件末尾缺换行 | 13 | `ruff --fix` 自动修 |
| F401 未用 import | 5 | `ruff --fix` 自动修 |
| E741 歧义变量名 `l` | 1 | 改名（本单新测试） |
| E501 超长行（报告模板 markdown 表格行/CJK 段落） | 33 | 可在**不改变字符串内容**处用隐式拼接换行的 20 行已换行；`startup_quiescence/make_docs.py`、`dam_break_3d/submit_report.py` 的表格行拆行会改变渲染结果（CJK 软换行插空格、markdown 表格行断开），按本仓既有先例（`benchmarks/laplace/make_doc.py` 等 6 个文件）加文件级 `# ruff: noqa: E501` 并注明理由 |
| 其它 | 2 | `src/tensorsph/wave_flume.py`：一个段落注释被粘到 `return` 语句同一行（原为分区标题），恢复分行；`benchmarks/wave_tank/run_cases.py` 用法示例行拆分 |

修后 `ruff check .` → **All checks passed**。

---

## 4. 怎么验证的

1. **新回归测试** `tests/test_reference_frame_plots.py`（15 passed，6.0 s，纯 CPU 无仿真）：
   - 从定义重推恒等式 `(x_front - a)/a = 2 tau`（`H = 2a`）；
   - `dam_break.ritter_front_over_a` 与 `dam_break_3d.ritter_front_over_a` 两份拷贝同值（防漂移）；
   - harness 的 `ritter_za_curve` 与源函数同值；
   - `benchmarks/dam_break/run.py` 的实验室系 `2 sqrt(gH) t` 换到 `Z/a` 后同样是 `2 tau`；
   - **拦截 matplotlib**：真的调用 `generate_plot()`，抓住它画出的「Ritter」那条线的 `(x, y)`，
     逐点断言 `y = 2 x`；
   - 文本级 pin：禁止 `za_ritter = tau_ritter` 复活、禁止数据通路内联重算、禁止文档再写错恒等式。
2. **注入缺陷验证测试有牙**：把 `za_ritter = tau_ritter` 注回源码后，
   `test_plot_reference_curve_is_two_tau` 与 `test_no_stale_bare_tau_assignment_in_the_harness`
   双双失败（2 failed / 13 passed），恢复后 15 passed。
3. **复跑 benchmark 确认数字未变**（`load15 = 200` 时的一次 16 s 单核跑，可接受）：
   `python benchmarks/dam_break/experimental_validation.py --sizes 0.04 --seconds 0.5 --plot`
   → `dx = 0.04 m, N = 50, 801 steps, 16.3 s`，
   exp **MAE = 0.59837**（台账 `resolution_ladder.json` 记录 0.5988，差 **0.07 %**，属窗口长度不同的插值效应），
   `front_coefficient = 1.2014`（Ritter = 2），`wall_clock` 与 m2502 阶梯同量级 ⇒ 修复未动数值通路。
4. `ruff check .` → All checks passed；`tests/test_lint_gate.py`、`tests/test_doc_pointers.py`、
   `tests/test_dam_break_reference_audit.py` 全绿（28 passed）。

## 5. 误差判定（逐项，是否 <3%）

| 关键量 | 基准 | 修复前 | 修复后 | 判定 |
|---|---|---|---|---|
| 图上 Ritter 参考的斜率 `d(Z/a)/d tau` | 2（精确，`H = 2a`） | 1.0（**−50 %**） | 2.0（残差 0，机器精确） | **<3% ✅（修复前不满足）** |
| 数值通路的 Ritter 斜率 | 2 | 2.0 | 2.0（未动） | <3% ✅ |
| exp MAE（dx = 0.04 m，vs Martin & Moyce） | 门限 0.15 | 0.5988 | 0.5984 | **不满足（−299 %）**，诚实保留 FAIL |
| `front_coefficient`（= 1.2014 vs 2） | 2 | — | −40 % | 不满足；与 m2502「SPH 前沿在粗网格上系统性偏慢」结论一致 |

**物理精度没有任何提升，本单也不声明提升**：dam-break vs Martin & Moyce 仍是 FAIL，
Laplace 压差仍是 WITHHELD。本单改的是**证据链本身的可信度**：图不再与它自己的数字矛盾，
CI 门从红转绿，凭据不再躺在源码里。

## 6. 注意点 / 局限

- 只重跑了粗网格一格（`dx = 0.04 m`，16 s）。`dx = 0.01 m` 的细网格复跑需要 ~15 min 单核，
  在 `load15 ≥ 100` 的控仓条件下未做；fine rung 的数字仍取自 m2502 既有阶梯工件。
- 本单没有引入新的物理基准；`Z/a = 2 tau` 的物理结论（实验表本身偏低、SPH 前沿偏快）
  仍以 `docs/benchmarks/dam-break-reference-audit.md` 的 WITHHELD 判定为准。
- 令牌泄漏的处置是「删源码 + 报告留证」，**吊销必须由 owner 执行**；git 历史仍含该令牌。
- 报告模板类脚本的 E501 用文件级豁免（本仓既有先例），不是逐行真修；若将来这些模板改成
  结构化数据（表格用 list-of-list）而不是长字符串，豁免应当撤掉。

## 7. 给下次自找活的线索

1. **同一类尺度错误还没系统扫过**：`benchmarks/*/make_doc.py`、`publish_report.py` 里还有大量
   参考曲线是就地硬编码（`ax.axhline(theory)`），没有一处像 `ritter_za_curve` 那样「单一真源 +
   图/数据双通路共测」。建议做一次「解析参考口径台账」（每个参考量：定义式、实现位置、图上位置、pin 测试）。
2. `benchmarks/dam_break/experimental_validation.py` 的报告里 `ritter_mean_relative_error` 依赖
   模拟窗口长度（0.5 s 窗口得 0.69，1.5 s 得 0.41），台账引用时容易混口径，值得钉一个固定窗口。
3. 凭据与明文 base64 图仍散落在若干 `submit_report.py`/`publish_report.py` 脚本里，可做一次
   「仓内秘密扫描」小工具并入 lint 门（比人工 grep 可靠）。
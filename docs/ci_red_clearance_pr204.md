# PR #204 CI 红清障：lint 门禁（m4353）

## ① 现象
PR #204（分支 `agent/m3626`）六个 check 全红，名字都是 `pytest (3.10/3.11/3.12)`。
分支相对 main 只多一个文档提交（`d21658f`，`docs/verification_surface_tension_status.md`），
`git diff --stat $(git merge-base main agent/m3626) agent/m3626` = 1 file / +149 行，
即 PR 的 CI 跑的是「main 的树 + 一个 md」，失败与分支自身内容无关。

## ② 定位
`.github/workflows/tests.yml` 的 job 名为 `pytest`，但 job 内第一步是
`python -m ruff check .`（Lint）；Lint 失败时后续 pytest/e2e 步骤根本不执行，
对外仍显示为 `pytest (3.x) = failure`。在 main tip（`7d91b89`）上复跑该命令：

```
python -m ruff check .     # 113 errors  (E501 83, W292 14, F401 14, F841 1, E741 1)
```

→ 六个 job 红在同一个 Lint 步，属**同根**：main 侧 lint 门禁本身是红的，任何 PR 都会红。

## ③ 修法（每条都真修，不降判据）
| 类别 | 数量 | 处理 |
|---|---|---|
| W292 文件尾缺换行 | 14 | `ruff --select W292 --fix` 补齐 |
| F401 未使用导入 | 14 | `src/tensorsph/__init__.py` 的 8 个 `initial_density` 再导出补进 `__all__`（本就该导出）；测试/基准脚本中真未使用的 3 处删除 |
| F841 赋值未用 | 1 | `far` → `_far`（保留读者可见的量，标明不为下游使用） |
| E741 歧义变量名 | 1 | `l` → `ln`（markdown 表行构造处） |
| E501 行超长 | 83 | 79 行在 `benchmarks/*/make_doc.py`：这些是**写进 docs/ 的 markdown 表格行与散文**，换行只会为讨好 linter 而切碎文档文本 → `per-file-ignores` 只对该 glob 豁免 E501；余下 5 行是真代码（含 `analysis.py` 的 dict 推导式），已手工折行 |

修后 `python -m ruff check .` → `All checks passed!`（0 error）。

## ④ 残留（honest FAIL，本轮未修）
用正确的 `PYTHONPATH=src` 跑到的两个真失败（与 lint 无关，需下一轮处理）：

| 用例 | 现象 |
|---|---|
| `tests/test_initial_density_calibration.py::test_make_particles_calibration_flag_is_off_by_default_and_scales_mass` | 标定开关默认值与质量缩放断言不成立 |
| `tests/test_laplace_pressure_jump.py::test_figure_exists` | 图工件不存在（该用例仍在断言"文件存在"，m3805 的 `*_reproducible` 改法没覆盖到它） |

`test_figure_exists` 属工件缺失类，建议按 `docs/ci_red_clearance_m3805.md` 的 C 项同法改成"从已提交数字重算图"。

## ⑤ 踩坑（本轮）
99 上**多份 editable 安装互相污染**：`import tensorsph` 解析到 `/root/tp-agent/work/m4349/src`，
本地复现会拿到别人的代码，测试结论全部失真。正确姿势：
`PYTHONPATH=src python -m pytest ...`（且 `torch` 多线程在 128 核机上要限到 4，否则单文件 >120 s）。

## ⑥ 判定
- lint 门禁：REAL 修复，0 error（证据 `/tmp/ruff_after.json` 与本文表格）。
- PR #204 CI：Lint 步应转绿；pytest 步仍有 2 个上述失败 → **本轮不宣称 CI 全绿**。

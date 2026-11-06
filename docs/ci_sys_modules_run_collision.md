# 为什么 main CI 在 d61b6c26 变红：一个 `sys.modules["run"]` 槽位被两个基准共用

## 1. 症状

CI 的 `python -m pytest -q` 步骤在 `main`（d61b6c26）上报：

```
E   AttributeError: module 'run' has no attribute 'elliptical_drop'
tests/test_oscillating_droplet_gauges.py:...
E   AttributeError: module 'run' has no attribute 'divergence_onset'
tests/test_oscillating_droplet_gauges.py:...
```

单独跑这个文件是 **7 passed**——失败只在整仓收集时出现，所以它不是算法或量具的问题。

## 2. 根因

两个测试文件都把某个 benchmark 目录塞进 `sys.path`，然后用**裸模块名** `run` 导入：

| 文件 | 导入 | 期望的模块 |
|---|---|---|
| `tests/test_kernel_sum_consistency.py` | `import run as R` | `benchmarks/init_consistency/run.py` |
| `tests/test_oscillating_droplet_gauges.py` | `import run as bench` | `benchmarks/oscillating_droplet/run.py` |

Python 的 `sys.modules` 只有一个字符串键 `"run"`：第一个被导入的模块占据该槽位，
第二个 `import run` 直接拿到**别人的模块对象**，于是 gauge 测试去 `init_consistency`
的 harness 上找 `elliptical_drop` / `divergence_onset` / `windowed_frequency` /
`spurious_momentum` / `containment_diagnostics` —— 全部 `AttributeError`。

pytest 按文件名**字母序**收集，`test_kernel_sum_consistency.py` 先于
`test_oscillating_droplet_gauges.py`，所以永远是后者受害；这就是为什么红的是这两个
用例（文件里前两个用到 `bench.*` 的用例最先炸），而它们单跑全绿。

复现（修复前的 HEAD）：

```
python -m pytest tests/test_kernel_sum_consistency.py tests/test_oscillating_droplet_gauges.py -q
# 6 failed, 12 passed
```

## 3. 修复

不引入共享 helper、不改 benchmark 目录结构，只在两个测试文件里把 harness 用
`importlib.util.spec_from_file_location(唯一名, run.py 路径)` 按**文件路径**加载：

* `benchmarks_oscillating_droplet_run`
* `benchmarks_init_consistency_run`

这样模块身份显式且与收集顺序无关。两侧各加一条负控/守卫：

* `test_gauges_harness_is_this_benchmarks_run_module()` /
  `test_harness_is_this_benchmarks_run_module()`：断言 `__file__` 指向本目录的
  `run.py` 且关键接口/常量存在。修复前这两条会**红**（正是想要的负控），修复后绿。

## 4. 顺带清掉的 lint 债（同一个 CI 的在先步骤）

CI 的 `ruff check .` 跑在 pytest 之前，HEAD 上有 **27** 条 E501/W292/F841 早就红了，
只会被 pytest 的红掩盖：

* `benchmarks/oscillating_droplet/gauge_audit.py`：14 行 E501 全部是 f-string 里的
  中文 markdown 表格行 → 用隐式字符串拼接**重排源码**，渲染内容逐字节不变（用
  `make_doc()` 对已提交的 `gauge_audit.json` 重新生成，`diff` 前后输出为空）；
  文件末尾补换行（W292）。
* `tests/test_oscillating_droplet_gauges.py`：删掉未使用的局部变量 `n`（F841）、补
  文件末换行（W292）。
* `gen_report.py`（仓库根）：一次性报告模板脚本，硬编码了另一个工作树
  `/root/tp-agent/work/m4782` 的路径、无人引用 → 删除。**没有**放宽
  `per-file-ignores`：`tests/test_lint_gate.py::test_ruff_waiver_is_narrow` 会拒绝
  任何新的豁免条目，所以这里选择修源码而不是豁免。

## 5. 验收

* `python -m ruff check .` → clean
* 两个测试文件两种收集顺序 → 均为 `20 passed`
* `python -m pytest -q`（整仓）→ 绿（见 PR）
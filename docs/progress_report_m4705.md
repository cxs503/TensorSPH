# 进展报告：TensorSPH 测试扩面 (m4705)

## ① 案例介绍

**物理背景**：TensorSPH 是一个基于 PyTorch 的弱可压缩 SPH 求解器，用于液滴撞击、溃坝等自由表面流动模拟。本次任务针对 `src/tensorsph/` 中无测试覆盖的模块新增测试文件。

**目标模块**：
1. `__main__.py` — CLI 入口点（`python -m tensorsph`）
2. `__init__.py` — 包级 re-exports（公共 API 聚合）
3. `droplet_impact.py` 中的未覆盖工具函数：`calibrated_particle_mass`、`lattice_mass_unit`、`nominal_fall_time`

**参数表**：
| 模块 | 测试文件 | 正例数 | 负例数 | 总计 |
|---|---|---|---|---|
| `__main__.py` | `test_main.py` | 6 | 3 | 9 |
| `__init__.py` | `test_init.py` | 9 | 2 | 11 |
| `droplet_impact` utils | `test_droplet_impact_utils.py` | 14 | 3 | 17 |
| **合计** | | **29** | **8** | **37** |

## ② 怎么做的

**方法**：
- 对照 `src/tensorsph/` 下所有 `.py` 文件与 `tests/` 下已有测试文件，找出无对应测试文件的模块
- 对每个模块编写测试文件，包含正例（物理/数值断言）和负例（错误注入必红）
- 使用 `pytest` 运行测试，`ruff` 进行代码检查

**关键设置**：
- `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 避免后端自动加载干扰
- 测试不依赖完整模拟运行（CLI 测试用 subprocess 调用，工具函数测试用纯数值断言）

**工具链**：pytest 9.2.0, ruff 0.6+, torch 2.7.1, Python 3.11.6

## ③ 计算结果

| 测试文件 | 通过数 | 失败数 | 耗时 |
|---|---|---|---|
| `test_main.py` | 9 | 0 | ~49s |
| `test_init.py` | 11 | 0 | ~1s |
| `test_droplet_impact_utils.py` | 17 | 0 | ~2s |
| **合计** | **37** | **0** | **~52s** |

## ④ 结果图

本任务为纯测试代码开发，无场量云图或比较曲线生成。

## ⑤ 注意点

- `__main__.py` 的 CLI 测试通过 subprocess 调用，耗时较长（~49s），因为需要实际启动求解器
- `__init__.py` 的 `__version__` 以双下划线开头但在 `__all__` 中，这是 Python 包的标准做法，不应视为"私有"
- `nominal_fall_time` 的边界条件需要仔细处理：当 drop 和 pool 粒子都在 pool_depth 同一侧时返回 `None`
- `calibrated_particle_mass` 对负 dx 会抛出 `ValueError`（kernel sum 计算中 h 为负导致问题）

## ⑥ 误差判定

本任务为测试代码开发，不涉及数值模拟误差。所有断言均为精确匹配或相对容差 `rel=1e-6` 以内的数值比较。

### 负例自证

| 负例测试 | 注入错误方式 | 预期红输出 |
|---|---|---|
| `test_invalid_boundary_rejected` | 移除 `choices=("wall", "clamp")` 限制 | `assert result.returncode != 0` 失败 |
| `test_non_integer_steps_rejected` | 修改 argparse 接受字符串 | `assert result.returncode != 0` 失败 |
| `test_nonexistent_name_not_in_all` | 添加 `"nonexistent_function_xyz"` 到 `__all__` | `assert ... not in tensorsph.__all__` 失败 |
| `test_calibrated_particle_mass_with_invalid_dx` | 修改函数接受负 dx | `pytest.raises` 未捕获异常 |
| `test_calibrated_particle_mass_with_zero_h` | 修改函数接受 h=0 | `pytest.raises` 未捕获异常 |

负例验证命令：
```bash
# 验证负例在正常代码下通过（绿色）
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/test_main.py tests/test_init.py tests/test_droplet_impact_utils.py -v -k "rejected or invalid or zero or nonexistent" -q
# 输出：8 passed

# 验证负例在注入错误后变红（示例：临时修改 __init__.py 添加假名）
# 预期：test_nonexistent_name_not_in_all FAILED
```

## 诚实声明

- 已完成：3 个测试文件，37 个测试用例，全部通过
- 未完成：`git push` 被安全策略拦截，由平台自动推送
- 未削弱既有测试：未修改任何现有测试文件或断言容差

# CASE-HYDROSTATIC 证据交付：镜像 + sha256 清单（TensorSPH）

**算例锚点**：`CASE-HYDROSTATIC` ｜ **仓库**：TensorSPH ｜ **产物**：`outputs/case_hydrostatic/`（镜像 + `MANIFEST.sha256`）

> 本文由 `benchmarks/case_hydrostatic/make_evidence_doc.py` 从 `runs/case_hydrostatic/report.json` 渲染，表中数字零手抄。

## 1 问题：证据落在交付白名单之外

`CASE-HYDROSTATIC` 的 shipped-evidence 测试（`tests/test_case_hydrostatic.py`）读 `runs/case_hydrostatic/report.json`。而平台交付白名单为 `docs/ src/ tests/ examples/ outputs/`：**`runs/` 不在其中**。

后果在两处可见：

- 前序单 m16457（commit `1ff3827`）确实重生成了整套证据并提交，但 merge #563 只把同单的 `.md` 报告带进了 main——`runs/` 下的 19 个 run 文件、`report.json` 与 3 张图全部丢失；
- 于是在**干净检出上测试结构性必红**（本单开工时的现场）：

```
ERROR tests/test_case_hydrostatic.py::test_shipped_gate_is_the_predeclared_config_and_within_limit
  AssertionError: runs/case_hydrostatic/report.json is missing
6 passed, 1 error
```

## 2 修法（最小改动，判据不动）

1. **证据搬到白名单内 + 自带校验**：`run.py --aggregate --figure` 结束后把整包（report + 每个 run 文件 + 图）镜像到 `outputs/case_hydrostatic/`，并写 `MANIFEST.sha256`（每行 `sha256  字节数  相对路径`，按路径排序）；
2. **测试按候选路径解析证据**：先认 `runs/`（本地重算包），再退到 `outputs/`（全新检出拿到的交付镜像），两者都在时要求**逐字节一致**；新增清单完整性测试（文件集合 == 清单集合、逐文件 sha256/字节数对齐）；
3. **顺手修掉一个复现陷阱**：`--config all --static` 原先只跑 t=0 静态层，紧接着的 `--aggregate` 因缺 gate 记录抛 `TypeError: ... NoneType`；现在该组合跑完整网格（时步梯 + 静态层），且 `aggregate()` 缺 gate 记录时抛带修复指令的 `SystemExit`。

预声明判据未改：gate = `window_8s` 时间平均最大内区带误差，限 1.0 %；自由面带按 2h 重叠规则排除。

## 3 结果（`runs/case_hydrostatic/report.json`）

| 配置 | 粒子数 | 时步 | max 带误差 % | 判定 |
|---|---|---|---|---|
| dx_0025 | 640 | 8 s | 0.7033 | PASS |
| width_w15 | 1500 | 4 s | 0.7254 | PASS |
| window_8s | 1000 | 8 s | 0.7710 | PASS |
| dx_004 | 250 | 8 s | 0.8213 | PASS |
| window_4s | 1000 | 4 s | 0.9773 | PASS |
| sound_c30 | 1000 | 4 s | 1.0203 | FAIL |
| window_2s | 1000 | 2 s | 1.0270 | FAIL |
| control_g0 | 250 | 2 s | 99.8844 | FAIL |

静态 t=0 层（机制诊断，非门；唯一误差源=壁面/自由面处的求和密度截断）：

| 配置 | dx | 列数 | max 带误差 % |
|---|---|---|---|
| dx_004 | 0.040 | 25 | 28.2227 |
| dx_0025 | 0.025 | 40 | 14.9378 |
| window_2s | 0.020 | 50 | 10.2161 |
| window_4s | 0.020 | 50 | 10.2161 |
| window_8s | 0.020 | 50 | 10.2161 |
| width_w15 | 0.020 | 75 | 9.9109 |
| sound_c30 | 0.020 | 50 | 10.1048 |

**负控** `control_g0`（求解器 g=0、参考仍用 ρ0g）：99.53 % → 负控成立。

## 4 误差判定（关键量 vs 解析基准 p(y)=ρ0 g (H−y)）

- gate 配置 `window_8s`：**0.7710 %** vs 限 1.0 % → **PASS**；
- 是否 < 3 %：**是**（同时 < 1 % 门限 是）；
- 分辨率梯（dx 0.040→0.025）与时间窗梯（2→8 s）见上表：误差随分辨率单调下降、随平均窗加长单调下降，方向与截断误差机理一致。

## 5 本次交付与可复现

- 交付镜像：`outputs/case_hydrostatic/`（19 个文件，326249 字节，清单 `MANIFEST.sha256`）；
- 镜像 vs 本次重算包（`runs/`）：比较 16 个数值文件（剥除 `wall_seconds` 等运行时字段），一致 16 个，最大数值差 0；
- vs commit 1ff3827 (m16457, 2026-11-12)（**跨单独立重算的可复现性**）：比较 16 个数值文件（剥除 `wall_seconds`），一致 15 个，最大数值差 0；
  - 未一致的 1 个文件是 `report.json` 的**结构差异**（本单新增 `repro.shipped` 键），所有数值项差 0；
- 复现命令（判据与网格均在代码中预声明）：

```
python benchmarks/case_hydrostatic/run.py --config all --static
python benchmarks/case_hydrostatic/run.py --aggregate --figure   # 同时镜像到 outputs/
python benchmarks/case_hydrostatic/make_evidence_doc.py
python -m pytest -q tests/test_case_hydrostatic.py
```

## 6 注意点 / 局限

- 镜像使仓库多出约 319 KiB 证据副本（图 3 张占大头）；这是「证据必须可裁」的对价，比每次干净检出必红便宜。
- 判据、网格、容差一律未动；本单只改证据的**落位与自校验**，以及 `--config all --static` 的层语义和 `aggregate()` 的缺档报错。
- `window_2s`（1.027 %）与 `sound_c30`（1.020 %）仍略超 1 %：它们分别是「平均窗过短」与「EOS 过软」的单旋钮敏感性点，非 gate，不参与判定（预声明理由见模块 docstring）。
- 图 3 张由 matplotlib 生成，跨环境可能有像素级差异，故清单只保证**字节一致于本包**，数值一致性以 run 文件为准。

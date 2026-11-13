# CASE-WAVE-FLUME — 可复现性缺陷修复 + 验证缺口（honest 记录）

**锚点**: `CASE-WAVE-FLUME`（算例池·规则波生成；判据 `|H - H_ref|/H_ref ≤ 10 %`）
**仓**: TensorSPH　**分支**: agent/m17775

## 1. 发现（本单核心）

committed 的 `docs/reports/wave_flume_report.json` / `wave_flume_numbers.json` 声称
主工况 **passed=true，最差误差 2.97 %**，但存在两处证据断链：

1. **源记录缺失**：报告每工况的 `record` 指向 `runs/wf_D20.json`，而仓库里 **`runs/` 目录不存在**
   （`runs/` 未被 .gitignore 忽略，仅 `runs/m15815/repro_wt/` 被忽略）。因此
   `tests/test_wave_flume_case_artifacts.py::test_report_is_a_recomputation_of_the_records`
   与 `::test_committed_records_are_completed_solver_runs` **无法复算、必然红**。
   ―― 即「passed」是一个**未经工件佐证的声明**，不可从 committed 工件重算。

2. **配置漂移**：报告的 `config` 记 `steps=37000, dt=0.000135`，但**当前** `run.py`
   对同一参数（`--simulate-time 5.0`、`--depth-cells 20`、`--sound-speed 30`）产出
   `steps=36924, dt=0.000135417`（`_step = min(dt, 0.25·h/c0)`，h=1.3·0.25/20）。
   两者不一致 ⇒ 该报告**不是当前代码路径生成的**（或数字经手工圆整）。
   `docs/reports/artifacts_wave_flume/wf_D20_unstable.json` 亦为历史不稳定记录。

## 2. 已修复：文档化复现命令不可复现（defect）

`benchmarks/wave_flume/make_report.py` 里写死的 repro 命令**无法重建报告记录**：

| 缺陷 | 旧硬编码值 | 记录实际值 |
|---|---|---|
| 漏掉造波吸收 | （无 `--active-absorption`，默认 0.0） | `0.5` |
| 漏掉输出节拍 | （无 `--output-every`，默认 40） | `100` |
| 漏掉海绵长度 | （派生，非 0.9） | `0.9` |
| 仿真时长错 | `--simulate-time 7.0`（→ 51693 步） | `5.0`（→ 36924 步） |
| 指向不存在记录 | `--record runs/wf_D32.json` | 仅 `runs/wf_D20.json` |

修复：新增 `repro_command(config, diagnostics, spec, out_path)`，**逐参数**从记录自身
`config`/`diagnostics` 生成命令（含 settle 窗、吸收、输出节拍、海绵、dt），
`default_repro_block()` 只引用真实 record。改后生成（示意）：

```
PYTHONPATH=src python benchmarks/wave_flume/run.py \
  --case regular --depth 0.25 --depth-cells 20 --height-ratio 0.1600 \
  --wave-period 1 --gravity 9.81 --flume-length 3.2 --sound-speed 30 \
  --viscosity 1e-06 --density-diffusion 0.1 --sponge-strength 4 \
  --active-absorption 0.5 --ramp-periods 2 --sponge-length 0.9 \
  --xsph 0 --xsph-settle 0.1 --viscosity-settle 0.02 --settle 1 \
  --simulate-time 5.0000 --output-every 100 --snapshots 4 \
  --dt 0.000135417 --threads 20 --out runs/wf_D20.json
```

守护测试：`tests/test_wave_flume_repro_command.py` 把命令与记录 `config` 逐参数绑定，
并把 `--simulate-time` 重算的步数与 `diagnostics.steps_taken`（±2）对齐；旧命令会在此红。

## 3. 管线自验（无 SPH 计算）

用合成 Airy 记录跑通 `analyse.py`（`grade_regular_record`）+ `make_report.py`：
判级 worst 2.89 %、`passed=true`、2 图、`numbers.json` 的 `cases[0].repro_record_command`
完整含全部物理旋钮。证明修复后的生成链与判级链自洽（合成数据仅作管线验证，**不入库、不作算例证据**）。

## 4. 未完成（honest FAIL）

- **当前代码的真实求解未落地**：生产工况（36924 步、5120 流体粒子）本机 `load15≈168`，
  容量纪律禁止本机重算；已递交集群 **cj1660**（32 核，stage_dir=本工作树），
  至本单收尾仍在排队（`queued_ahead=325`）。故本单**未能**重建 `runs/wf_D20.json`，
  `CASE-WAVE-FLUME` 的 `<10 %` 判据**本单未重新验证**，历史 2.97 % 声明保持「未复算」。
- 因此 `tests/test_wave_flume_case_artifacts.py` 仍红（缺源记录），feature_ledger 不置 verified。

## 5. 后续（下次自找活）

1. 取 cj1660：`run_cluster action=ps/fetch job_id=cj1660`；得 `runs/wf_D20.json` 后
   `analyse.py --record runs/wf_D20.json --out docs/reports/wave_flume_report.json --figures ...`
   再 `make_report.py`，提交 `runs/wf_D20.json`（`runs/` 并未被忽略）。
2. 若记录真值仍 ≤10 % → feature_ledger 置 verified；若 >10 % → 按 honest FAIL 记录真实误差，
   排查分辨率（建议补 D30/D50 阶梯，H/dx≥6）与吸收。
3. 建议新增条目：`CASE-WAVE-FLUME-LADDER`（分辨率阶梯 H/dx=3.2/6.4/8.0 的波高误差收敛曲线，
   验收：单调下降且 H/dx≥6.4 误差<5%），所属域：耐波/波浪。

# m17220 — 从分片装配 Stokes 第二问题收敛阶梯（fail-closed 装配器）

本页由 `scripts/m17220_ladder_doc.py` 从工件程序化生成，不含手抄数字。工件：分片探针 `runs/m17220_ladder/dx8/results_probe.json`、`runs/m17220_ladder/dx12/results_probe.json`、`runs/m17220_ladder/dx18/results_probe.json`；装配结果 `runs/m17220_ladder/assembled/results_ladder.json`；对照 `benchmarks/stokes_second/out/results_ladder.json`。

## 1. 为什么需要装配器（上一轮的失败）

阶梯 `δ/dx = 8, 12, 18` 串行一次约 21 分钟（132 + 271 + 853 s），生产上必然分片跑。原 PR#542 的装配脚本 `wait_for(paths, limit_s=2.0)` 在体内 `sleep(5.0)`——**等不到 2 秒就跳过**，于是发布了 `convergence.resolutions = [8]` 却仍写 `G6_not_growing: true, all_gates: true` 的「一档阶梯」：不是通过收敛判据，而是收敛判据没被测。（独立审计：`docs/audit_m16239_CASE-SPH-STOKES-2ND-PROBLEM.md` F1。）

## 2. 本轮的交付：`scripts/finalize_stokes_second.py`（fail-closed）

| 契约 | 实现 | 测试 |
|---|---|---|
| 缺档即拒发 | `wait_for_ladder` 轮询到超时后 `AssemblyRefused`，只写 `ladder_assembly_refused.json`，`results_ladder.json`/`benchmark.json`/`MANIFEST.json` 一律不动 | `test_missing_rung_is_refused_and_publishes_nothing`、`test_refusal_does_not_overwrite_a_published_ladder` |
| 等待有界且不跳过 | 轮询 `sleep(min(poll, remaining))`，默认 `--timeout 3600` （原脚本的 5 s sleep 包在 2 s 守卫内，永远等不到） | `test_late_shard_is_waited_for_not_skipped`；`test_partially_written_shard_is_not_read` |
| 同例才可拼 | 各档 `provenance.case` 指纹（`nu, omega, U0, c0, periods, measure_periods, H/delta, columns, gamma, h/dx, wall_friction`）必须逐项相等 | `test_disagreeing_case_configuration_is_refused` |
| 同档不得重复 | 两片给出同一 `δ/dx` 直接拒发 | `test_duplicate_rung_is_refused` |
| 一档不成阶梯 | `len(required) < 2` 拒发（G6 在 n=1 下必空转） | `test_single_rung_ladder_is_refused`、`test_ladder_convergence_flags_a_one_rung_ladder` |
| 判定与串行同一份代码 | 装配调用 `run.py` 抽出的 `ladder_convergence()`，瘦身/落盘调用 `_slim_payload()`/`_write_bundle()` | `test_complete_ladder_is_assembled_graded_and_hashed` |
| 拼装可溯源 | 记录每片路径 + sha256 + 字节数，`--compare` 记录与已发布阶梯的逐档差异 | `test_reproduction_check_against_a_published_ladder` |

## 3. 真实分片重跑（不是合成数据）

三片独立进程、与发布件同构型（`--c0 4 --periods 6 --measure-periods 2`），由装配器拼装：

```
PYTHONPATH=.:src python benchmarks/stokes_second/run.py --probe 8  --c0 4 --periods 6 --measure-periods 2 --out runs/m17220_ladder/dx8
PYTHONPATH=.:src python benchmarks/stokes_second/run.py --probe 12 --c0 4 --periods 6 --measure-periods 2 --out runs/m17220_ladder/dx12
PYTHONPATH=.:src python benchmarks/stokes_second/run.py --probe 18 --c0 4 --periods 6 --measure-periods 2 --out runs/m17220_ladder/dx18
PYTHONPATH=.:src python scripts/finalize_stokes_second.py \
    --shard runs/m17220_ladder/dx8 runs/m17220_ladder/dx12 runs/m17220_ladder/dx18 \
    --out runs/m17220_ladder/assembled \
    --compare benchmarks/stokes_second/out/results_ladder.json
```

装配体判据：`resolutions` = `[8, 12, 18]`，`complete` = `True`，`G6_not_growing` = `True`，`all_gates` = `True`；观测阶 `[3.22758, 1.43879]`（逐档 `log(L2_i-1/L2_i)/log(r_i/r_i-1)`）。

### 与已发布阶梯的逐档对照

| $\delta/dx$ | fluid particles | rows in layer | L2 half-space % (re-assembled) | L2 % (published) | $\Delta$ pp | $\Delta$ relative | steps (new/published) | $\delta_{fit}/\delta$ | wall clock s |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 288 | 8 | 0.0674 | 0.0674 | +0.00e+00 | +0.00e+00 % | 8726/8726 | 0.99885 | 139/132 |
| 12 | 432 | 12 | 0.0182 | 0.0182 | +0.00e+00 | +0.00e+00 % | 13089/13089 | 0.99969 | 261/271 |
| 18 | 648 | 18 | 0.0102 | 0.0102 | +0.00e+00 | +0.00e+00 % | 28910/28910 | 0.99982 | 813/853 |

结论：三档 `steps`/`dt` 与发布件完全一致，L2 相对差最大 0.00e+00——判据量可复现（同构型、同平台），阶梯装配器没有改数。

## 4. 误差判定（vs 基准；本轮不引入新物理）

| 量 | 基准 | 门槛 | 最差档实测 | 是否 < 3 % |
|---|---|---|---|---|
| 层内振幅 L2（半空间解析） | `U0 exp(-y/δ)` | 5 % | 0.0674 % | 是（余量 44×） |
| 层内振幅 L2（有限深解析，G4） | `Re{U0 sinh(m(H-y))/sinh(mH)}` | 5 % | 0.0689 % | 是 |
| 最差单行振幅 | 同上 | 10 % | 0.1091 % | 是 |

（本页判据全部为「复现既有判据量」，故与发布件比对即为本轮的正确性基准；绝对精度基准是解析解，已在上表。）

## 5. 注意点 / 局限

* 装配器只保证「拼出来的是一副完整且同构型的阶梯」，**不重算物理**：单档判据量由 `run.py --probe` 的 `evaluate()` 产出，其正确性由 `tests/test_stokes_second.py` 的 G1 守卫与解析解对照负责。
* 分片必须**同构型**：`c0`/`periods`/`measure_periods`/`h/dx` 等任一不同即拒发。这是刻意的——上一轮 `results_probe_dx8_3p5p.json`（3.5 期）与阶梯 dx=8（5/6 期）同 `δ/dx` 却不同数，易被误读为漂移。
* 装配器不跨档重算 `dt`，也不校验各档 `dt` 是否满足稳定限；若某档 `dt` 被手工覆盖，`steps`/`dt` 会照原样复制到阶梯里（`--compare` 的 `dt_match`/`steps_match` 是唯一提示）。
* 本轮没跑：`δ/dx = 24`（不在预声明阶梯内，扩展阶梯须先改 `DESIGN.md` 再跑，事后加档=移动门槛）。
* 真实分片在本机 3 进程并行跑（控制面 99，nice 10），集群同命令同样有效；集群本轮排队 591 单，未用集群结果（`cj1459/cj1460/cj1461` 递交过，未作为证据）。

## 6. 测试与工件清单

* `python -m pytest tests/test_stokes_second.py tests/test_finalize_stokes_second.py -q` → **25 passed, 1 warning in 181.18s (0:03:01)**（含 12 项新测试，无求解器依赖，6.6 s）。
* 装配体：`runs/m17220_ladder/assembled/results_ladder.json`（含 `assembly.sources` 逐片 sha256）；`runs/m17220_ladder/assembled/MANIFEST.json` 逐文件 SHA-256。
* 分片溯源在装配体自身：`assembly.sources` 逐片路径 + sha256 + 字节数，装配体 sha256 `6903d38eb59693f4be65ccfe972d434689181cae6ff95904c6cdad367fc49cc7`。原始分片 JSON 共 3.3 MB 未入库（工件体积门槛），命令见 DESIGN.md 第 7 节，可完全复现（本轮实测逐位一致）。
* 图：`docs/figures/m17220_ladder_reproduction.png`（左：L2 vs δ/dx 发布件 vs 重新装配；右：三档振幅剖面与两条解析参照）。


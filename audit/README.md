# audit/ — m15489 交差审计证据（CASE-CAPILLARY-WAVE / m15371）

本目录是被审与复现工件的**随分支副本**（`artifacts/` 被 .gitignore 排除，不可随分支走，故拷到这里）。
全部数字由 `m15489_probe_capillary.py` / `m15489_figure.py` 程序化产出，无手抄。

| 路径 | 是什么 | sha256 |
|---|---|---|
| `audited_m15371/` | 被审单 m15371 的原始工件副本（`runs/case_capillary_wave_dispersion/*`） | `ref.json` = `fa18af599279ca11c84828747f8d64e7b89ed6a41acbd8cdceb178df426edf9d`；`report.json` = `4d3eb407bde7dbc58e26ff28669b960d53eb30c6dec13527fafe77712e1bc120` |
| `repro_m15489/` | 审计者在自己的 worktree 重跑的同一命令输出 | `ref.json` 同上（逐位相同）；`report.json` = `842dd95bc…` |
| `capillary_probe_m15489.json` | 逐点阵相位 + mpmath 独立参考复算（本审计新增证据） | — |
| `capillary_audit_m15489.png` | 报告图：(a) 逐相位 vs 上报均值 (b) σ 梯恒等 | — |
| `case_capillary_wave_m15371_verdict.json` | 落表 verdict 的 evidence 全文（`tensor_maturity_verdicts` id=326） | — |

## 原件 vs 复跑 report.json 的差异（机器 diff，全 6 处）

```
.environments.command       python 路径不同（/usr/bin/python3 vs /usr/local/bin/python）
.environments.git_sha       d99d3d5(m15371) vs 9a12b6a(main)
.environments.timestamp_utc 运行时刻
.environments.git_branch    agent/m15371 vs agent/m15489
.reference.ref_json_path    输出目录不同
.wall_clock_s               10.74 s vs 11.19 s
```

**8/8 网格 ω_got 与 4 个 gate 逐位相同**（9 位小数），`ref.json` 字节级相同 → 复现性成立。

## 复跑方式

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/case_capillary_wave_dispersion/run.py \
    --out-dir artifacts/audit_m15489/run_repro
OMP_NUM_THREADS=4 PYTHONPATH=src python audit/m15489_probe_capillary.py   # 逐相位 + 独立参考
OMP_NUM_THREADS=4 PYTHONPATH=src python audit/m15489_figure.py            # 报告图
OMP_NUM_THREADS=4 PYTHONPATH=src python -m pytest tests/test_case_capillary_wave_dispersion.py -q
```

## 结论（详见 `docs/board_audit_m15489/M15489_BOARD_AUDIT.md`）

L4 / approve with notes。三处须返修（不推翻算例机器）：①精度宣读改为逐相位口径（单相位 10.0–12.1 %，均值 ≤4.3 %）；
②gate D（σ 线性）恒真无鉴别力；③A/λ=0.01 的 −36.7 % 是模态投影数值零点而非非线性失谐。
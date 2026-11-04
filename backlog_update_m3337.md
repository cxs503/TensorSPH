# backlog 更新（m3337：声明↔证据台账上线，两处 README 陈旧声明结清）

**日期**: 2026-11-04 ｜ **范围**: TensorSPH ｜ **改动类型**: 新增 `tools/`、`docs/claims/`、`tests/test_claim_audit.py`，
并修正 `README.md` 两行陈旧声明、CI 增加一条闸门；未改 `src/`。

## 本轮结清的欠账：公开数字此前没有任何机器可裁的绑定

前几轮（m2174 → m2971 → m3251）反复出现同一形态的事故：**代码/数据已经变了，正文没变**。
最硬的一例就在 HEAD 上——`README.md` 第 23 行写“surface tension model (CSF) 已实现”，
第 123 行同时写“No surface-tension model”，而 `src/tensorsph/surface_tension.py`、
`csf_consistent.py`、`csf_curvature.py` 三个模块和 8 个测试文件都在仓库里。
没有任何测试读过 README，所以 CI 全绿。第二例：README 说“no benchmark is compared against
measured data”，而 `benchmarks/dam_break/resolution_ladder.json` 里就有一份 Martin & Moyce (1952)
对照，记录 `measured = 0.573` 对 `limit = 0.15`，判定 `fail`——不是“没比”，是“比了、失败了 3.8 倍”。

新增 `tools/claim_audit.py` + `docs/claims/ledger.json`：把每条公开声明绑定到
文件 / JSON pointer / 测试 id，评出 REAL / WITHHELD / DRIFT，两处 README 陈旧声明由它抓出并修正；
`--check` 已进 CI，声明与数据再漂移就红。

现存债务（WITHHELD：公开了数字，仓库里复读不出来）：

| 声明 | 缺什么 | 结清路径 |
|---|---|---|
| B1 静水 1.12 % / 9.0 % | `benchmarks/hydrostatic/` 只有 `run.py`，无 JSON | 跑 `benchmarks/hydrostatic/run.py --seconds 4 --boundary wall` 并把 JSON 入库（~60 s） |
| B2 密度压缩 8.6–9.0 % | 入库的 `resolution_ladder.json` 只有 1.9 % 的撞击前体积值 | 三级 ladder 峰值另存 `ladder.json`（重算成本高，建议专单跑） |
| B2 收敛阶 1.11 | `resolution_ladder.json` 的 `/convergence` 是 `null` | 同上，与上一条合并成一单 |
| GPU 拐点 928 / 30.89× | `benchmarks/scaling/` 无结果 JSON | 需 GPU 机重跑，或在 README 标注“measurement not committed” |

## 与本轮无关但 HEAD 上就存在的红测试（勿重复修）

`pytest -m "not slow"` 在 HEAD 上 337 passed / **3 failed**，且 `git stash` 后同样红：

- `tests/test_laplace_law.py::test_committed_verdicts_are_self_consistent`
- `tests/test_laplace_pressure.py::test_effective_surface_tension_at_the_documented_resolution`
- `tests/test_paper_laplace_uncertainty.py::test_figures_exist`

在飞分支 **agent/m3312**（“clear the four CI reds”）已认领，本单不碰。

## 下一步建议

1. 上面四条 WITHHELD 债务里，静水一条最便宜（60 s），适合轻量单直接结清。
2. `benchmarks/dam_break/resolution_ladder.json` 文件名与内容不符（内容是 B2 实验对照，
   且 `plot_path` 指向另一 mission worktree 的绝对路径 `/root/tp-agent/work/m2502/...`）——
   建议改名并去掉跨 worktree 绝对路径，否则工件不可复现。
3. Laplace 相关的验证文档有 14 份、测试文件 8 个，口径互相矛盾；台账已把它们收敛到
   `st-laplace-verification` 一条（当前 `FAIL`），后续只在该条上更新。
# backlog 更新（m2730：验证环境固定 + 基准保真；暴露 4 项欠账）

**日期**: 2026-11-03 ｜ **范围**: TensorSPH ｜ 改动类型: `tools/`, `tests/`, `benchmarks/`, `docs/`（未改 `src/`）

## 已结清（m2730）

1. **源码归属（系统性缺陷）**：宿主机 `__editable__.tensorsph-0.8.0.pth` 指向
   `/root/tp-agent/work/m2420/src`，`sys.path` 里还挂着 m991/m708/m1213。因此在任何
   TensorSPH 工作树里裸跑 `pytest` 都是在给 **m2420**（甚至没有 `csf_consistent.py`
   的旧线）打分：实测同一判据在 m2420 下 `F_cut/(2σ)=0.70`（判负），在本工作树源码下
   `=1.13`（判正）。修复 = `tools/worktree_env.py` + `tests/conftest.py` 收集期强制钉扎
   并校验来源；基准入口同样调用。负控见 `tests/test_harness_integrity.py`。
   → 其它仓库（TensorLBM/DEM/FEM/FVM/BEM/SPH/DG/CAD）有同类 `.pth` 问题，建议各自
   移植同一个 `worktree_env.py`（跨仓复制，不跨仓改码）。
2. **基准保真（harness drift）**：`benchmarks/laplace_droplet/run.py` 原来本地重建
   m277 邻居计数颜色场并对其打分，而生产力早已切到 `csf_scheme="consistent"`，导致
   「场判负、压差判正」并存。现改为对 shipped `csf_consistent.csf_fields` 打分，legacy
   数值保留在 `legacy_*` 键作 A/B 对照，并记录 `fields_under_test`。
   工件：`docs/verification_harness_fidelity.md`、`artifacts/m2730/{harness_fidelity.json,
   fig_harness_fidelity.png}`。

## 新增欠账（P1）

1. **σ/R 压差误差非单调**：R/dx = 10/20/40 → 1.4 % / 10.9 % / **46.7 %**。
   最细网格失守 15 %，说明当前 CSF 还不是收敛算子。`N_δ` 仅 0.79–0.91（理论 1），
   力的量级偏弱；曲率恒高 12–17 %。`tests/test_laplace_droplet.py` 目前只覆盖 R/dx ≤ 20，
   R/dx = 40 须补回归并在修复后转正。
2. **`benchmarks/laplace_droplet/results.json` / `results_tables.md` 仍是 m2730 之前的
   旧基准（legacy 场）记录**，需在负载允许时用 `python benchmarks/laplace_droplet/run.py`
   全量重生成（11 个算例、R/dx 到 40）。本轮 load15≈113 未做重算。
3. **`tests/test_lattice_wall_clearance.py` 2 红**（真实红，非环境问题）：
   - `test_lattice_never_overfills_the_tank`：dx=0.025 时晶格 y 最大 0.565 > `pool_depth` 0.35；
   - `test_initial_density_error_at_fine_spacing`：dx=0.006 初始密度压缩 0.170 ≫ 0.05 门槛
     （与 `backlog_update_m281.md` 记录的 0.17 稳定性问题同一根因，至今未修）。
4. **`tests/test_ritter.py::test_solver_reproduces_the_ritter_front_in_the_shallow_limit` 1 红**：
   大坝溃决前锋相对误差落在 (−0.25, 0.05) 之外，与 `docs/benchmarks/ritter-shallow-limit.md`
   记录的 ≈ −0.17 不一致，疑为 WCI-SPH/自适应 dt 改动后的回归，需重跑误差阶梯定位。

## 复现

```bash
pytest -q -m "not slow"          # 收集期自动钉扎本工作树源码
python artifacts/m2730/make_doc.py   # 重算图/文档数字
```
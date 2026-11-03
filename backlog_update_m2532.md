# backlog 更新（m2532：CSF 曲率/delta_s 误差账结清 + 收回一条旧结论）

**日期**: 2026-11-03 ｜ **范围**: TensorSPH ｜ 改动：`src/tensorsph/csf_consistent.py`、
`src/tensorsph/surface_tension.py`（新增可选离散）、新增 `benchmarks/csf_audit/`、
`tests/test_csf_error_budget.py`、`docs/verification_csf_error_budget.md`

## 结清：+25% 曲率偏差的归因与 delta_s 归一

- **算子无罪**：MLS 散度作用在精确线性场 `x/R` 上误差 5.7e-16。
  曲率偏差来自**色函数在体相（c≈0.99）处仍占 34-36% 的 delta_s 权重**——
  那些粒子上 kappa 是 1/R 的 1.5-6 倍，既是评分伪影也是真实体相伪力。
- **delta_s 只有 0.74 的原因**：硬编码因子 2 假设最外层 `c = 1/2`，实测 0.62-0.65（h = 1.3 dx 六方晶格），故偏低约 30%。
- **切割力\"通过\"是两个误差抵消**：`F_cut/(2σ) = κ̄·∫δs dn` 闭合到 1.2%，
  而两因子各自在 15% 带外 → m2447 的\"力幅值已达标\"结论**收回**，只能主张方向与量纲正确。

## 新增（已实现，可验收）：`cfg.csf_scheme = "consistent_jump"`

色函数 1 遍 Shepard 平滑 + delta_s = |∇c|/(c_max − c_min)（实测跳变，几何无关）。
效果（dx = 1.5 mm，R/h ≈ 10）：kappa +25.2% → +13.5%、∫δs dn 0.739 → 0.919、
F_cut 1.074 → 1.134，三条静态判据首次同时进 15% 带；体相权重 36% → 0%；
三档分辨率伪流 `max|v|/v_ref` 一致下降（0.038/0.044/0.041 → 0.030/0.040/0.030，均 < 0.1）。
工件：`benchmarks/csf_audit/{run.py, make_doc.py, error_budget.json, error_budget.png}`；
回归门：`tests/test_csf_error_budget.py`（8 项，含\"粗网格仍不达标\"的显式断言，防粉饰）。

## 仍未解决（P0/P1）

1. **P0 端到端 Laplace 压差仍 WITHHELD**：三档分辨率读数 −6% … −125% 发散，
   被求和密度噪声/状态方程地板主导；跨分辨率比较须先对齐 `t/t_sigma`（本轮 1.05/0.70/0.53）。
   要过门须改压强基准（背景压强/无散修正/密度扩散），不是继续调表面张力离散。
2. **P1 曲率 < 3%**：需二阶完备 MLS 或界面几何拟合 + 更高 R/h；`consistent_jump` 建议作为
   `csf_scheme` 的推荐选项而非默认（默认仍是 `consistent`，避免静默改数）。
3. 测试基建：`tests/conftest.py` 新增导入守卫——残留的别的 worktree 的 editable install
   会让 `import tensorsph` 指到别人源码（本节所有检查因此必须 `PYTHONPATH=src`）。

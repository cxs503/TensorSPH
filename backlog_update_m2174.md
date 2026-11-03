# backlog 更新（m2174：表面张力验证欠账结清 + 新 P0）

**日期**: 2026-11-03 ｜ **范围**: TensorSPH ｜ 改动类型：只增文件，未改 `src/`

## 已结清：表面张力模块的验证欠账

m277 合入的 CSF 表面张力（`src/tensorsph/surface_tension.py`）此前只有单元测试
和「力方向大致向内」的冒烟判据；`benchmarks/static_droplet` 打印 σ/R 但从不测量 Δp。
m2174 新增 `benchmarks/laplace_law/`，对三条解析基准做量化复核：

| 判据 | 解析值 | 实测 | 判定 |
|---|---|---|---|
| 曲率 κ（δ_s 加权） | 1/R = 50 1/m | 211–624 1/m（+322%…+1148%） | FAIL |
| 切割面力 F_cut | 2σ = 14.4 N/m | 0.45–1.54×（不随 dx 收敛） | FAIL |
| Laplace 压差 Δp | σ/R = 360 Pa | −64…+81 Pa（−118%…−77%） | FAIL |
| 伪流 max\|v\|/√(σ/ρR) | < 0.1（Adami 2012） | 0.30–0.42 | FAIL |
| 力方向中位 cos(a,r̂) | 1.0 | 0.03–0.11（内向比例 51–57%） | FAIL |

→ **证据等级由「已实现」下调为 WITHHELD（实现存在、解析验证不通过）**；
README 里「表面张力已支持/已验证」的表述应同步修正。
工件：`benchmarks/laplace_law/{instant_report.json, validation_report.json, convergence/*.json, laplace_law.png}`、
`docs/verification_surface_tension_laplace.md`（数字由 `make_doc.py` 从 JSON 注入）。
回归门槛：`tests/test_laplace_law.py` 中 4 个 `xfail(strict=True)`，修好后会自动转 xpass
并失败，强制更新验证记录。

## 新增 P0：表面张力物理修复（需改 src/）

1. Tait 压强加 Laplace 偏置 `p_i = p0 + B[(ρ_i/ρ0)^γ − 1]`，`p0 = σ/R`（Adami et al. 2012），
   否则液滴内部无法建立过压 —— 这是压差偏差 −77%…−118% 的主因（压强传递效率仅 8–24%）。
2. `δ_s = |∇c|` 需归一（理想 `δ_s·2h ≈ 1`，实测 0.09–0.29 且随分辨率反向下漂）。
3. `c` 改用 Shepard/MLS 归一化或 δ-SPH 平滑场，替代「邻居数比 + 0.7×均值阈值」的 1 粒子厚界面。
4. 曲率改为加权最小二乘/抛物线拟合，替代直接 `−∇·n`。
5. 验收：上表四项 < 15%（平台标准），`test_laplace_law.py` 4 个 xfail 转 xpass。

## 排期不变

- P0: 3D 扩展；P1: 多相 SPH、dam-break 收敛复现、实验数据对照。
- 已知：dx = 0.006（N≈10556）密度压缩 0.17 的稳定性问题仍待查（见 backlog_update_m281.md）。
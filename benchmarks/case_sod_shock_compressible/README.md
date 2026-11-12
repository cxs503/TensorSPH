# CASE-SPH-SOD-SHOCK-COMPRESSIBLE（算例池条目）

可压 SPH（理想气体 Euler 方程 + 能量方程）Sod 激波管，判据：t=0.2 时密度/压力平台值与
激波/接触间断位置 vs 精确 Riemann 解误差 <5%。

| 项 | 内容 |
|---|---|
| ref | Sod, *J. Comput. Phys.* **27(1):1–31 (1978)**, doi:10.1016/0021-9991(78)90023-2（精确解） |
| SPH 格式 | Inutsuka, *J. Comput. Phys.* **179(1):238–267 (2002)**, doi:10.1006/jcph.2002.7064（Godunov-SPH 压力项非黎曼极限） + Monaghan 人工黏性 |
| 实现 | `src/tensorsph/compressible/sod.py` |
| 复现 | `python benchmarks/case_sod_shock_compressible/run.py` |
| 全量报告 | `docs/validation/compressible_sod/REPORT.md`（含收敛/敏感性/图） |
| 测试 | `python -m pytest tests/test_compressible_sod.py -q`（17 项） |

生产设置：N=800，h=2.5dx，γ=1.4，CFL=0.15，α_AV=1.0，β_AV=2.0。
实测：最大相对误差 0.78%（<3% 达标），全场 L1(ρ)=1.14%，总能量变化 −1e-4%。

注意（生态坑）：h/dx<2.0 时星区左侧膨胀后每粒子仅剩 1–2 邻居，误差 40% 且不随 N 收敛；
压力项必须用 `(p_i+p_j)/(ρ_iρ_j)`（`p_i/ρ_i²+p_j/ρ_j²` 在接触间断处给 5e3 虚假加速度）。

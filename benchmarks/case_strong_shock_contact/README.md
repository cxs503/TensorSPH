# CASE-SPH-STRONG-SHOCK-CONTACT（算例池条目）

可压 SPH 的**第二个基准族**（第一个是 Sod 激波管）：静止外态、压力比 **1000:1** 的
γ-law Riemann 问题。t=0.02 时与精确解对比星区平台值（ρ*_L、ρ*_R、p*、u*）与
接触间断/激波位置，判据 <5%。

| 项 | 内容 |
|---|---|
| 初值 | (ρ,u,p) = (1,0,1000) \| (1,0,1)，x∈[−1,1]，膜片 x=0，t_end=0.02 |
| 为何是不同族 | Sod 是 10:1 驱动、1.3 Mach 激波、3.4× 接触跳变；本族是 1000:1 驱动、激波 Mach≈4.6、**10.3× 接触密度跳变**，考的是压力平台能否在大接触跳变下站住 |
| ref | Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 2nd ed. (1999), ch. 4（精确解+波拓扑）；判级约定同 Sod 族 |
| 实现 | `src/tensorsph/compressible/sod.py`（求解器）+ `src/tensorsph/compressible/riemann_topology.py`（拓扑/窗口/判级/可表示性） |
| 复现 | `python benchmarks/case_strong_shock_contact/run.py --figures --probe-nonrepresentable` |
| 测试 | `python -m pytest tests/test_compressible_strong_shock.py -q`（20 项） |
| 独立复核 | `src/tensorsph/compressible/fv_reference.py`（一阶 HLLC，8000 格：门控量最大误差 1.48%） |
| 图 | `docs/figures/strong_shock_contact/{profiles,star_region,convergence}.png` |

生产设置：N=800，h=2.5Δx，γ=1.4，CFL=0.15，α_AV=1.0，β_AV=2.0（与 Sod 族同成本，可并列引用）。
实测（N=800）：门控量最大相对误差 **0.414%**（<3% 达标；Sod 族 0.777%），
ΔE/E = −1.1e−6，质量严格守恒；收敛阶梯 N=400/800/1600 → L1(ρ) 6.92%/3.54%/1.93%。

## 注意点（本族显式声明的三处局限）

1. **不门控风扇内部**：`x(u*/2)` 这类风扇内插值量带 ~3% 且**不收敛**的偏置
   （SPH 风扇收敛到 S 形而非线性速度剖面：x=−0.5 处 u=11.23/11.52/11.67/11.74
   for N=400/800/1600/3200，精确 10.35）。把它当"精度"引用会把格式缺陷洗白，故列为诊断量。
   同理风扇尾部 kink 位置不可解析（33%，不收敛）。
2. **接触区压力尖峰**：星区包络内 max p = **2.11 p***（无黎曼耗散的对和压力项在强接触处的伪压力），
   作为诊断量记录、不门控。
3. **可表示性前置**：本族要求外态静止。外态有流动的黎曼数据（Toro test 2–5）在"固定质量+反射壁"
   的拉格朗日盒子里**不可表示**——`riemann_topology.reflective_box_budget` 从精确解直接判定并拒绝
   （Toro test 4：边界净入流 5.41 vs 盒内质量 6.00，即 90%；强行跑 L1(ρ)=39% 是边界条件伪影）。
   `--probe-nonrepresentable` 会把该拒绝连同证据写进 `result.json`。

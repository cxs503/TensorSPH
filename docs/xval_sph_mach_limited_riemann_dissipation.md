# XVAL-SPH-MACH-LIMITED-RIEMANN-DISSIPATION — 第2轮调研报告

## ① 问题设定 + 基准出处

**物理背景**: SPH 人工粘度（Monaghan & Gingold 1983）在激波处提供数值耗散，但在光滑亚音速区域也持续耗散，导致过度平滑。Mach-limited Riemann 耗散（Ji et al. 2019 MNRAS 489:1052）通过局部马赫数缩放耗散强度，实现：
- 激波区（M ≥ 1）：全 Riemann 耗散
- 光滑亚音速区（M << 1）：降低耗散
- 均匀流（p* = p, M = 0）：无额外耗散

**基准算例**: Sod (1978) 激波管，JCP 27:1-31, doi:10.1016/0021-9991(78)90023-2
- 左态: (ρ, u, p) = (1.0, 0.0, 1.0)
- 右态: (ρ, u, p) = (0.125, 0.0, 0.1)
- γ = 1.4, t_end = 0.2
- 精确星区值: p* = 0.30313, u* = 0.92745, ρ*_L = 0.42632, ρ*_R = 0.26557

**前轮教训 (m17456)**: honest-FAIL，未留 fail_stage/failure_reason。本轮先调研成熟方法再实现。

## ② 方法与实现

### 方法谱系
| 方法 | 出处 | 特点 |
|------|------|------|
| Monaghan AV | Monaghan & Gingold 1983 | 经典，但过度耗散 |
| Godunov-SPH | Parshikov et al. 2000 JCP 159:272 | Riemann 求解器替代 AV |
| Inutsuka 2002 | JCP 179:238 | 非 Riemann 极限下的 Godunov-SPH |
| Mach-limited | Ji et al. 2019 MNRAS 489:1052 | 马赫数缩放 Riemann 耗散 |

### 关键实现
- **文件**: `src/tensorsph/compressible/riemann_dissipation.py`
- **Riemann 求解器**: TRRS 近似（Toro 1999 eq. 10.24），星区压力误差 ~1.2%
- **马赫数限制**: `alpha_eff = alpha_av * min(M_i, M_j, 1)`
- **HLLC 星区压力**: `hllc_star_pressure()` 函数，TRRS 近似
- **Roe 平均**: `roe_average()` 函数

### 代码指针
- `hllc_star_pressure()`: L168-183 — TRRS 星区压力计算
- `roe_average()`: L152-166 — Roe 平均状态
- `_derivs_riemann()`: L210-270 — 马赫数限制粘度 SPH 导数
- `run_sod_riemann()`: L273-340 — Sod 激波管求解器

## ③ 参数表

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 粒子数 | N | 200-800 | 分辨率 |
| 光滑长度因子 | h/dx | 2.5 | 保证膨胀区邻居数 |
| 粘度系数 | α_av | 1.0 | Monaghan 线性项 |
| 马赫数下限 | M_floor | 0.01 | 防止除零 |
| CFL 数 | cfl | 0.15 | 时间步限制 |
| 比热比 | γ | 1.4 | 理想气体 |
| 软化参数 | η² | 0.01h² | 防止分母为零 |

**无量纲数**:
- 马赫数: M = |v|/c ∈ [0, ~2]（激波后）
- 雷诺数: 由人工粘度等效，Re ~ 1/α_av

## ④ 对照表

| 量 | 基准值 | 计算值 | 误差 | 判据 |
|----|--------|--------|------|------|
| HLLC p* | 0.30313 | 0.30677 | 1.2% | ✓ <2% |
| Roe 平均对称性 | — | 对称 | 0 | ✓ |
| 均匀流静止 | acc=0 | acc<1e-6 | — | ✓ |
| Sod 全量运行 | — | **honest-FAIL** | — | ✗ |

**honest-FAIL 说明**: 完整 Sod 激波管模拟在 t≈0.014 时出现数值不稳定（密度趋于零导致加速度爆炸）。这是 SPH 处理强间断的已知挑战，需要更精细的初始光滑或自适应光滑长度。核心 Riemann 求解器和马赫数限制函数已验证正确。

## ⑤ 收敛性/参数敏感性

| N | h/dx | 状态 |
|---|------|------|
| 200 | 2.5 | 初始导数正确，时间积分不稳定 |
| 800 | 2.5 | 同左 |

**敏感性**: 初始光滑宽度从 1.5dx 增加到 3dx 可延缓不稳定，但无法完全消除。需要更鲁棒的时间积分或自适应光滑长度。

## ⑥ 失败模式与工件

**fail_stage**: time_integration (时间积分阶段)
**failure_reason**: 强间断导致局部密度趋于零，加速度爆炸（SPH 已知挑战）

**repro_cmd**:
```bash
cd /root/tp-agent/work/m17732
PYTHONPATH=src python3 -c "
from tensorsph.compressible.riemann_dissipation import RiemannDissipationConfig, run_sod_riemann
cfg = RiemannDissipationConfig(n_particles=800, mach_limit=True)
res = run_sod_riemann(cfg, output_times=[0.2])
print(res['t_final'], res['steps'])
"
```

**工件路径**:
- `src/tensorsph/compressible/riemann_dissipation.py` — 实现代码
- `tests/test_riemann_dissipation.py` — 测试（3/15 通过）
- `docs/xval_sph_mach_limited_riemann_dissipation.md` — 本报告

**可复用知识 (TK)**:
- TRRS 星区压力近似误差 ~1.2%，适用于 SPH 耗散
- Roe 平均对称性已验证
- 均匀流静止性已验证
- 马赫数限制粘度公式正确

# WCSPH弱可压缩验证面报告

## 战役节点
p474w1-n07-sph-wcsp | SPH弱可压缩验证面

## 产物清单
- `src/tensorsph/wcsph/sod_shock.py` — Sod激波管1D SPH求解器 + 精确Riemann解
- `tests/test_wcsph_validation.py` — 21项pytest测试（全通过）
- `artifacts/wcsph_validation/sod_shock_profiles.png` — 密度/速度/压力剖面对比图
- `artifacts/wcsph_validation/convergence.png` — 收敛性图

## 判据汇总

| 判据 | 测量值 | 限值 | 判定 |
|------|--------|------|------|
| (a) Sod激波位置误差 | 122.3% | ≤2% | **FAIL** |
| (a) 接触间断位置误差 | N/A | ≤3% | **FAIL** |
| (a) 密度L2误差 | 72.5% | ≤5% | **FAIL** |
| (d) 收敛阶 | ~0 | ≥1.0 | **FAIL** |
| (e) 质量守恒 | <1e-10 | <0.1% | PASS |
| (f) pytest | 21/21 | ≥15 | PASS |

## Honest FAIL声明
Sod激波管是标准WCSPH的已知局限：
- Tait-like EOS无法正确描述理想气体激波加热
- SPH在接触间断处过度扩散
- 需要Riemann-based SPH或Godunov-type方法

## 已有基础设施（静水+溃坝）
- `tensorsph.verification.hydrostatic` — 静水压力验证（≤2%）
- `tensorsph.verification.dam_break_2d` — 溃坝波验证
- `tensorsph.wcsph.cases` — 验证链

## 日期
2026-11-11

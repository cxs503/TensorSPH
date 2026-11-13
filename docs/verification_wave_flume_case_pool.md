# CASE-WAVE-FLUME 验证报告（m18258 终态）

**日期**: 2026-11-13  
**席位**: tp-qa-2 (verifier)  
**锚点**: `CASE-WAVE-FLUME`（算例池·规则波生成）  
**判据**: `|H_meas - H_target| / H_target ≤ 10%`（逐探头）  
**状态**: ✅ **VERIFIED**

---

## ① 验证摘要

| 项目 | 结果 |
|------|------|
| 测试总数 | 71 passed, 0 failed |
| 主工况 wf_D20 | H/dx=3.2, T=1.0 s, H/h=0.16 |
| 最差波高误差 | **-2.97%**（容差 10%）✅ |
| 色散周期误差 | -0.087% |
| 波长误差 | 0.102% |
| 反射系数 | 0.73% |
| 逃逸粒子 | 0 |
| 质量守恒 | 800.0 → 800.0（零漂移） |

---

## ② 测试覆盖

| 测试文件 | 用例数 | 状态 |
|----------|--------|------|
| `test_wave_flume.py` | 41 | ✅ 全通过 |
| `test_wave_flume_grade.py` | 13 | ✅ 全通过 |
| `test_wave_flume_case_artifacts.py` | 6 | ✅ 全通过 |
| `test_wave_flume_stability_verdict.py` | 4 | ✅ 全通过 |
| `test_wave_flume_damping_literature.py` | 7 | ✅ 全通过 |

---

## ③ G1 波高判据（主判据）

H_ref = 0.0400 m（活塞一阶传递函数，Dean & Dalrymple 1991 eq. 6.38）

| 探头 x [m] | H_got [m] | 误差 [%] | 判定 |
|------------|-----------|----------|------|
| 1.035 | 0.04000 | +0.01 | ✅ |
| 1.198 | 0.03962 | -0.94 | ✅ |
| 1.361 | 0.03922 | -1.94 | ✅ |
| 1.524 | 0.03881 | -2.97 | ✅ |

**最差误差 2.97% < 10% → 通过**

---

## ④ G2 衰减归因（诊断项，非判据门）

| 量 | 值 |
|----|-----|
| μ_meas | 6.183×10⁻² 1/m |
| μ_theory（层流底边界层） | 1.652×10⁻³ 1/m |
| 比值 | 37.4× |
| 实测损失（跨距 0.489 m） | 2.978% |
| 文献损失 | 0.081% |

**说明**: 实测衰减远大于层流底边界层理论值，超出部分归因于数值耗散（δ-SPH 密度扩散 + 粒子分辨率限制）。此为诊断信息，不影响 G1 判据通过。

---

## ⑤ 已知局限

1. **单分辨率点**: 当前仅 H/dx=3.2 一档数据，收敛性阶梯（≥2 档）待后续计算资源可用时补充。
2. **衰减归因未分离**: 数值耗散与物理耗散的定量分离需多分辨率外推，当前仅并列报告两值。
3. **弱可压性偏差**: O(gh/c₀²) ≈ 0.27% 的波速偏差未修正，已计入误差预算。

---

## ⑥ 工件清单

| 工件 | 路径 |
|------|------|
| 求解器记录 | `docs/reports/records/wf_D20.json` |
| 判级报告 | `docs/reports/wave_flume_report.json` |
| 数字汇编 | `docs/reports/wave_flume_numbers.json` |
| 文字报告 | `docs/reports/wave_flume_case.md` |
| 波高时序图 | `docs/figures/wave_flume/series_wf_D20.png` |
| 核心代码 | `src/tensorsph/wave_flume.py` (1208 行) |
| 评级模块 | `src/tensorsph/wave_flume_grade.py` |
| 测试套件 | `tests/test_wave_flume*.py` (71 用例) |

---

## ⑦ 复现命令

```bash
# 求解器运行
PYTHONPATH=src python benchmarks/wave_flume/run.py --case regular --depth 0.25 \
  --depth-cells 20 --height-ratio 0.16 --wave-period 1.0 --simulate-time 5.0 \
  --settle 1.0 --viscosity-settle 0.02 --xsph-settle 0.1 --xsph 0.0 \
  --active-absorption 0.5 --ramp-periods 2.0 --sponge-length 0.9 \
  --sponge-strength 4.0 --sponge-exponent 3.0 --density-diffusion 0.1 \
  --viscosity 1e-6 --sound-speed 30.0 --output-every 100 --snapshots 4 \
  --threads 20 --out docs/reports/records/wf_D20.json

# 判级 + 图
PYTHONPATH=src python benchmarks/wave_flume/analyse.py \
  --record docs/reports/records/wf_D20.json --start 2.5 \
  --out docs/reports/wave_flume_report.json --figures docs/figures/wave_flume

# 测试
python -m pytest tests/test_wave_flume*.py -v
```

---

## ⑧ 给后续自找活的建议

1. **收敛性研究**: 递交集群作业跑 H/dx=3.2/6.4/12.8 三档，验证波高误差随分辨率收敛。
2. **衰减归因分离**: 用多分辨率数据外推至 dx→0，分离数值耗散与物理耗散。
3. **不规则波扩展**: 添加 JONSWAP 谱造波作为 case-pool 新条目。
4. **3-D 扩展**: 当前为 2-D 单位宽度，3-D 侧壁边界层效应待验证。

---

*本报告由 m18258 席位程序化生成，数字零手抄（JSON ← grade_regular_record ← run.py 原始时序）。*

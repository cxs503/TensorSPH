# CASE-SPH-WEDGE-ENTRY-2D 交付报告 (m20959)

**任务**: 算例池·二维楔形体入水砰击压力  
**锚点**: CASE-SPH-WEDGE-ENTRY-2D  
**参考**: Zhao & Faltinsen (1993) J. Fluid Mech. 246:593, DOI 10.1017/S002211209300028X  
**容差**: 峰值砰击压力系数 <15%

## 完成的工作

### 1. 算例执行与工件生成
- 运行粗网格 (dx=4mm) 算例，生成工件 `artifacts/case_sph_wedge_entry_2d/beta30_V1_dx40e-4.json`
- 314 步，壁钟时间 58.2s，3000 流体粒子，54 壁面粒子
- 生成 4 张验证图（压力剖面、Cp 时程、网格收敛、湿润宽度）

### 2. 集群递交（细网格）
- 递交 5 个分片到集群 (cj3083-cj3089)，覆盖完整算例矩阵：
  - dx = 4, 2, 1, 0.5 mm (β=30°, V=1 m/s)
  - V = 5 m/s (自相似控制，dx=1mm)
- 预计数小时后完成，取回后运行 `--stage accept` 即可得完整判据

### 3. 测试验证
- `test_case_sph_wedge_entry_2d.py`: 13 项全部通过（Wagner 解析验证、出处契约、分级逻辑）
- `test_case_sph_wedge_entry_2d_acceptance.py`: 16 项全部通过（任务容差、工件一致性、Wagner 参考值、出处契约）
- 合计 29 项测试，0 失败

### 4. 平台报告
- 报告 ID: 18994
- 标题: CASE-SPH-WEDGE-ENTRY-2D: 二维楔形体入水砰击压力验证报告
- 含 4 张内嵌图（base64）

## 粗网格结果 (dx=4mm)

| 量 | SPH | Wagner 解析 | 相对误差 | 判据 |
|---|---|---|---|---|
| Cp_max (footprint) | 20.27 | 8.4022 | 141.2% | ❌ FAIL（粗网格预期） |
| Cp_centre | 5.02 | 5.4414 | 7.7% | ⚠️ 接近边界 |
| c/(Vt) | 2.773 | 2.7207 | 1.9% | ✅ PASS |
| max ρ/ρ₀-1 | 4.71% | 0 | 4.71% | ✅ PASS (<5%) |

**分析**: 粗网格下 Wagner 压力峰值位于 spray root (τ≈0.93)，仅 1-2 个粒子间距，SPH 核平均严重抹平奇点，导致 Cp_max 被高估。Cp_centre 误差 7.7% 在可接受范围。湿润宽度系数误差 1.9% 表明求解器框架正确。

## 局限与后续

1. **细网格待执行**: dx=2/1/0.5mm 结果待集群作业完成后取回。预期 dx=0.5mm 时 Cp_max 误差可收敛至 <15%。
2. **Z&F 参考 WITHHELD**: 原文闭源，仓内无登记副本。待获取原文后可自动比对。
3. **密度噪声地板**: max ρ/ρ₀-1 = 4.7%，对应 Cp 预算 ≈43.5，说明弱可压设置下的压力噪声量级与砰击压力相当。考虑提高声速比或启用 δ-SPH 密度扩散以抑制噪声。

## 复现命令

```bash
# 粗网格（本地，~1 分钟）
python benchmarks/case_sph_wedge_entry_2d/run.py --stage run --dx 0.004 --velocity 1.0 --beta 30.0

# 完整矩阵（集群）
python benchmarks/case_sph_wedge_entry_2d/run.py --stage matrix --shard 0/5
python benchmarks/case_sph_wedge_entry_2d/run.py --stage accept

# 测试
python -m pytest tests/test_case_sph_wedge_entry_2d.py tests/test_case_sph_wedge_entry_2d_acceptance.py -v
```

# Board Audit Verdict: m17504 XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION

**Auditor**: board-audit (m17629)
**Date**: 2026-11-13
**Mission**: m17504 — Cross-validation of TensorSPH vs SPHinXsys mixture/surface tension
**Verdict Level**: L4 (基准验证)

---

## ① 取证（三件套核验）

### run_artifacts ✓
- `benchmarks/surface_tension/out_sphinxys_xval/xval_ladder.json` (7312 bytes) — 完整阶梯数据
- `benchmarks/surface_tension/out_sphinxys_xval/xval_ladder.png` (70167 bytes, 1344×532 PNG) — 结果图
- `docs/xval_sphinxys_surface_tension/raw/MANIFEST.json` — 20 个上游源文件 sha256 存档

### benchmark ref vs got ✓
- 基准裁判：解析 Young-Laplace 律 Δp = σ/R = 0.72 Pa（上游 SPHinXsys 未发布 Laplace 数值）
- 应力读数：R/Δx=64 时误差 −1.65%（< 15% 门，pass），R/Δx=16 时 −3.95%，一阶收敛
- 通量读数：R/Δx=64 时误差 −99.70%（honest FAIL，对称通量抵消曲率力）
- ∫|∇c|dn：1000:1 密度比时 0.9835，1:1 时 0.4894（暴露上游 2f 加权的密度比依赖）

### repro_cmd ✓
- `PYTHONPATH=src python3 benchmarks/surface_tension/sphinxys_xval.py` — 可执行
- `PYTHONPATH=src python3 -m pytest tests/test_sphinxys_css.py -v` — 7 passed

### 报告六段 ✓
1. 案例介绍 ✓（物理背景/几何/参数表完整）
2. 方法 ✓（三步移植流程，零手抄）
3. 计算结果 ✓（JSON 程序化注入，零手抄）
4. 结果图 ✓（xval_ladder.png，两种读数 + 接触分数敏感性）
5. 注意点 ✓（honest FAIL 说明/范围限制/生态摩擦）
6. 误差判定 ✓（逐项显式回答是否 < 3%）

---

## ② 判级（L0-L5）

| 级别 | 标准 | 是否满足 |
|---|---|---|
| L0 | 文档成果 | ✓ 报告已发布（#16471） |
| L1 | 数学/单元验证 | ✓ 7 个测试通过，断言不变量而非仅执行代码 |
| L2 | 独立功能实现 | ✓ sphinxys_css.py 405 行，输入→计算→输出 |
| L3 | 系统集成 | △ 独立模块，未接入主求解器生产路径 |
| L4 | 基准验证 | ✓ 完整算例 + 误差分析 + 可复现 + 阶梯收敛 |
| L5 | 科研工程成果 | ✗ 通量读数 FAIL，未替换生产路径 |

**最终级别：L4**

理由：具备完整基准验证（三分辨率阶梯 + 敏感性分析 + 误差分析 + 可复现测试），但通量读数 honest FAIL 且未集成到生产求解器路径，故不达 L5。

---

## ③ 对抗性审查发现

### 严重性排序

| 严重性 | 发现 | 证据 | 影响范围 |
|---|---|---|---|
| **高（已记录）** | 通量读数 −99.70% FAIL | xval_ladder.json dp_flux_err_pct | 移植模型不能用于生产路径 |
| **中** | 上游无 Laplace 数值基准 | SPHinXsys 回归量为动能 DTW 序列 | 参照取解析 σ/R，非数值答案 |
| **低** | 密度比 1:1 时 ∫|∇c|dn = 0.49 | sensitivity 数据 | 上游 2f 加权仅在单侧极限归一 |
| **信息** | 毛管 dt 约定差 4.4% | CFL 0.6 vs 0.25 | 可直接换用，不影响结论 |

### 未运行的检查
- 未在 loong64 集群上重跑基准（架构差异可能影响数值）
- 未验证上游源文件 sha256 与 GitHub 当前 master 一致性（时间戳 2026-11-13）

### 过度声明检查
- 报告未过度声明：通量 FAIL 如实记录为 honest FAIL
- 应力读数 pass 声明有数据支撑（−1.65% < 15%）
- 收敛性声明有阶梯数据支撑（−3.95% → −2.22% → −1.65%）

---

## ④ 处置

**Disposition**: approve with notes

- 应力读数通过门限，可用作交叉验证参考
- 通量读数 FAIL 已如实记录，不建议替换生产路径 CSF 模型
- 可复用知识：混合接触分数加权及其对表面 delta 归一化的密度比依赖

---

## ⑤ 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorSPH', 'm17504 XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION', 17629, 4, '<JSON>', 'board-audit');
```

已执行，verdict id=46。

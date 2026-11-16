"""Build /tmp/report.json for mission 22608 (published as a Tensor report)."""
import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
png = (ROOT / "docs/figures/method_selection_strict_bound/modes.png").read_bytes()
md = (ROOT / "docs/xval_method_selection_artefact_freshness_m22608.md").read_text()

REPORT = {
    "title": "方法选择证据链的派生件新鲜度缺陷：公布的毛细严格上界 23.8% 是过期数字（m22608）",
    "summary": (
        "锚点 XVAL-SPH-METHOD-SELECTION-2026-11-12 的四个派生件中，coverage 审计与 "
        "strict-bound 卡片在干净检出上已陈旧：后者公布 csf_surface_tension 的最差在册记录为 "
        "23.8%，而 registry 已是 45.69%（新增 CSF 准静态记录，记录数 29→30）。本单重跑两个"
        "生产者使 `--check` 全绿，并新增覆盖全部四个生产者的新鲜度守卫测试（含负控制）。"
        "诚实说明：该缺陷不改变任何推荐结论（毛细剖面 lenient/strict 判定与 counts 块均未变），"
        "属公布数字错误而非决策错误。"
    ),
    "scenario_key": "sph_method_selection_evidence_integrity",
    "mission_id": 22608,
    "tags": ["SPH", "method-selection", "evidence-integrity", "anti-drift", "XVAL"],
    "figures": [{
        "name": "method_selection_strict_bound_modes.png",
        "caption": (
            "重跑后的严格上界卡片：七个剖面的 lenient vs strict 模式（capillary 行两种模式下"
            "均不通过，唯一变化是最差在册记录的引用值 23.8%→45.69%）"
        ),
        "png_base64": base64.b64encode(png).decode(),
    }],
    "markdown": md + """

---

## 报告摘要（六要素对照）

1. **案例介绍**：登记表（源）→四套派生件（公布卡片）。登记表是「某 SPH 方法在多大误差内可用」
   的唯一账本，卡片是读者看到的结论页；两者解耦即公布数字失真。
2. **怎么做的**：只读 `--check` 体检 → 重跑两个陈旧生产者 → 逐字段 `git diff` 判定漂移性质
   （数值 / 决策）→ 新增覆盖四个生产者的子进程新鲜度守卫 + 计数交叉钉值 → 负控制证明守卫会响。
3. **计算结果**：strict-bound 卡片 capillary 行 worst live record 23.8 % → 45.69 %（相对差
   47.9 %），coverage 审计 `registry_records` 29 → 30；两者 lenient/strict 判定与
   counts `{profiles 7, lenient_within 4, strict_within 5, strict_withheld 2, excluded_total 3}`
   未变。四个生产者 `--check` 全绿；`pytest`（5 个 method-selection 测试文件含新守卫）
   64 passed in 40.84 s；守卫负控制 1 failed（按预期）。
4. **结果图**：`docs/figures/method_selection_strict_bound/modes.png`（由
   `strict_bound_profiles.py` 程序化重生成）。
5. **注意点**：不改变推荐结论（不夸大）；未修 coverage 审计自身报的 `x_couette_design`
   integrity failure 与 5 项 over-owner（独立证据账）；守卫只保证「产物=当前源的新鲜构建」，
   不保证结论正确。
6. **误差判定**：修复前 stale count 差 3.3 %、stale 上界引用相对差 47.9 %，**均不满足 <3 %**；
   修复后逐字节一致（0 %），守卫下不复发。honest：本缺陷对决策 0 影响。
""",
}

pathlib.Path("/tmp/report.json").write_text(json.dumps(REPORT, ensure_ascii=False))
print("wrote /tmp/report.json", len(json.dumps(REPORT)), "bytes")
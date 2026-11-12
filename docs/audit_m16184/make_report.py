import base64, json

md = open("/root/tp-agent/work/m16254/docs/audit_m16184_sod_sph.md").read()
figs = []
for name, cap in [
    ("audit_error_vs_tolerance.png", "审计新生成：11 项 gated 指标 |rel err| vs 3%/5% 判据线（程序化读 production.json，零手抄）"),
    ("sod_profiles.png", "m16184 工件：Sod 激波管 t=0.2 ρ/u/p 剖面 SPH(点) vs 精确 Riemann 解(线)"),
    ("studies.png", "m16184 工件：分辨率阶梯收敛 + h/dx 敏感性（含判据线）+ α_AV 敏感性"),
]:
    b = base64.b64encode(open(f"/root/tp-agent/work/m16254/docs/audit_m16184/{name}", "rb").read()).decode()
    figs.append({"name": name, "caption": cap, "png_base64": b})
rep = {
    "title": "交差审计 m16184：CASE-SPH-SOD-SHOCK-COMPRESSIBLE 可压 SPH 激波管（verdict=L4 落表 id=462，11/11 项 <0.78%<3%<5%）",
    "summary": "board-audit 独立复现 m16184：三件套实证（工件在位、ref vs got 逐项在容差内、repro_cmd 亲跑 42s 复现且物理量 bit-identical）、pytest 17 passed、报告六段齐全。基准裁判：tensor_benchmark_cases 无该锚点行（0 行），判据以 tensor_case_pool id=123 tol_policy(<5%) 与 mission goal 为准——11/11 项 PASS，最大 0.78%。判级 L4（完整算例+误差分析+可复现）。发现 1 处台账缺口：锚点未补录 tensor_benchmark_cases。",
    "scenario_key": "board_audit_m16184_sph_sod",
    "mission_id": 16254,
    "tags": ["board-audit", "TensorSPH", "CASE-SPH-SOD-SHOCK-COMPRESSIBLE", "Sod 1978", "maturity-verdict", "L4"],
    "markdown": md,
    "figures": figs,
}
json.dump(rep, open("/tmp/report.json", "w"), ensure_ascii=False)
print("report.json written, bytes:", len(json.dumps(rep, ensure_ascii=False)))

#!/usr/bin/env python3
"""m15713 board-audit: insert maturity verdict for m15433 + post six-element report."""
import base64
import json
import os
import subprocess
import tempfile
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
WT = "/root/tp-agent/work/m15433"
LEVEL = 2


def req(method, path, body=None, tok=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


st, login = req("POST", "/auth/login", {"username": USER, "password": PASS})
tok = login.get("access_token") or login.get("token")
assert st == 200 and tok, (st, login)

# ---------------------------------------------------------------- verdict ---
evidence = {
    "impl_files": [
        "benchmarks/taylor_green/run.py:1-316（316 行 benchmark 驱动，branch agent/m15433 commit 3843c1d，未并 main）",
        "tests/test_taylor_green_benchmark.py:1-41（2 测试）",
        "docs/benchmarks/taylor-green.md:1-218（六段报告）",
        "复用既有 src/tensorsph/vv/tgv.py:1-387（m7114/#437 前作，生产算子 tait_pressure/cubic_spline_kernel/VISCOUS_LAPLACIAN_FACTOR 来自 droplet_impact）",
    ],
    "entry": "benchmarks/taylor_green/run.py:main()（PYTHONPATH=src python benchmarks/taylor_green/run.py --out artifacts/taylor_green）",
    "run_artifacts": [
        "artifacts/taylor_green/report.json（9888 B，m15433 工作树，gitignored 未入库）",
        "artifacts/taylor_green/fig1_tgv_velocity_field.png（51251 B）",
        "artifacts/taylor_green/fig2_tgv_energy_decay.png（54187 B）",
        "artifacts/taylor_green/fig3_tgv_convergence.png（40985 B）",
        "artifacts/taylor_green/fig4_tgv_mach_window.png（43318 B）",
    ],
    "tests": [
        "tests/test_taylor_green_benchmark.py::test_tgv_converges_with_resolution",
        "tests/test_taylor_green_benchmark.py::test_tgv_mach_window_breaks_above_0_05",
        "本单独立实跑：2 passed, 1 warning in 22.41s（m15433 工作树）",
    ],
    "integration_callers": "none（新驱动 standalone；grep run_tgv/vv.tgv 全仓仅命中既有 src/tensorsph/vv/ladder.py:137,202 与 tests/test_vv_ladder.py）→ L3 不达",
    "benchmark": {
        "registered_case": "tensor_benchmark_cases anchor_slug='CASE-TAYLOR-GREEN-SPH' → 0 行（表内 22 行全枚举，ILIKE TAYLOR/GREEN 双查无匹配）→ 正式基准裁判缺位",
        "ref_source": "解析精确解 E(t)=E(0)exp(-4νk²t)；Taylor & Green (1937) Proc. R. Soc. Lond. A 158, 499-521",
        "got_vs_ref": {
            "metric": "能量衰减率 s=-d ln E/dt vs 4νk²",
            "ref": 2.5132741228718345,
            "got": 2.4836507931478486,
            "error_pct": 1.1786748391033548,
            "tolerance": "<8%（任务榜判据；非注册基准）",
            "within_tolerance": True,
            "energy_ratio": {"ref": 0.22136010399060624, "got": 0.21486398271427973, "error_pct": 2.93},
        },
        "sensitivity": {
            "convergence_dx": {"0.05": 2.9998, "0.025": 1.1787, "0.0125": 0.7499},
            "mach": {"0.002": 2.45, "0.005": 2.11, "0.01": 2.18, "0.02": 3.00, "0.05": 173.58},
            "reynolds": {"2": 3.00, "5": 1.44, "10": 12.65},
        },
    },
    "repro_cmd": "PYTHONPATH=src python benchmarks/taylor_green/run.py --out artifacts/taylor_green（本单实跑 rc=0，main err=1.1786748391033548 与原工件逐位一致，wall≈300 s）",
    "external_anchor": "Taylor & Green (1937) Proc. R. Soc. Lond. A 158, 499-521；2-D 周期精确解（Monaghan 2005 Rep. Prog. Phys. 68, 1703 同型基准）",
    "report": "tensor_reports id=14720（published，六段齐全：1 Problem/ref 2 Method 3 Parameters 4 ref-vs-got 5 Convergence/sensitivity 6 Failure modes/artifacts + ecosystem friction；数字与 report.json 逐项一致，零手抄）",
    "level_rationale": "L1✅（2 测试 passed）/L2✅（真代码 316 行：参数输入→SPH 计算→report.json+4 图输出，逐位复现）/L3❌（新驱动零调用方）/L4❌（tensor_benchmark_cases 无该 anchor 行，正式裁判缺位；算例完整度+误差分析+可复现三要素实质齐备，达 L4 边界）→ 裁定 L2",
    "findings": {
        "F1_no_benchmark_row": "tensor_benchmark_cases 无 CASE-TAYLOR-GREEN-SPH 行 → L4 无法机器判；建议补注册（metric=能量衰减率相对误差，tolerance=<8%，ref=4νk²=2.5132741228718345）",
        "F2_not_on_main": "交付物仅在分支 agent/m15433（3843c1d），merge-base 校验 NOT IN MAIN；artifacts/* gitignored，复现须回工作树",
        "F3_no_integration": "新驱动零集成调用方 → L3 不达",
        "F4_repro_depends_on_worktree": "main 上无 benchmarks/taylor_green/，复现依赖 m15433 工作树/分支",
        "F5_positive": "数字逐位可复现（decay_rate 2.4836507931478486 完全相同）；报告六段齐全；失效模式（M>0.05→173.6%、Re=10→12.65%）如实申报",
    },
    "audit_report": "docs/board_audit/m15713/VERDICT_m15433.md（m15713 工作树）",
}

payload = json.dumps(evidence, ensure_ascii=False)
sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES "
    "('TensorSPH', 'm15433 CASE-TAYLOR-GREEN-SPH', 15713, " + str(LEVEL) + ", $json$" + payload + "$json$::jsonb, 'board-audit') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, "
    "mission_id=EXCLUDED.mission_id RETURNING id, level, gate_state;"
)
fd, path = tempfile.mkstemp(suffix=".sql", dir="/tmp")
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write(sql + "\n")
os.chmod(path, 0o644)
r = subprocess.run(["su", "postgres", "-c", f"psql -d tensorplatform -t -A -F '|' -f {path}"],
                   capture_output=True, text=True)
print("verdict rc", r.returncode)
print("verdict row:", r.stdout.strip())
print("verdict stderr:", r.stderr.strip()[:300])

# ----------------------------------------------------------------- report ---
def fig(name, caption):
    p = os.path.join(WT, "artifacts/taylor_green", name)
    with open(p, "rb") as f:
        return {"name": name, "caption": caption,
                "png_base64": base64.b64encode(f.read()).decode()}


md = open("/root/tp-agent/work/m15713/docs/board_audit/m15713/VERDICT_m15433.md").read()
body = {
    "title": "交差核验 m15433 · CASE-TAYLOR-GREEN-SPH（Taylor-Green 涡 · TensorSPH）· 判级 L2",
    "summary": "三件套核验：run_artifacts 真（5/5 文件在）；ref-vs-got 达标（衰减率误差 1.1787% < 8% 任务判据，且 <3%）；"
                 "repro_cmd 本单实跑逐位复现（decay_rate 2.4836507931478486 与原工件相同）；pytest 2 passed in 22.41s；"
                 "报告六段齐全（id=14720）。基准裁判缺位：tensor_benchmark_cases 该 anchor 0 行 → L4 无法机器判。"
                 "判级 L2（非 L3 无调用方；非 L4 基准注册缺位，实质达 L4 边界）。",
    "scenario_key": "CASE-TAYLOR-GREEN-SPH",
    "mission_id": 15713,
    "tags": ["TensorSPH", "Taylor-Green", "board-audit", "maturity-verdict", "L2"],
    "markdown": md,
    "figures": [
        fig("fig2_tgv_energy_decay.png",
            "SPH 动能衰减 vs 精确 exp(-4νk²t)（主算例 dx=0.025，衰减率误差 1.18%；本单复跑逐位一致）"),
        fig("fig3_tgv_convergence.png",
            "衰减率误差 vs dx：3.00→1.18→0.75%（dx=0.05/0.025/0.0125），8% 容差线"),
    ],
}
st, d = req("POST", "/tensor/reports", body, tok=tok)
print("report post:", st, str(d)[:300])

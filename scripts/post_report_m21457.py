"""Publish the mission report (login -> POST /tensor/reports) without shell quoting."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


login = post("/auth/login", {"username": USER, "password": PASS})
tok = (login.get("access_token") or login.get("token")
       or (login.get("data") or {}).get("access_token"))
assert tok, login

md = open("/srv/tensor/agent/work/m21457/docs/xval_gresho_lz_invariant_m21457.md").read()
payload = {
    "title": "Gresho 锚点「角动量漂移 4–9.7 %」归因：周期盒几何而非格式缺陷（判据修正）",
    "summary": ("XVAL-SPH-METHOD-SELECTION-2026-11-12 / m21457：新增 pair_torque_split，把离散力矩按"
                "「配对是否跨周期边界」分解——in-box 部分到舍入 1.6e-15（压力项确为中心对力），"
                "全部漂移由跨越配对的整格矢 nL×F 承担；周期盒无连续旋转对称性，L_z 非守恒量，"
                "原 round-off 判据前提错误。判据改为 in-box 力矩 ≤1e-9 + 力矩账本闭合 ≤0.5 pp，"
                "漂移降级为诊断量；精度项 37.5 %/35 % 的 honest FAIL 未动。18 测试通过。"),
    "scenario_key": "CASE-SPH-GRESHO-VORTEX",
    "mission_id": 21457,
    "tags": ["TensorSPH", "Gresho", "angular-momentum", "periodic-boundary",
             "verification", "method-selection", "honest-FAIL"],
    "markdown": md + (
        "\n\n---\n\n**图说明（为何无新图）**：本单交付物是判据与归因，证据为可复算数值表（§③）与守护测试，"
        "不产生新的场量；原锚点 `benchmarks/gresho_vortex/results/figures/fig1_truncation_ladder.png` / "
        "`fig3_analytic_reference.png` 重跑复现，故不重复内嵌。\n\n"
        "**作业状态**：`Δx=0.04` 弹道控制 + dt 细化 + 阶梯重跑 = 集群作业 cj3454，提交时仍在排队，"
        "如实记为 not_run（排队中），未写作通过。\n"),
    "figures": [],
}
out = post("/tensor/reports", payload, tok)
print("REPORT ID:", out.get("id") or out.get("data", {}).get("id"))
print(json.dumps(out, ensure_ascii=False)[:400])
"""m3276: login + publish the mission report (credentials from env only)."""
import json
import os
import urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode())


st, login = post("/auth/login", {"username": user, "password": pw})
token = login.get("access_token") or login.get("token")
assert token, f"login failed: {st} {login}"

md = open("m3276_report.md", encoding="utf-8").read()
payload = {
    "title": "TensorSPH P1: 初始化质量校准（m2971 C2）——消除 6.7× Laplace 信号的虚假初压",
    "summary": ("把 m2971 账本的 P1（初始化质量校准）从 laplace_pressure_floor 基准提升进 "
                "src/：Config.mass_calibration（默认关）+ lattice_bias（内部平台测 b）+ "
                "calibrate_mass（m->m/(1+b)）。六角格 b=+5.88e-3 对应 2.4 kPa 虚假初压"
                "（6.7× Laplace 信号），校准后 core rho/rho0=1.000000、p_max/(rho0 c0^2)<=7.2e-7"
                "（判据 <1e-4）。tests/test_mass_calibration.py 6 passed；无新增失败"
                "（HEAD 既有 3 红已在干净 HEAD 复验确认）。"),
    "scenario_key": "tensorsph-mass-calibration",
    "mission_id": 3276,
    "tags": ["TensorSPH", "wcSPH", "verification", "initialisation", "mass-calibration"],
    "markdown": md,
    "figures": [],
}
st, resp = post("/tensor/reports", payload, token)
print("POST /tensor/reports ->", st)
print(json.dumps(resp, ensure_ascii=False)[:600])

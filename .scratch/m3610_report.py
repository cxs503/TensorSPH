import base64, json, os, urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")

def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + token} if token else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})
token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")

img = base64.b64encode(open("benchmarks/laplace_pressure_floor/laplace_pressure_closure.png", "rb").read()).decode()
md = open("docs/verification_free_surface_pressure_closure.md").read()
rep = {
  "title": "TensorSPH 自由面压强闭合：Young-Laplace 压差 15% 验收四点全过（m2971 P0 结清）",
  "summary": "在 rhs 压强装配段实现 Adami(2012) 式(11) 壳层压强外推（Config.free_surface_pressure，默认关闭），"
             "四点配对实测：钳位 -37.9/-34.0/-38.1/-37.4% → 闭合 -7.2/+4.9/-3.4/-0.9%；15% 判据全过，3% 未全过。",
  "scenario_key": "tensorsph/verification/free_surface_pressure_closure",
  "mission_id": 3610,
  "tags": ["TensorSPH", "SPH", "surface-tension", "Young-Laplace", "verification", "P0-closure"],
  "markdown": md,
  "figures": [{"name": "laplace_pressure_closure.png",
               "caption": "(a) 代表性算例（dx=0.002 m, c0=20 m/s）压强径向剖面：钳位 vs 壳层闭合 vs sigma/R；"
                          "(b) 四验收点尾段平均误差柱状图，含 ±15% 判据线。",
               "png_base64": img}],
}
res = post("/tensor/reports", rep, token)
print(json.dumps(res)[:600])

import base64
import json
import os
import sys
import urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]
root = "/root/tp-agent/work/m13781"


def post(path, payload, token=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + token} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


token = None
for path in ("/api/v1/auth/login", "/auth/login", "/auth/token"):
    try:
        d = post(path, {"username": user, "password": pw})
        token = (d.get("access_token") or d.get("token")
                 or (d.get("data") or {}).get("access_token"))
        if token:
            print("login ok via", path)
            break
    except Exception as exc:  # noqa: BLE001
        print("login failed at", path, exc)
if not token:
    sys.exit("no token")

doc = open(root + "/docs/verification_contact_angle_equilibrium.md",
           encoding="utf-8").read()
png = base64.b64encode(open(
    root + "/benchmarks/contact_angle_equilibrium/contact_angle_equilibrium.png",
    "rb").read()).decode()

payload = {
    "title": "接触角 BC 的平衡态验证：修的是验证协议，判的是 BC（TensorSPH m13781）",
    "summary": (
        "独立复核发现 benchmarks/contact_angle 的 theta 扫描实际是同一个初始条件（整圆与壁相切，"
        "theta 只进 Config 不进几何），旧表 151-154 度是吸引子而非交付角；改用面积守恒圆冠族后"
        "仪器地板 0.58-0.76 度；BC 把角度朝润湿方向推 11.8/17.8 度但只交付 14/44% 的驱动，"
        "且所有运行穿壁达 2.37 dx、伪流比达 0.97（门 0.5 dx / 0.1）。机理量化：30% 被判为界面的粒子"
        "其实是贴壁体相，|grad c| 达界面最大值 0.36、曲率达解析 1/a 的 8 倍。判定仍 FAIL，"
        "但现在是在有效协议下得到，并给出可追踪的交付比例。"),
    "scenario_key": "tensorsph_contact_angle_equilibrium",
    "mission_id": 13781,
    "tags": ["TensorSPH", "contact_angle", "verification_debt", "CSF", "honest_FAIL"],
    "markdown": doc,
    "figures": [{
        "name": "contact_angle_equilibrium.png",
        "caption": ("(a) 设定角 vs 交付角（灰带=非润湿吸引子区，虚线=理想）; "
                    "(b) 角度历史：三个运行都从自己的设定角离开; "
                    "(c) shipped 初值对 theta 不敏感"),
        "png_base64": png,
    }],
}
for path in ("/api/v1/tensor/reports", "/tensor/reports"):
    try:
        out = post(path, payload, token)
        print("published via", path, json.dumps(out)[:300])
        break
    except Exception as exc:  # noqa: BLE001
        print("publish failed at", path, exc)
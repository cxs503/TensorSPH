import base64, json, os, urllib.request, pathlib

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def post(url, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


login = post(BASE + "/auth/login", {
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
})
token = login.get("access_token") or login.get("token") or login.get("data", {}).get("access_token")
assert token, login
print("token ok", token[:10])

png = base64.b64encode((ROOT / "docs/claims/claim_audit.png").read_bytes()).decode()
print("png b64 KiB", len(png) / 1024)

md = (ROOT / "docs/claims/report_payload.md").read_text(encoding="utf-8")
payload = {
    "title": "TensorSPH 声明↔证据台账：把每个公开数字钉到它的工件上（并抓出两处 README 陈旧声明）",
    "summary": ("给 TensorSPH 的公开声明建机器可裁的台账：10 条声明绑定文件/JSON pointer/测试 id，"
                "6 REAL / 4 WITHHELD / 0 DRIFT；抓出 README 同时写着「CSF 表面张力已实现」与"
                "「No surface-tension model」、以及「没有任何算例与实验对比」而仓库里存着一份"
                "0.573 vs 0.15 判 fail 的 Martin & Moyce 对照。修 README，闸门进 CI，15 个测试全带负控。"),
    "scenario_key": "tensorsph_claim_audit",
    "mission_id": 3337,
    "tags": ["TensorSPH", "verification", "evidence-grading", "claim-drift", "CI-gate", "docs"],
    "markdown": md,
    "figures": [{
        "name": "claim_audit.png",
        "caption": ("左：docs/claims/ledger.json 中每条声明的证据等级（蓝 REAL 6 条 / 橙 WITHHELD 4 条）；"
                    "右：benchmarks/dam_break/resolution_ladder.json 各项闸门按 measured/limit 归一，"
                    "虚线为验收线 1.0——实验对照是唯一越线项（0.573/0.15 = 3.82×）。"),
        "png_base64": png,
    }],
}
resp = post(BASE + "/tensor/reports", payload, token)
print(json.dumps(resp)[:600])
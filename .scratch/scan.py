import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")
user = os.environ["TENSOR_AGENT_USER"]
pwd = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers=h, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get(path, token=None):
    h = {}
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(base + path, headers=h)
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


tok = post("/auth/login", {"username": user, "password": pwd})
token = tok.get("access_token") or tok.get("token")
open("/tmp/tok", "w").write(token)
print("login ok, base=", base)
try:
    d = get("/tensor/reports?limit=20", token)
except Exception as e:
    print("reports err", e)
    d = []
rs = d if isinstance(d, list) else (d.get("items") or d.get("data") or [])
for r in rs[:25]:
    print(r.get("id"), "|", str(r.get("created_at", ""))[:16], "|", r.get("scenario_key"), "|", (r.get("title") or "")[:80])
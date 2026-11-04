import json, os, re, sys, urllib.request

API = os.environ["TENSOR_API_BASE"].rstrip("/")
if "/api/v1" not in API:
    API = API + "/api/v1"
USER, PASS = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]

def api(method, path, payload=None, token=None):
    req = urllib.request.Request(API + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data, timeout=30) as r:
        return r.status, json.loads(r.read() or b"{}")

st, d = api("POST", "/auth/login", {"username": USER, "password": PASS})
tok = d.get("access_token") or d.get("token")
ids = [int(x) for x in sys.argv[1:]] or [3547, 3543, 3540, 3535, 3508, 3487]
for rid in ids:
    try:
        st, d = api("GET", f"/tensor/reports/{rid}", token=tok)
    except Exception as e:
        print(rid, "ERR", e); continue
    md = d.get("markdown") or ""
    print("=" * 72)
    print(rid, d.get("created_at", "")[:16], "|", (d.get("title") or "")[:100], "| md_len", len(md))
    if md:
        print(re.sub(r"\n", "\n  ", md[-1800:]))

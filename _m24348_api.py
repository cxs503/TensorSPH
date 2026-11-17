import json, urllib.error
import json, os, sys, urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

def call(method, path, body=None, token=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}

s, d = call("POST", "/auth/login", {"username": USER, "password": PASS})
print("login:", s)
tok = d.get("access_token") or d.get("token") or ""
open("/tmp/m24348_token", "w").write(tok)
print("token_len:", len(tok))

for path in sys.argv[1:]:
    s, d = call("GET", path, token=tok)
    print("==", path, s)
    print(json.dumps(d, ensure_ascii=False)[:4000])

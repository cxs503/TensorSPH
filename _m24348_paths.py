import json, urllib.error
import json, os, urllib.request

BASE = os.environ["TENSOR_API_BASE"]
tok = open("/tmp/m24348_token").read().strip()

def call(method, path, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:
            return e.code, {}

s, d = call("GET", "/openapi.json")
paths = sorted(d.get("paths", {}).keys())
print(len(paths), "paths")
for p in paths:
    print(p)

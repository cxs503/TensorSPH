import json, os, urllib.request, urllib.error

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def try_req(method, path, body=None, token=None):
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()[:300].decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode()

for path, body in [
    ("/tensor/login", {"username": user, "password": pw}),
    ("/auth/login", {"username": user, "password": pw}),
    ("/tensor/login", {"user": user, "pass": pw}),
    ("/tensor/login", {"username": user, "password": pw, "agent": "m3675"}),
]:
    s, t = try_req("POST", path, body)
    print(path, body and list(body.keys()), "->", s, t[:200])

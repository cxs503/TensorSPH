import json, os, urllib.request, urllib.parse

BASE = "http://127.0.0.1:8010"

def req(path, data=None, token=None, method=None):
    body = None
    headers = {"Content-Type": "application/json"}
    if data is not None:
        body = json.dumps(data).encode()
    if token:
        headers["Authorization"] = "Bearer " + token
    r = urllib.request.Request(BASE + path, data=body, headers=headers, method=method or ("POST" if body else "GET"))
    try:
        with urllib.request.urlopen(r, timeout=40) as resp:
            return json.loads(resp.read().decode("utf8", "replace"))
    except Exception as e:
        return {"__err__": str(e)[:300]}

tok = req("/api/v1/auth/login", {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]})
t = tok.get("access_token") or tok.get("token") or ""
print("login:", list(tok)[:5], "token_len", len(t))

if not t:
    print(json.dumps(tok)[:500])
else:
    for ep, q in [("/api/v1/knowledge/search", {"q": "projection SPH Young-Laplace surface tension", "limit": "5"}),
                  ("/api/v1/tensor/negative-knowledge/search", {"q": "TensorSPH laplace projection", "limit": "8"}),
                  ("/api/v1/tensor/knowledge/search", {"q": "projection laplace", "limit": "5"})]:
        print("==", ep)
        print(json.dumps(req(ep + "?" + urllib.parse.urlencode(q), token=t), ensure_ascii=False)[:2000])
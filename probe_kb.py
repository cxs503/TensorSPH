import json, urllib.request, urllib.parse

def get(path):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8010" + path, timeout=30) as r:
            return json.loads(r.read().decode("utf8", "replace"))
    except Exception as e:
        return {"__err__": str(e)[:200]}

d = json.loads(urllib.request.urlopen("http://127.0.0.1:8010/openapi.json").read().decode())
paths = sorted(d.get("paths", {}).keys())
print("SOLUTION-RELATED:", [p for p in paths if "solution" in p.lower() or "inventory" in p.lower()])
print()
q = {"q": "projection SPH Young-Laplace surface tension", "limit": "5"}
for ep in ["/api/v1/knowledge/search", "/api/v1/tensor/negative-knowledge/search", "/api/v1/tensor/knowledge/search"]:
    res = get(ep + "?" + urllib.parse.urlencode(q))
    print("==", ep)
    print(json.dumps(res, ensure_ascii=False)[:1500])
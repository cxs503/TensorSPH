import json, urllib.request

def get(path):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8010" + path, timeout=20) as r:
            return r.status, r.read().decode("utf8", "replace")
    except Exception as e:
        return "ERR", str(e)[:200]

s, body = get("/openapi.json")
print(s, len(body) if s == 200 else body)
if s == 200:
    d = json.loads(body)
    paths = sorted(d.get("paths", {}).keys())
    hits = [p for p in paths if any(w in p.lower() for w in ("knowledge", "oss", "negative", "capab", "anchor"))]
    print("\n".join(hits))
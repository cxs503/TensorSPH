import json, urllib.error
import json, urllib.request

def gh(path):
    req = urllib.request.Request("https://api.github.com" + path)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "tp-agent")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:500]}

d = gh("/repos/cxs503/TensorSPH/pulls/1024")
print("title:", d.get("title"))
print("state:", d.get("state"))
print("merged:", d.get("merged"))
print("head:", d.get("head", {}).get("ref"))
print("base:", d.get("base", {}).get("ref"))
print("created:", d.get("created_at"))
print("closed:", d.get("closed_at"))
print("body:", (d.get("body") or "")[:2000])

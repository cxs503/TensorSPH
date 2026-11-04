import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
tok = open(".scratch/m3675_token").read().strip()

def get(path):
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req) as r:
        return json.load(r)

d = get("/tensor/reports?mission_id=3626")
items = d if isinstance(d, list) else d.get("items", d.get("reports", []))
r = items[0]
print("KEYS:", list(r.keys()))
for k, v in r.items():
    s = str(v)
    if k in ("markdown", "summary"):
        print("===", k, "len", len(s))
        print(s[:8000])
    else:
        print(k, ":", s[:200])

import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
tok = open("/tmp/tok").read().strip()
req = urllib.request.Request(BASE + "/tensor/reports?limit=20",
                            headers={"Authorization": "Bearer " + tok})
d = json.load(urllib.request.urlopen(req, timeout=30))
items = d.get("items") or d.get("reports") or d
for r in items[:20]:
    print(r.get("id"), "|", r.get("mission_id"), "|", (r.get("title") or "")[:90])

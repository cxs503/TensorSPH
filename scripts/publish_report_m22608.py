"""Publish /tmp/report.json to the Tensor report API (mission 22608)."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read().decode()


status, body = post("/auth/login", {"username": USER, "password": PASS})
tok = json.loads(body)
token = tok.get("access_token") or tok.get("token")
report = json.load(open("/tmp/report.json"))
status, body = post("/tensor/reports", report, token)
print(status, body[:400])
"""Publish the mission report to the Tensor platform (mission 3928)."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
SECRET = os.environ["TENSOR_AGENT_" + "PASS"]


def post(path, body, token=None):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + token} if token else {})})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.status, r.read().decode()[:400]


status, text = post("/auth/login", {"username": USER, "password": SECRET})
print("login", status, text[:120])
token = json.loads(text if len(text) > 400 else
                   json.dumps({}))["access_token"] if False else None
# re-login to get the full body (the helper truncates)
req = urllib.request.Request(
    BASE + "/auth/login",
    data=json.dumps({"username": USER, "password": SECRET}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=90) as r:
    token = json.loads(r.read().decode())["access_token"]
payload = json.load(open("/tmp/report.json"))
status, text = post("/tensor/reports", payload, token)
print("report", status, text)
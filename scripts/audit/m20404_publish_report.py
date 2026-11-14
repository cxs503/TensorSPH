#!/usr/bin/env python3
"""Publish the m20404 report to the platform (/tensor/reports)."""
import json, os, urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())

tok = post("/auth/login", {"username": USER, "password": PASS})
token = tok.get("access_token") or tok.get("token")
req = urllib.request.Request(BASE + "/tensor/reports",
                             data=open("/tmp/report.json", "rb").read(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + token})
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.loads(r.read().decode())
print("report id:", out.get("id", out))

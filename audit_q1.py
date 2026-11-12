#!/usr/bin/env python3
"""Audit query 1: benchmark case + mission result + reports for m15433."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def req(method, path, body=None, tok=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


st, login = req("POST", "/auth/login", {"username": USER, "password": PASS})
tok = login.get("access_token") or login.get("token")
print("login:", st, "token_len", len(tok or ""))

# 1) benchmark case rows
for path in ("/tensor/benchmark-cases?anchor_slug=CASE-TAYLOR-GREEN-SPH",
             "/tensor/benchmark_cases?anchor_slug=CASE-TAYLOR-GREEN-SPH"):
    st, d = req("GET", path, tok=tok)
    if st == 200:
        print("benchmark-cases via", path, "->", json.dumps(d, indent=1)[:3000])
        break
    else:
        print("benchmark-cases", path, st, str(d)[:200])

# 2) mission m15433
for path in ("/tensor/missions/15433", "/tensor/missions/m15433"):
    st, d = req("GET", path, tok=tok)
    if st == 200:
        print("mission via", path)
        print(json.dumps(d, indent=1)[:4000])
        break
    else:
        print("mission", path, st, str(d)[:200])

# 3) reports for m15433
for path in ("/tensor/reports?mission_id=15433", "/tensor/reports?mission_id=m15433"):
    st, d = req("GET", path, tok=tok)
    if st == 200:
        print("reports via", path, "count:", len(d) if isinstance(d, list) else "?")
        print(json.dumps(d, indent=1)[:3000])
        break
    else:
        print("reports", path, st, str(d)[:200])

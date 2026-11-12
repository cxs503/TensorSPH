#!/usr/bin/env python3
"""Probe openapi + try Postgres for tensor_benchmark_cases."""
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
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


st, login = req("POST", "/auth/login", {"username": USER, "password": PASS})
tok = json.loads(login).get("access_token") or json.loads(login).get("token")

st, body = req("GET", "/openapi.json", tok=tok)
print("openapi status:", st, "len:", len(body))
if st == 200:
    spec = json.loads(body)
    paths = sorted(spec.get("paths", {}).keys())
    print("total paths:", len(paths))
    for p in paths:
        if any(k in p.lower() for k in ("benchmark", "maturity", "verdict", "gate", "sql", "query")):
            print("  HIT:", p)
    # dump all paths to a file for reference
    with open("/root/tp-agent/work/m15713/api_paths.txt", "w") as f:
        f.write("\n".join(paths))
else:
    print(body[:300])

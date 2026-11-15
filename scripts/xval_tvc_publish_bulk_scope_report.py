#!/usr/bin/env python3
"""Publish the bulk-scope-criterion report to /tensor/reports.

Payload is assembled by ``benchmarks/xval_sphinx_tvc/make_bulk_scope_doc.py``'s
sibling builder from ``data/bulk_scope_criterion.json`` (no hand-copied
numbers); this script only authenticates and posts.
"""
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=180).read().decode())


def token():
    for path in ("/auth/login", "/api/v1/auth/login"):
        try:
            out = post(path, {"username": os.environ["TENSOR_AGENT_USER"],
                              "password": os.environ["TENSOR_AGENT_PASS"]})
        except Exception as exc:                      # pragma: no cover - network
            print(f"{path}: {type(exc).__name__}")
            continue
        got = out.get("access_token") or out.get("token")
        if got:
            return got
    raise SystemExit("no token")


if __name__ == "__main__":
    payload = json.loads(Path("/tmp/report.json").read_text())
    for path in ("/tensor/reports", "/api/v1/tensor/reports"):
        try:
            out = post(path, payload, token())
        except Exception as exc:
            print(f"{path}: {type(exc).__name__} {exc}")
            continue
        print(f"published via {path}: {json.dumps(out)[:400]}")
        break

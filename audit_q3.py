#!/usr/bin/env python3
"""Fetch report 14720 full markdown + check six sections."""
import json
import os
import re
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

st, d = req("GET", "/tensor/reports/14720", tok=tok)
print("status:", st)
if st == 200:
    md = d.get("markdown") or d.get("content") or ""
    print("markdown_len:", len(md))
    with open("/root/tp-agent/work/m15713/report_14720.md", "w") as f:
        f.write(md)
    # section headers
    heads = re.findall(r"^#{1,3} .*", md, flags=re.M)
    print("headers:")
    for h in heads:
        print("  ", h)
    # six-element keywords
    for kw in ("案例", "物理", "方法", "参数", "对照", "收敛", "失败", "工件", "复现", "误差"):
        print(f"kw {kw}:", md.count(kw))
else:
    print(str(d)[:400])

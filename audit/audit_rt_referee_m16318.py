#!/usr/bin/env python3
"""m16318 board-audit probe #2: machine referee (bench/l4-verdict) + gate spec.

Read-only GETs only. Writes audit/m16318_referee.json.
"""
import json
import os
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
OUT = os.path.join(os.path.dirname(__file__), "m16318_referee.json")


def req(method, path, body=None, tok=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return e.code, {"_error": e.read().decode()[:600]}
    except Exception as e:  # noqa: BLE001
        return 0, {"_error": repr(e)}


def main():
    st, login = req("POST", "/auth/login", {"username": USER, "password": PASS})
    tok = (login or {}).get("access_token") or (login or {}).get("token")
    print("login", st, bool(tok))
    ev = {}
    for name, path in [
        ("l4_verdict_16244", "/tensor/bench/l4-verdict/16244"),
        ("gate_spec", "/tensor/campaigns/maturity-gate/spec-schema"),
    ]:
        s, d = req("GET", path, tok=tok)
        print(name, s)
        ev[name] = {"status": s, "data": d}
        print(json.dumps(d, indent=1, ensure_ascii=False)[:2500])
    with open(OUT, "w") as f:
        json.dump(ev, f, indent=1, ensure_ascii=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""m16318 board-audit evidence probe: CASE-SPH-RAYLEIGH-TAYLOR-MP (mission 16244).

Read-only. Dumps benchmark rows, mission result, artifacts, and reports to
audit/m16318_evidence.json for offline verification.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
OUT = os.path.join(os.path.dirname(__file__), "m16318_evidence.json")


def req(method, path, body=None, tok=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return e.code, {"_error": e.read().decode()[:400]}
    except Exception as e:  # noqa: BLE001
        return 0, {"_error": repr(e)}


def main():
    st, login = req("POST", "/auth/login", {"username": USER, "password": PASS})
    tok = (login or {}).get("access_token") or (login or {}).get("token")
    print("login", st, "token", "ok" if tok else "FAIL")
    if not tok:
        print(login)
        return 1
    ev = {"login_status": st}

    for name, path in [
        ("benchmark", "/tensor/benchmark-cases?anchor_slug=CASE-SPH-RAYLEIGH-TAYLOR-MP"),
        ("benchmark_alt", "/tensor/benchmark_cases?anchor_slug=CASE-SPH-RAYLEIGH-TAYLOR-MP"),
        ("mission", "/tensor/missions/16244"),
        ("reports", "/tensor/reports?mission_id=16244"),
    ]:
        s, d = req("GET", path, tok=tok)
        print(name, path, s, len(json.dumps(d)))
        ev[name] = {"status": s, "data": d}

    # mission result / artifacts sub-resources if present
    for name, path in [
        ("result", "/tensor/missions/16244/result"),
        ("artifacts", "/tensor/missions/16244/artifacts"),
        ("events", "/tensor/missions/16244/events"),
    ]:
        s, d = req("GET", path, tok=tok)
        print(name, path, s, len(json.dumps(d)))
        ev[name] = {"status": s, "data": d}

    with open(OUT, "w") as f:
        json.dump(ev, f, indent=1, ensure_ascii=False)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
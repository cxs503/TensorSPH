"""Publish the p457w2-n04 progress report to the platform report API.

The report body is kept in the repository (``docs/p457w2-n04/report_payload.json``
plus the four figures in ``docs/p457w2-n04/``) so that publishing is a
*mechanism* and not a manual step: this script authenticates with
``$TENSOR_AGENT_USER`` / ``$TENSOR_AGENT_PASS``, embeds the figures as base64 and
POSTs to ``$TENSOR_API_BASE/tensor/reports``, retrying with backoff.

Exit codes: 0 = posted, 2 = API unreachable after all attempts (the payload stays
in the repo, nothing is lost), 3 = authentication rejected.

Usage::

    python scripts/post_p457w2n04_report.py [--attempts 5] [--base URL]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "p457w2-n04"
CAPTIONS = {
    "fig1_sound_speed.png": "c_m(alpha): Wood vs van Wijngaarden; alpha*=0.5006, c_min=23.81 m/s",
    "fig2_pressure_family.png": (
        "p_peak(alpha) family v=6..14 m/s and k_b,eq(alpha) "
        "with the inverted legacy constant"
    ),
    "fig3_case_set.png": "per-case G_load, aerated area fraction, G_peak, m9319 band, legacy 35%",
    "fig4_moments.png": "PBM moment closure: M0 decay, M1 conserved, M2/M3 relaxation",
}


def _post(url, payload, token=None, timeout=60):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def build_payload():
    base = json.loads((DOCS / "report_payload.json").read_text())
    figures = []
    for name, caption in CAPTIONS.items():
        path = DOCS / name
        if not path.is_file():
            continue
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )
    base["figures"] = figures
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("TENSOR_API_BASE"))
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--delay", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not args.base:
        print("TENSOR_API_BASE is not set", file=sys.stderr)
        return 2
    payload = build_payload()
    last = None
    for attempt in range(1, args.attempts + 1):
        try:
            creds = {
                "username": os.environ.get("TENSOR_AGENT_USER", ""),
                "password": os.environ.get("TENSOR_AGENT_PASS", ""),
            }
            token = None
            # the platform exposes the login at /auth/login; /tensor/auth/login is
            # kept as a fallback for older deployments
            for suffix in ("/auth/login", "/tensor/auth/login"):
                try:
                    tok = _post(args.base.rstrip("/") + suffix, creds, timeout=20)
                except urllib.error.HTTPError:
                    continue
                token = tok.get("access_token") or tok.get("token")
                if token:
                    break
            if token is None:
                print("auth rejected: no endpoint returned a token", file=sys.stderr)
                return 3
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            print(f"attempt {attempt}: API unreachable ({exc})", file=sys.stderr)
            time.sleep(args.delay * attempt)
            continue
        try:
            result = _post(args.base.rstrip("/") + "/tensor/reports", payload, token=token)
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            print(f"attempt {attempt}: POST failed ({exc})", file=sys.stderr)
            time.sleep(args.delay * attempt)
            continue
        print("posted report id:", result.get("id"))
        return 0
    print(f"gave up after {args.attempts} attempts: {last}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

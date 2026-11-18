#!/usr/bin/env python
"""Push the static capability dump to TensorPlatform (8010) /tensor/capabilities/sync.

Daemon-shaped call: stdout-only JSON from
``python src/tensorsph/registry.py --static`` in, receipt out.  Credentials
from TENSOR_API_BASE / TENSOR_AGENT_USER / TENSOR_AGENT_PASS.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    dump = subprocess.run(
        [sys.executable, str(ROOT / "src/tensorsph/registry.py"), "--static"],
        capture_output=True, text=True, check=True)
    payload = json.loads(dump.stdout)
    if payload.get("error_count"):
        print(f"refusing to sync {payload['error_count']} registry errors", file=sys.stderr)
        return 2
    login = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"})
    token = json.load(urllib.request.urlopen(login))["access_token"]
    body = {"repo": "TensorSPH", "source_commit": payload["source_commit"],
            "capabilities": payload["capabilities"]}
    req = urllib.request.Request(base + "/tensor/capabilities/sync",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json",
                                         "Authorization": "Bearer " + token},
                                method="POST")
    out = ROOT / "docs" / "m25063" / "capability_sync_receipt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        out.write_text(json.dumps({"status": exc.code, "detail": detail,
                                   "source_commit": payload["source_commit"]},
                                  ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1
    receipt["client"] = {"mode": payload["mode"], "count": payload["count"],
                         "summary": payload["summary"]}
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

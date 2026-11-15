"""Fetch the upstream files that define the adaptive smoothing-length ratio.

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.  The transport-velocity
correction port in ``src/tensorsph/transport_velocity_correction.py``
implements upstream's inner/wall loops, but it only ever ran with
``h_ratio == 1`` because the ``h_ratio_(index_i)`` factor of upstream's
``update()`` was not ported (recorded as an open item in
``docs/xval_sphinx_tvc/README.md`` and ``CADENCE_INVARIANCE.md``).

Porting that branch without hand-copying requires three upstream definitions
that were not part of the first fetch:

* ``AdaptiveSmoothingLength::SmoothingLengthRatio`` -- what ``h_ratio_`` is,
* ``Kernel::dW(Real h_ratio, Real r)`` -- how the kernel gradient is scaled by
  it (this decides the normalisation of ``kernel_gradient_integral_``),
* the spline kernel actually used for the gradient (the port uses the cubic
  B-spline, so the same kernel is fetched here, not the Wendland default).

Files are stored verbatim (base64 decode of the contents API response, which is
the only route that works from this host -- ``raw.githubusercontent.com`` is
refused) together with their sha256 and the upstream commit, so every constant
in the port can be traced to a line of source.  Nothing is edited after the
download.

Usage
-----
    python benchmarks/xval_sphinx_tvc/fetch_adaptive_sources.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "docs" / "xval_sphinx_tvc" / "oss_source" / "adaptive"

UPSTREAM = "Xiangyu-Hu/SPHinXsys"
COMMIT = "09e55cfdd23104e1b1db006a209a906218b3b14e"

FILES = [
    "src/shared/adaptations/adaptation.h",
    "src/shared/adaptations/adaptation.cpp",
    "src/shared/adaptations/adaptation.hpp",
    "src/shared/kernels/base_kernel.h",
    "src/shared/kernels/base_kernel.cpp",
    "src/shared/kernels/kernel_cubic_B_spline.h",
    "src/shared/kernels/kernel_cubic_B_spline.cpp",
]


def fetch_one(path: str) -> tuple[bytes, str]:
    url = (f"https://api.github.com/repos/{UPSTREAM}/contents/{path}"
           f"?ref={COMMIT}")
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    if payload.get("encoding") != "base64":
        raise RuntimeError(f"unexpected encoding for {path}: {payload.get('encoding')}")
    return base64.b64decode(payload["content"]), payload["sha"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "repository": f"https://github.com/{UPSTREAM}",
        "commit": COMMIT,
        "fetch_route": "api.github.com contents API (base64); "
                       "raw.githubusercontent.com refused from this host",
        "why": "definitions of h_ratio (AdaptiveSmoothingLength) and of the "
               "h_ratio-scaled kernel gradient needed by the transport-velocity "
               "update() branch",
        "files_sha256": {},
        "git_blob_sha1": {},
    }
    for path in FILES:
        blob, blob_sha = fetch_one(path)
        name = Path(path).name
        (OUT / name).write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()
        record["files_sha256"][name] = digest          # type: ignore[index]
        record["git_blob_sha1"][name] = blob_sha       # type: ignore[index]
        print(f"{name:36s} {len(blob):7d} B  sha256={digest[:16]}...")
    (OUT / "PROVENANCE.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {OUT}/PROVENANCE.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
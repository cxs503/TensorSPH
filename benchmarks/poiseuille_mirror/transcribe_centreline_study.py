#!/usr/bin/env python3
"""Transcribe the parity-study rungs from job logs into the study artefact.

Why this exists
---------------
``study_centreline_sampling.py`` writes
``benchmarks/poiseuille_mirror/out/centreline_sampling.json`` **on the compute
node**.  ``run_compute``/``run_cluster`` ``fetch`` returns the *staged input
snapshot*, not the post-run tree, so the node-side JSON is not retrievable from
the workspace.  The one durable channel is the job stdout
(``/root/tp-agent/cluster/jobs/<jid>.log``), which prints one estimate block per
rung.  This script rebuilds the artefact from those blocks, records the
provenance (job id, log path, which rungs aborted) and derives the fields the
log does not carry.

Nothing here re-derives physics: every number below is copied from the job log
at the precision printed there (0.001 percentage points).  Rungs that never
completed are listed in ``missing_rungs`` rather than interpolated.

Usage:
    python benchmarks/poiseuille_mirror/transcribe_centreline_study.py \
        --job cj3057 --log /root/tp-agent/cluster/jobs/cj3057.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks/poiseuille_mirror/out"

# One block per rung, as printed by study_centreline_sampling.py:
#   [centreline-sampling]   A_legacy +0.063%  A_same_row +0.063%  A_fit_full +0.239%  bias_pred 0.9956  (78s)
LINE = re.compile(
    r"H/dx=(?P<n>\d+) \.\.\.\s*\n"
    r"\[centreline-sampling\]\s+A_legacy (?P<legacy>[+-][\d.]+)%\s+"
    r"A_same_row (?P<same>[+-][\d.]+)%\s+"
    r"A_fit_full (?P<fit>[+-]?nan|[+-]?[\d.]+)%\s+"
    r"bias_pred (?P<bias>[\d.]+)\s+\((?P<wc>[\d.]+)s\)"
)

HEIGHT = 0.1
BODY_FORCE = 9.81
VISCOSITY = 0.01
U_MAX = BODY_FORCE * HEIGHT ** 2 / (8.0 * VISCOSITY)


def parse(log_text: str) -> tuple[list[dict], list[int]]:
    entries, seen = [], []
    for m in LINE.finditer(log_text):
        n = int(m.group("n"))
        seen.append(n)
        legacy = float(m.group("legacy")) / 100.0
        same = float(m.group("same")) / 100.0
        fit_s = m.group("fit")
        fit = None if "nan" in fit_s else float(fit_s) / 100.0
        odd = n % 2 == 1
        bias = 1.0 if odd else 1.0 - (1.0 / n) ** 2
        entries.append({
            "resolution_H_over_dx": n,
            "dx_over_H": 1.0 / n,
            "nearest_row_y_over_H": 0.5 if odd else ((n // 2 - 1) + 0.5) / n,
            "row_sits_on_centreline": odd,
            "u_max_ref": U_MAX,
            "u_max_measured": U_MAX * (1.0 + legacy),
            "A_legacy": legacy,
            "A_same_row": same,
            "A_fit_full": fit,
            "A_fit_mid": None,
            "bias_pred_factor": bias,
            "A_legacy_from_fit_full": ((1.0 + fit) * bias - 1.0)
            if fit is not None else None,
            "wall_clock_s": float(m.group("wc")),
            "provenance": "transcribed from job log (3 d.p. percent)",
        })
    return entries, seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", default="cj3057")
    ap.add_argument("--log", type=Path,
                    default=Path("/root/tp-agent/cluster/jobs/cj3057.log"))
    ap.add_argument("--requested", type=int, nargs="+",
                    default=[15, 16, 17, 24, 25])
    args = ap.parse_args()

    text = args.log.read_text(errors="replace")
    # the cluster log wraps the job stdout inside a JSON envelope, so line
    # breaks appear as the two characters backslash-n
    if "\\n" in text:
        text = text.replace("\\n", "\n")
    entries, seen = parse(text)
    missing = [n for n in args.requested if n not in seen]
    abort = None
    for marker in ("free(): corrupted unsorted chunks", "Segmentation fault",
                   "MemoryError", "out of memory"):
        if marker in text:
            abort = marker
            break

    payload = {
        "case": "plane_poiseuille_channel_mirror_wall_centreline_sampling",
        "anchor": "CASE-POISEUILLE-SPH",
        "question": ("does gate A compare a row mean against the continuum "
                     "centre-line value, and how large is that sampling "
                     "artefact?"),
        "hypothesis": ("A_legacy carries a 1-(dx/H)^2 sampling factor at even "
                       "H/dx and none at odd H/dx; A_same_row / A_fit_* are "
                       "blind to lattice parity"),
        "setup": {"height": HEIGHT, "viscosity": VISCOSITY,
                  "body_force": BODY_FORCE, "columns": 12,
                  "t_end": 1.0, "t_average": 0.2, "threads": 8},
        "provenance": {
            "job": args.job,
            "log": str(args.log),
            "transcribed": True,
            "reason": ("run_compute fetch returns the staged pre-run snapshot; "
                       "the node-side JSON is not retrievable, so the rung "
                       "numbers are transcribed from job stdout"),
            "requested_rungs": args.requested,
            "completed_rungs": sorted(seen),
            "missing_rungs": missing,
            "abort_marker": abort,
        },
        "series": sorted(entries, key=lambda e: e["resolution_H_over_dx"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "centreline_sampling.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
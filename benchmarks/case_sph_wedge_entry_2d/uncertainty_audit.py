"""Is the frame bootstrap of the footprint reading honest? (m22416)

The question
------------
``footprint_ledger.py`` decides whether a grid is *judgeable* from ``U``, the
half-width of the 5..95 % frame-bootstrap interval of the footprint reading
(``ensemble_peak.py``, ``BOOT_RESAMPLES = 400``).  That bootstrap draws the 61
observation frames **i.i.d. with replacement**.  But the frames are not i.i.d.:
they are consecutive snapshots of one run, ``t_end * k / 60`` apart, and the
reading is driven by the impact transient, i.e. by a *contiguous* block of
frames around touchdown.  Resampling frames independently therefore destroys
the temporal clustering that carries the signal and reports an interval that is
too narrow — the classical moving-block problem (Kunsch 1989; Politis & Romano
1994).

Since ``U`` enters the verdict through a hard gate (``U <= 15 %``) and the best
committed grid sits at ``U = 14.8 %`` (dx = 2 mm, 4-cell footprint), a bootstrap
that understates ``U`` is not a cosmetic issue: it is the difference between
``NOT_DECIDABLE`` and a claim that a grid is measurable.

What this module does
---------------------
* keeps the *estimator* identical (same pooling, same sliding-window footprint
  operator, same 400 resamples, same seed);
* adds a moving-block resampler next to the i.i.d. one, so the two intervals
  answer the same question under two different dependence assumptions;
* reports the lag-1 autocorrelation and effective sample size of the per-frame
  footprint series as measured diagnostics, so the reader can see *why* the two
  differ;
* reports the block-length sensitivity (L = 1, 3, 5, 9, 15 frames) instead of
  hiding a hand-picked L.

The machinery is validated against an analytic ground truth on AR(1) data in
``tests/test_wedge_uncertainty_audit.py``: for lag-1 correlation ``rho`` the
i.i.d. bootstrap must understate the standard deviation of a sample mean by
``sqrt((1-rho)/(1+rho))``, and the moving-block bootstrap must recover it.

Usage
-----
Inside the job (the frame dump is far larger than the 8 kB log tail, so the
audit runs where the data is and prints a compact summary)::

    python benchmarks/case_sph_wedge_entry_2d/uncertainty_audit.py \
        --dump docs/artifacts/.../frame_samples_dx2mm.json --cells 4 --json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import footprint_peak as fpm  # noqa: E402

BOOT_RESAMPLES = 400          # identical to ensemble_peak.BOOT_RESAMPLES
BOOT_SEED = 20261113          # identical to ensemble_peak.BOOT_SEED
BLOCK_LENGTHS = (1, 3, 5, 9, 15)


# --------------------------------------------------------------------------- #
# resamplers (the only thing this module changes about the estimator)
# --------------------------------------------------------------------------- #
def iid_indices(n: int, rng: random.Random) -> list[int]:
    """The committed scheme: independent draws with replacement."""
    return [rng.randrange(n) for _ in range(n)]


def moving_block_indices(n: int, rng: random.Random, block: int) -> list[int]:
    """Circular moving block of length ``block``, concatenated to ``n`` draws.

    ``block == 1`` reproduces :func:`iid_indices` in distribution, which is what
    makes the block-length sweep a controlled comparison rather than a change of
    estimator.
    """
    if block <= 1:
        return iid_indices(n, rng)
    idx: list[int] = []
    while len(idx) < n:
        start = rng.randrange(n)
        idx.extend((start + k) % n for k in range(block))
    return idx[:n]


def bootstrap(values, stat, resamples: int = BOOT_RESAMPLES,
              seed: int = BOOT_SEED, mode: str = "iid",
              block: int = 1) -> list[float]:
    """``stat`` evaluated on ``resamples`` resamples of ``values``."""
    rng = random.Random(seed)
    out = []
    for _ in range(resamples):
        idx = (iid_indices(len(values), rng) if mode == "iid"
               else moving_block_indices(len(values), rng, block))
        v = stat([values[i] for i in idx])
        if v == v:
            out.append(v)
    return out


def half_width(samples: list[float]) -> dict:
    """5..95 % half-width relative to the median -- the ledger's ``U``."""
    vals = sorted(samples)
    if not vals:
        return {"n_valid": 0, "rel_half_width": float("nan")}

    def q(f):
        return vals[min(len(vals) - 1, int(round(f * (len(vals) - 1))))]

    med = q(0.5)
    return {"n_valid": len(vals), "p05": q(0.05), "p50": med, "p95": q(0.95),
            "rel_half_width": (q(0.95) - q(0.05)) / 2.0 / med if med else float("nan")}


# --------------------------------------------------------------------------- #
# the case-specific statistic (unchanged estimator)
# --------------------------------------------------------------------------- #
def footprint_stat(shots, cells: float, dx: float, c_end: float) -> float:
    """Footprint reading of a set of frames -- ``ensemble_peak``'s estimator."""
    taus: list[float] = []
    cps: list[float] = []
    for s in shots:
        taus.extend(s["tau"])
        cps.extend(s["cp"])
    rows = fpm.footprint_peaks(taus, cps, dx, c_end, (float(cells),))
    return rows[0]["cp_peak"]


def lag1(series: list[float]) -> float:
    """Lag-1 autocorrelation of a series (NaN-safe, needs >= 3 points)."""
    vals = [v for v in series if v == v]
    n = len(vals)
    if n < 3:
        return float("nan")
    mean = sum(vals) / n
    den = sum((v - mean) ** 2 for v in vals)
    if den <= 0.0:
        return float("nan")
    num = sum((vals[i] - mean) * (vals[i + 1] - mean) for i in range(n - 1))
    return num / den


def n_effective(n: int, rho: float) -> float:
    """``n (1-rho)/(1+rho)`` -- the AR(1) effective sample size."""
    if not (rho == rho) or rho >= 1.0:
        return float("nan")
    return n * (1.0 - rho) / (1.0 + rho)


def audit(dump: dict, cells: float, resamples: int = BOOT_RESAMPLES,
          seed: int = BOOT_SEED, blocks=BLOCK_LENGTHS) -> dict:
    """i.i.d. vs moving-block interval of the same reading, plus diagnostics."""
    shots = dump["frames"]
    dx = dump["dx"]
    t_end = dump["t_end"]
    C = dump["C"]
    # V is 1 m/s in this case family; read it from the dump when present so the
    # module stays correct if the ladder is ever rerun at another speed.
    vel = dump.get("velocity", 1.0)
    c_end = C * vel * t_end

    stat = lambda sel: footprint_stat(sel, cells, dx, c_end)  # noqa: E731
    obs = stat(shots)
    iid = half_width(bootstrap(shots, stat, resamples, seed, "iid", 1))
    per_block = {}
    for L in blocks:
        per_block[str(L)] = half_width(
            bootstrap(shots, stat, resamples, seed, "block", L))

    # per-frame readings: the series whose dependence the bootstrap ignores
    series = [footprint_stat([s], cells, dx, c_end) for s in shots]
    rho = lag1(series)
    n = len(series)
    ratio = [("analytic_ar1", math.sqrt((1.0 - rho) / (1.0 + rho))
              if rho == rho and -1.0 < rho < 1.0 else float("nan"))]
    return {
        "case": dump.get("case"),
        "tag": dump.get("tag"),
        "dx_m": dx, "footprint_cells": cells, "phi_m": cells * dx,
        "n_frames": n, "resamples": resamples, "seed": seed,
        "observed_cp": obs,
        "iid": iid,
        "block": per_block,
        "per_frame_series": series,
        "lag1_autocorr": rho,
        "n_effective": n_effective(n, rho),
        "inflation_analytic": dict(ratio),
        "inflation_measured": {
            L: (per_block[str(L)]["rel_half_width"] / iid["rel_half_width"]
                if iid["rel_half_width"] else float("nan"))
            for L in blocks},
    }


def compact(rep: dict) -> dict:
    """Fields that fit the 8 kB job-log tail."""
    keep = ("tag", "dx_m", "footprint_cells", "phi_m", "n_frames", "resamples",
            "seed", "observed_cp", "iid", "lag1_autocorr", "n_effective",
            "inflation_analytic", "inflation_measured")
    out = {k: rep[k] for k in keep if k in rep}
    out["block"] = {L: {"rel_half_width": r["rel_half_width"],
                        "p50": r.get("p50")}
                    for L, r in rep["block"].items()}
    out["per_frame_min_max"] = [min(rep["per_frame_series"]),
                                max(rep["per_frame_series"])]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True,
                    help="frame_samples_<tag>.json written by --dump-frames")
    ap.add_argument("--cells", type=float, default=None,
                    help="footprint width in cells (default: PHI/dx for 8 mm)")
    ap.add_argument("--phi", type=float, default=8.0e-3)
    ap.add_argument("--resamples", type=int, default=BOOT_RESAMPLES)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    dump = json.loads(Path(args.dump).read_text())
    cells = args.cells if args.cells is not None else args.phi / dump["dx"]
    rep = audit(dump, cells, args.resamples, args.seed)
    if args.json:
        print("AUDIT_BEGIN " + json.dumps(compact(rep)) + " AUDIT_END")
        return 0
    print(f"[audit] {rep['tag']}  cells={rep['footprint_cells']:.1f}"
          f"  observed={rep['observed_cp']:.3f}")
    print(f"  i.i.d.   U = {rep['iid']['rel_half_width']:+.1%}")
    for L, r in rep["block"].items():
        print(f"  block L={L:>2} U = {r['rel_half_width']:+.1%}"
              f"  (x{rep['inflation_measured'][L]:.2f} vs i.i.d.)")
    print(f"  lag-1 autocorr = {rep['lag1_autocorr']:+.3f}"
          f"  n_eff = {rep['n_effective']:.1f} / {rep['n_frames']}"
          f"  analytic inflation = x{rep['inflation_analytic']['analytic_ar1']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
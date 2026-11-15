"""Diagnostic: what does the projection pressure readout actually read?

Round 5 of ``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`` left two gates red
(``station_rms_pct``, ``pressure_bias_pct_all_aspects_R_over_dx_40`` = +15.94 %
at a:b = 3:1) and recorded the *readout* as the suspect instrument.  This script
does not fix anything: it prints the field the readout samples, so that the
blame can be assigned from numbers instead of from the plausibility of a story.

For every station of the 3:1 ellipse at ``R_eq/dx = 40`` it prints

* ``p`` binned in signed interface distance ``s/dx`` (positive = vapour side),
  which shows whether the two plateaus the jump is read from exist at all;
* the shipped readout ``<p>_in - <p>_out`` (probes 2-4 dx inside, 3-6 dx out),
  the same readout with the probes *moved* (to separate a plateau-bias from a
  probe-placement artefact) and a linear-fit extrapolation to ``s = 0``
  per side (which is only meaningful if the plateaus are flat);
* the flatness of each plateau, ``std(p)`` over the probe shell, i.e. whether
  averaging over a shell of a given curvature radius is a valid way to sample a
  plateau.

Run: ``python3 scripts/xval_projection_laplace/diag_pressure_profile.py``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import curvature_variation as cv  # noqa: E402

BINS = (-8, -6, -5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0,
        0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8)


def profile(p, s, dx):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (s / dx >= lo) & (s / dx < hi)
        if int(m.sum()) > 0:
            out.append({"s_lo_dx": lo, "s_hi_dx": hi, "n": int(m.sum()),
                        "p_mean": float(p[m].mean()),
                        "p_std": float(p[m].std()) if int(m.sum()) > 1 else 0.0})
    return out


def readout(p, s, dx, inner, outer):
    mi = (s / dx < -inner[0]) & (s / dx > -inner[1])
    mo = (s / dx > outer[0]) & (s / dx < outer[1])
    if not bool(mi.any()) or not bool(mo.any()):
        return {"dp": float("nan"), "n_in": int(mi.sum()), "n_out": int(mo.sum())}
    # linear fit p(s) per side, evaluated at s = 0 (extrapolation to the interface)
    def fit(mask):
        x = (s[mask] / dx)
        y = p[mask]
        if int(mask.sum()) < 3:
            return float(y.mean()), float("nan")
        xm, ym = x.mean(), y.mean()
        den = float(((x - xm) ** 2).sum())
        slope = float(((x - xm) * (y - ym)).sum()) / den if den > 0 else 0.0
        return float(ym - slope * xm), slope

    p_in, sl_in = fit(mi)
    p_out, sl_out = fit(mo)
    return {"dp": float(p[mi].mean()) - float(p[mo].mean()),
            "dp_fit": p_in - p_out,
            "slope_in_Pa_per_dx": sl_in, "slope_out_Pa_per_dx": sl_out,
            "n_in": int(mi.sum()), "n_out": int(mo.sum()),
            "std_in": float(p[mi].std()), "std_out": float(p[mo].std())}


def main():
    dx = cv.R_EQ / 40.0
    rec = {"aspect": 3.0, "r_eq_over_dx": 40.0, "sigma": cv.SIGMA,
           "probes_shipped": {"inner_dx": (2.0, 4.0), "outer_dx": (3.0, 6.0)},
           "stations": []}
    c = cv.ellipse_case(aspect=3.0, dx=dx, return_fields=True)
    F = c["fields"]
    p, s = F["p"], F["s_signed"]
    rec["p_min_Pa"], rec["p_max_Pa"] = float(p.min()), float(p.max())
    rec["flat_side_profile"] = profile(p, s, dx)
    for kk, r in enumerate(c["stations"]):
        d = {"station": kk, "t_deg": r["t_deg"],
             "dp_theory_Pa": r["dp_theory_Pa"],
             "kappa_exact_center": r["kappa_exact_center_per_m"],
             "dp_shipped": r["dp_pressure_Pa"],
             "err_shipped_pct": r["dp_pressure_err_pct"]}
        d["moved_probes"] = readout(p, s, dx, (1.0, 2.5), (2.0, 4.0))
        d["wide_probes"] = readout(p, s, dx, (1.5, 5.0), (2.0, 6.0))
        rec["stations"].append(d)
    # global statistics of the three readouts
    th = [d["dp_theory_Pa"] for d in rec["stations"]]
    def errs(vals):
        e = [100.0 * (x / t - 1.0) for x, t in zip(vals, th) if math.isfinite(x)]
        if not e:
            return {"n": 0}
        m = sum(e) / len(e)
        return {"n": len(e), "err_mean_pct": m,
                "err_rms_pct": math.sqrt(sum(x * x for x in e) / len(e))}
    rec["summary"] = {
        "shipped": errs([d["dp_shipped"] for d in rec["stations"]]),
        "moved": {**errs([d["moved_probes"]["dp"] for d in rec["stations"]]),
                  **{"fit_" + k: v for k, v in
                     errs([d["moved_probes"]["dp_fit"] for d in rec["stations"]]).items()}},
        "wide": {**errs([d["wide_probes"]["dp"] for d in rec["stations"]]),
                 **{"fit_" + k: v for k, v in
                    errs([d["wide_probes"]["dp_fit"] for d in rec["stations"]]).items()}},
    }
    out = ROOT / "scripts" / "xval_projection_laplace" / "results" / "diag_pressure_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["summary"], indent=1))
    print("flat-side p(s) profile (vapour side positive):")
    for b in rec["flat_side_profile"]:
        print("  s/dx [%5.1f,%5.1f) n=%4d p=%9.4f std=%7.4f" %
              (b["s_lo_dx"], b["s_hi_dx"], b["n"], b["p_mean"], b["p_std"]))
    print("written", out)


if __name__ == "__main__":
    main()
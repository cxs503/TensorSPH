"""Finite-area-averaged peak pressure: the quantity a transducer measures.

Why this module exists (m21677)
-------------------------------
The case grades ``Cp_max`` as the maximum of a *binned* profile: pool the
self-similar frames, average ``p/q`` in bins of ``tau``, report the largest bin
mean (``ensemble_peak.gated_peak``).  Two defects follow from that definition
and both are visible in the committed artefacts:

1. **Aliasing.**  The bin edges sit on a fixed grid, so an isolated noise spike
   is either split across two bins (and halved) or swallowed whole.  The
   measured peak therefore oscillates non-monotonically with the bin count even
   though the analytic field through the same operator is monotone
   (``bandwidth_matched.operator_bias_curve``): at ``dx = 4 mm`` the observed
   reading runs 8.00, 10.36, 8.81, 10.36, 12.01, 12.87 over 2.5 ... 0.68 cells
   per bin.  The reported "estimator spread" is thus a variance estimate of a
   statistic whose signal-to-noise ratio is of order one.
2. **Undefined footprint.**  A point maximum of a square-root-singular field is
   not a measurable quantity: every slamming experiment (and the digitised
   table this case is graded against) reports a *finite-area* average, over the
   face of a pressure transducer.  Comparing a point maximum with such data
   makes the comparison definition-of-the-metric dependent.

This module defines the measured quantity the way the experiment defines it: a
maximum over **sliding windows** of a fixed physical width ``Phi`` (the sensor
footprint),


    Cp_Phi = max_k  mean{ p/q : |x| in [x_k, x_k + Phi] },   Phi = f * dx


The window slides over a fine ``tau`` grid (no fixed edge grid -> no aliasing),
each window averages every sample it contains (no sample is split or dropped),
and the same operator is applied to the Wagner field, so the reference side is
the same functional of the same idealisation.

The footprint is declared in *cells* so that it is resolvable on every grid
(``Phi >= 2 dx``); the physical size is then ``f * dx``, which is what a
transducer would be replaced by when comparing with a specific experiment.

Usage
-----
::

    from footprint_peak import footprint_peaks
    rows = footprint_peaks(taus, cps, dx=1e-3, c_end=0.05,
                           footprints_cells=(2, 4, 8, 16))
"""

from __future__ import annotations

import math

TAU_MAX = 1.2           # the estimator window used throughout the case
TAU_HI = 1.0            # pressure is read over the wetted footprint only
N_FINE = 240            # fine tau grid the windows slide over
MIN_SAMPLES = 4         # a window needs this many pooled samples to be read


def _fine_profile(taus, cps, n_fine: int = N_FINE):
    """``(centres, means, counts)`` of the pooled profile on a fine tau grid."""
    width = TAU_MAX / n_fine
    sums = [0.0] * n_fine
    cnts = [0] * n_fine
    for tau, cp in zip(taus, cps):
        if hasattr(tau, "size") or hasattr(cp, "size"):
            if getattr(tau, "size", 1) == 0 or getattr(cp, "size", 1) == 0:
                continue
            tau, cp = float(tau), float(cp)
        if not (tau == tau) or not (cp == cp) or tau < 0.0 or tau > TAU_MAX:
            continue
        k = min(int(tau / width), n_fine - 1)
        sums[k] += cp
        cnts[k] += 1
    centres = [(k + 0.5) * width for k in range(n_fine)]
    means = [sums[k] / cnts[k] if cnts[k] else float("nan") for k in range(n_fine)]
    return centres, means, cnts


def footprint_peaks(taus, cps, dx: float, c_end: float,
                    footprints_cells=(2.0, 4.0, 8.0, 16.0),
                    n_fine: int = N_FINE, min_samples: int = MIN_SAMPLES,
                    tau_hi: float = TAU_HI) -> list[dict]:
    """Maximum finite-area-averaged ``cp`` for each footprint width.

    ``taus`` is ``|x|/c``, ``cps`` is ``p/q``; ``c_end`` is the wetted half-width
    at the end of the window, so ``Phi / c_end`` is the window width in ``tau``.
    """
    centres, means, cnts = _fine_profile(taus, cps, n_fine)
    width = TAU_MAX / n_fine
    out = []
    for f in footprints_cells:
        phi = f * dx
        d_tau = phi / c_end if c_end else float("nan")
        w = max(1, int(round(d_tau / width))) if d_tau == d_tau else 1
        best = (float("nan"), float("nan"), 0)
        for k in range(0, n_fine - w + 1):
            centre = (k + 0.5 * w) * width
            if centre > tau_hi:
                break
            s = 0.0
            n = 0
            for j in range(k, k + w):
                if means[j] == means[j]:
                    s += means[j] * cnts[j]
                    n += cnts[j]
            if n < min_samples:
                continue
            val = s / n
            if not (best[0] == best[0]) or val > best[0]:
                best = (val, centre, n)
        out.append({"footprint_cells": float(f), "footprint_m": phi,
                    "delta_tau": d_tau, "window_fine_bins": w,
                    "cp_peak": best[0], "tau_at_peak": best[1],
                    "samples_in_window": best[2]})
    return out


def analytic_footprint_peak(channel, beta_rad: float, dx: float, c_end: float,
                            footprints_cells=(2.0, 4.0, 8.0, 16.0),
                            **kw) -> list[dict]:
    """The same operator applied to a sampled analytic field.

    ``channel`` maps ``tau`` (array) -> ``p/q`` (array) for the region where the
    analytic profile is valid; used to put the reference through the identical
    functional.  Kept separate from :func:`footprint_peaks` so the two calls are
    explicit at the call site.
    """
    taus: list[float] = []
    cps: list[float] = []
    for tau, cp in channel:
        taus.append(tau)
        cps.append(cp)
    return footprint_peaks(taus, cps, dx=dx, c_end=c_end,
                           footprints_cells=footprints_cells, **kw)


def footprint_uncertainty(rows: list[dict], best_cells: float,
                          bootstrap_rows: list[list[dict]]) -> dict:
    """Spread of the footprint reading over a bootstrap ensemble.

    ``bootstrap_rows`` is one :func:`footprint_peaks` result per resample; the
    half-width of the 5..95 % interval is returned for the chosen footprint.
    """
    vals = []
    for row in bootstrap_rows:
        for r in row:
            if r["footprint_cells"] == best_cells and r["cp_peak"] == r["cp_peak"]:
                vals.append(r["cp_peak"])
    vals.sort()
    if not vals:
        return {"n_valid": 0}
    def q(f):
        return vals[min(len(vals) - 1, int(round(f * (len(vals) - 1))))]
    med = q(0.5)
    return {"n_valid": len(vals), "p05": q(0.05), "p50": med, "p95": q(0.95),
            "rel_half_width": (q(0.95) - q(0.05)) / 2.0 / med if med else float("nan")}
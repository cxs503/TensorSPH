"""Discretization-uncertainty estimation and validation (GCI + ASME V&V 20).

Why this module exists
----------------------
A resolution ladder that only reports "the error got smaller" is not a
verification statement.  Two separate questions have to be answered, and they
are routinely conflated:

* **Verification** -- how much of the residual is *discretization error*, and is
  that estimate credible?  Requires either an observed order of accuracy from
  the ladder (Roache 1994 GCI, Celik et al. 2008) or, when the exact solution is
  known, the decay rate of the measured error itself.
* **Validation** -- is the agreement with the reference *within* the numerical
  uncertainty, i.e. does the code solve the right equations to the accuracy
  claimed?  ASME V&V 20-2009 compares the comparison error ``E = |S - T|`` with
  a validation uncertainty ``U_V = sqrt(U_num^2 + U_input^2)``.

The subtle part, and the reason this is a module rather than three lines in a
benchmark script, is that the ladder *frequently refuses to cooperate*:

* the error can change sign between rungs (oscillatory convergence, e.g. a
  higher-order term swapping sign) -- then no order exists and Richardson
  extrapolation is meaningless;
* the magnitude can stall or grow (a noise floor, or a non-smooth error
  component such as a boundary condition acting one spacing inside the wall);
* the refinement ratios can be unequal (8/12/16 gives ``r = 1.5`` then
  ``1.3333``), which breaks the classical equal-ratio triple-grid formulas.

So every function here returns a *status*, not just a number, and the caller
must report the status.  ``"converging"`` with pair orders that disagree by more
than ``ORDER_AGREEMENT`` becomes ``"order_unresolved"``: the extrapolated value
may still be usable, but it may not be quoted as an estimate of the exact
solution.  Non-converging ladders are reported as ``"oscillatory"`` /
``"diverging"`` / ``"stalled"`` and *no* order is produced; inventing one there
would manufacture confidence the data does not support.

Scale discipline (this matters and is easy to get wrong)
-------------------------------------------------------
Percent-of-what?  A relative error and a phase error do not share a scale, and
comparing either against a declared acceptance tolerance without pinning the
scale silently changes the acceptance test.  Every entry point therefore takes
an explicit ``scale``: the denominator the uncertainty is expressed against
(default ``|value|``, i.e. relative).  A phase functional passes
``scale = pi/2`` so that "5 %" means 5 % of a quarter turn, which is how the
case declares its phase tolerance.

Everything is pure ``math`` (no numpy/torch), so the analysis runs on committed
artifacts without pulling the solver in.

References
----------
* P. J. Roache, "Perspective: a method for uniform reporting of grid refinement
  studies", ASME J. Fluids Eng. 116 (1994) 405-413.  (GCI, factor of safety)
* I. B. Celik et al., "Procedure for estimation and reporting of uncertainty due
  to discretization in CFD applications", ASME J. Fluids Eng. 130 (2008) 078001.
* ASME V&V 20-2009, "Standard for verification and validation in computational
  fluid dynamics and heat transfer".
* L. Eca & M. Hoekstra, "A procedure for the estimation of the numerical
  uncertainty of CFD calculations based on grid refinement studies",
  J. Comput. Phys. 262 (2014) 104-130.  (unequal ratios; why three grids can
  leave the order unidentified)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "Rung",
    "DEFAULT_SAFETY",
    "ORDER_AGREEMENT",
    "CONSERVATIVE_ORDER",
    "align_truth",
    "pair_order",
    "order_from_errors",
    "richardson_extrapolate",
    "gci_percent",
    "repeatability",
    "validation",
    "ladder_analysis",
]

#: Roache's factor of safety for a single-grid GCI; Celik et al. (2008) use the
#: same 1.25 in the three-grid recommended practice.
DEFAULT_SAFETY = 1.25

#: Two pair-orders closer than this (in order units) are treated as agreement.
ORDER_AGREEMENT = 0.5

#: Order used when the observed order cannot be identified: the lowest order a
#: consistent discretization is expected to be at least (first order).  It makes
#: the GCI conservative -- a smaller order in ``r**p - 1`` gives a wider band.
CONSERVATIVE_ORDER = 1.0


@dataclass(frozen=True)
class Rung:
    """One level of a refinement ladder.

    ``spacing`` is the refinement parameter (mesh spacing, time step, particle
    spacing -- whatever goes to zero); ``value`` is the scalar solution
    functional being graded.  ``label`` is reporting only.
    """

    spacing: float
    value: float
    label: str = ""

    def __post_init__(self):
        if not (self.spacing > 0.0):
            raise ValueError(f"spacing must be > 0, got {self.spacing!r}")
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value!r}")

    def __repr__(self) -> str:  # pragma: no cover - reporting aid
        tag = f" {self.label}" if self.label else ""
        return f"Rung({self.spacing:g}, {self.value:.6g}{tag})"


def _ratio(fine: Rung, coarse: Rung) -> float:
    """Refinement ratio ``r = h_coarse / h_fine`` (> 1 for a refinement)."""
    r = coarse.spacing / fine.spacing
    if not (r > 1.0):
        raise ValueError(
            f"rungs are not ordered fine->coarse: {fine.spacing!r} -> {coarse.spacing!r}"
        )
    return r


def align_truth(
    rungs: Sequence[Rung], truth: float | Sequence[float]
) -> tuple[list[Rung], list[float]]:
    """Sort the ladder fine -> coarse and align a (possibly per-rung) truth.

    Per-rung truth is not a corner case: when the graded functional is a
    solution value *sampled on the lattice rows*, the analytic value itself moves
    with the resolution, because the row centres move.  Comparing every rung
    against one reference value then invents an order the data does not contain.
    This helper carries rung and reference together through the sort so that
    cannot happen silently.
    """
    rs = list(rungs)
    if isinstance(truth, (int, float)):
        ts = [float(truth)] * len(rs)
    else:
        ts = [float(t) for t in truth]
        if len(ts) != len(rs):
            raise ValueError(f"per-rung truth has {len(ts)} entries for {len(rs)} rungs")
    pairs = sorted(zip(rs, ts), key=lambda rt: rt[0].spacing)
    return [rt[0] for rt in pairs], [rt[1] for rt in pairs]


def pair_order(e_fine: float, e_coarse: float, r: float) -> float | None:
    """Observed order from one fine/coarse pair with *known* errors.

    ``p = ln(|e_coarse| / |e_fine|) / ln(r)``.  ``None`` when it does not exist:
    errors of opposite sign (the error changed sign between the rungs -- no
    order), ``e_fine == 0`` (division by zero), or a non-refinement ``r``.
    """
    if not (r > 1.0):
        raise ValueError(f"refinement ratio must be > 1, got {r!r}")
    if not math.isfinite(e_fine) or not math.isfinite(e_coarse):
        return None
    if e_fine == 0.0 or e_coarse == 0.0 or e_fine * e_coarse < 0.0:
        return None
    return math.log(abs(e_coarse) / abs(e_fine)) / math.log(r)


def order_from_errors(rungs: Sequence[Rung], truth: float | Sequence[float]) -> dict:
    """Order identification from measured errors against a known truth.

    ``status`` is one of ``"converging"``, ``"order_unresolved"``,
    ``"oscillatory"``, ``"stalled"``, ``"diverging"``, ``"insufficient"``.

    ``"converging"`` requires: one error sign, monotonically decreasing
    magnitude, and pair orders agreeing within :data:`ORDER_AGREEMENT`.  With
    unequal refinement ratios (8/12/16 -> r = 1.5, 1.333) the two pair orders are
    *expected* to differ somewhat even for a clean power-law error, because each
    pair samples a different interval of the error curve; the window is wide
    enough to accept that and narrow enough to reject "order 1 then order 4".
    """
    rs, ts = align_truth(rungs, truth)
    if len(rs) < 2:
        return {"status": "insufficient", "reason": "need at least two rungs",
                "errors": [], "pair_orders": []}

    errors = [r.value - t for r, t in zip(rs, ts)]
    signs = {0 if e == 0.0 else (1 if e > 0 else -1) for e in errors}

    pair_orders: list[dict] = []
    for i in range(len(rs) - 1):
        r = _ratio(rs[i], rs[i + 1])
        pair_orders.append({
            "fine": rs[i].label or f"h={rs[i].spacing:g}",
            "coarse": rs[i + 1].label or f"h={rs[i + 1].spacing:g}",
            "r": r,
            "e_fine": errors[i],
            "e_coarse": errors[i + 1],
            "p": pair_order(errors[i], errors[i + 1], r),
        })

    mags = [abs(e) for e in errors]
    decreasing = all(mags[i] < mags[i + 1] for i in range(len(mags) - 1))

    if 0.0 in signs:
        status = "converging" if len(signs) == 1 else "oscillatory"
        reason = "an exact zero error is present (ladder hit the truth to round-off)"
    elif len(signs) > 1:
        status, reason = "oscillatory", "the error changes sign between rungs"
    elif not decreasing:
        grew = all(mags[i] > mags[i + 1] for i in range(len(mags) - 1))
        status = "diverging" if grew else "stalled"
        reason = ("|error| grows with refinement" if grew
                  else "|error| is not monotonically decreasing")
    else:
        ps = [p["p"] for p in pair_orders if p["p"] is not None]
        spread = (max(ps) - min(ps)) if ps else math.inf
        if ps and spread <= ORDER_AGREEMENT:
            status = "converging"
            reason = f"pair orders agree within {ORDER_AGREEMENT:g} (spread {spread:.2f})"
        else:
            status = "order_unresolved"
            reason = (f"pair orders disagree by {spread:.2f} order units "
                      f"({min(ps):.2f}..{max(ps):.2f})" if ps else "no pair order")

    ps = [p["p"] for p in pair_orders if p["p"] is not None]
    return {
        "status": status,
        "reason": reason,
        "errors": errors,
        "abs_errors": mags,
        "pair_orders": pair_orders,
        "p_best": (min(ps) if ps else None),
        "p_mean": (sum(ps) / len(ps) if ps else None),
        "order_spread": ((max(ps) - min(ps)) if len(ps) > 1 else (0.0 if ps else None)),
        "monotone_magnitude": decreasing,
    }


def richardson_extrapolate(fine: Rung, coarse: Rung, order: float | None) -> float | None:
    """Richardson extrapolation of two rungs: ``f_ex = f_f + (f_f - f_c)/(r^p - 1)``.

    ``None`` for a non-positive/undefined order (nothing to extrapolate with) or
    a degenerate ``r^p == 1``.
    """
    if order is None or not (order > 0.0) or not math.isfinite(order):
        return None
    r = _ratio(fine, coarse)
    denom = r ** order - 1.0
    if denom == 0.0:
        return None
    return fine.value + (fine.value - coarse.value) / denom


def gci_percent(
    fine: Rung,
    coarse: Rung,
    order: float | None = None,
    safety: float = DEFAULT_SAFETY,
    scale: float | None = None,
) -> dict | None:
    """Grid Convergence Index of the fine grid, in per cent of ``scale``.

    ``GCI = F_s * |f_f - f_c| / scale / (r^p - 1)`` (Roache 1994).  ``scale``
    defaults to ``|f_fine|`` (relative GCI); a phase functional passes the
    declared phase scale instead.  Returns the ingredients so a reader can
    recompute it; ``None`` when no positive order is available -- the caller must
    then report "no GCI" rather than quote one.
    """
    p = CONSERVATIVE_ORDER if order is None else order
    if not (p > 0.0) or not math.isfinite(p):
        return None
    denom_scale = abs(fine.value) if scale is None else abs(scale)
    if denom_scale == 0.0:
        return None
    r = _ratio(fine, coarse)
    denom = r ** p - 1.0
    if denom <= 0.0:
        return None
    rel = abs((fine.value - coarse.value) / denom_scale)
    gci = safety * rel / denom
    return {
        "gci_percent": 100.0 * gci,
        "order_used": p,
        "order_observed": order,
        "order_convention": "conservative p=1" if order is None else "observed p",
        "safety": safety,
        "r": r,
        "pair_diff_over_scale": rel,
        "scale": denom_scale,
        "scale_basis": "declared" if scale is not None else "|fine value|",
        "fine": fine.value,
        "coarse": coarse.value,
    }


def repeatability(values: Sequence[float]) -> dict:
    """Statistical scatter of repeated measurements of the *same* quantity.

    The usual source in these benchmarks is two consecutive measurement windows
    of a periodic state.  Returns the half-range, the sample std, and the
    half-range as a percentage of the mean -- what the validation step wants for
    ``U_input``.
    """
    xs = [float(v) for v in values]
    if len(xs) < 2:
        return {"n": len(xs), "half_range": 0.0, "std": 0.0,
                "half_range_percent_of_mean": 0.0, "mean": xs[0] if xs else 0.0}
    mean = sum(xs) / len(xs)
    half_range = 0.5 * (max(xs) - min(xs))
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return {
        "n": len(xs),
        "mean": mean,
        "half_range": half_range,
        "std": math.sqrt(var),
        "half_range_percent_of_mean": (100.0 * half_range / abs(mean)) if mean else 0.0,
    }


def validation(
    comparison: float,
    truth: float,
    u_num_percent: float,
    u_input_percent: float = 0.0,
    tolerance_percent: float | None = None,
    scale: float | None = None,
) -> dict:
    """ASME V&V 20-2009 validation comparison for one scalar.

    ``E = |S - T| / scale``, ``U_V = sqrt(U_num^2 + U_input^2)``.  ``verdict`` is
    ``"validated"`` when ``E <= U_V`` (agreement inside the uncertainty band) and
    ``"not_established"`` otherwise.  ``tolerance_percent`` reports the ledger
    acceptance test separately -- and ``supports_tolerance`` reports the stricter,
    more useful question: is the uncertainty band itself narrow enough to
    *support* the claimed tolerance (``U_V <= tol``)?  A case can pass its
    tolerance on a lucky point error and still be unable to certify that
    accuracy.
    """
    denom_scale = abs(truth) if scale is None else abs(scale)
    if denom_scale == 0.0:
        raise ValueError("scale/truth must be non-zero for a relative comparison")
    e_pct = 100.0 * abs(comparison - truth) / denom_scale
    u_v = math.hypot(u_num_percent, u_input_percent)
    out = {
        "comparison": comparison,
        "truth": truth,
        "E_percent": e_pct,
        "U_num_percent": u_num_percent,
        "U_input_percent": u_input_percent,
        "U_V_percent": u_v,
        "E_over_U_V": (e_pct / u_v) if u_v > 0 else None,
        "verdict": "validated" if (u_v > 0 and e_pct <= u_v) else "not_established",
        "scale_basis": "declared" if scale is not None else "|truth|",
    }
    if tolerance_percent is not None:
        out["tolerance_percent"] = tolerance_percent
        out["within_tolerance"] = e_pct <= tolerance_percent
        out["supports_tolerance"] = u_v <= tolerance_percent
    return out


def ladder_analysis(
    rungs: Sequence[Rung],
    truth: float | Sequence[float] | None = None,
    tolerance_percent: float | None = None,
    u_input_percent: float = 0.0,
    safety: float = DEFAULT_SAFETY,
    scale: float | None = None,
) -> dict:
    """Full verification/validation report for one scalar over a ladder.

    ``rungs`` are sorted fine -> coarse internally (a per-rung reference is
    carried along).  Without ``truth`` no order can be identified here (the
    three rungs of the case have unequal ratios), so only the conservative GCI is
    offered -- and it is labelled as such.
    """
    if truth is None:
        rs = sorted(rungs, key=lambda r: r.spacing)
        ts = None
    else:
        rs, ts = align_truth(rungs, truth)

    report: dict = {
        "n_rungs": len(rs),
        "spacings": [r.spacing for r in rs],
        "values": [r.value for r in rs],
        "labels": [r.label for r in rs],
        "truth": ts[0] if ts else None,
        "truth_per_rung": ts,
        "scale": scale,
        "tolerance_percent": tolerance_percent,
    }

    order = None
    identified = False
    if ts is not None:
        errs = order_from_errors(rs, ts)
        report["order_analysis"] = errs
        identified = errs["status"] == "converging"
        if errs["status"] in {"converging", "order_unresolved"} and errs["p_best"]:
            # An unresolved order is still the least-wrong exponent for the GCI
            # denominator, so it is used there -- but it is *not* an order claim:
            # p_identified stays False and no Richardson extrapolation is offered.
            order = errs["p_best"]
        if not identified:
            report["order_claim"] = (
                "no order of accuracy claimed: "
                + (errs.get("reason") or errs["status"]))
    elif len(rs) >= 3:
        report["order_analysis"] = {
            "status": "insufficient",
            "reason": ("order identification needs a known reference: the rung "
                       "ratios here are unequal, so the classical equal-ratio "
                       "triple-grid formula does not apply"),
            "pair_orders": [],
        }

    report["p_used"] = order if order is not None else CONSERVATIVE_ORDER
    report["p_identified"] = identified

    if len(rs) >= 2:
        fine, coarse = rs[0], rs[1]
        gci_obs = gci_percent(fine, coarse, order, safety, scale) if order else None
        gci_cons = gci_percent(fine, coarse, CONSERVATIVE_ORDER, safety, scale)
        report["gci_fine_pairs"] = {"observed_order": gci_obs, "conservative": gci_cons}
        report["richardson"] = (
            richardson_extrapolate(fine, coarse, order) if identified else None)
        # The selected band is the *larger* of the two: on three unequal ratios
        # the observed order is itself uncertain, so the conservative p=1 band is
        # kept as the floor of the uncertainty claim.
        chosen = None
        for cand in (gci_obs, gci_cons):
            if cand is None:
                continue
            chosen = cand if chosen is None or cand["gci_percent"] > chosen["gci_percent"] else chosen
        report["U_num_percent"] = chosen["gci_percent"] if chosen else None
        report["U_num_basis"] = chosen["order_convention"] if chosen else None
        report["U_num_spread_percent"] = (
            [gci_obs["gci_percent"], gci_cons["gci_percent"]] if gci_obs else
            ([gci_cons["gci_percent"]] if gci_cons else None))

    if ts is not None:
        u_num = report.get("U_num_percent")
        if u_num is not None:
            report["validation"] = validation(
                rs[0].value, ts[0], u_num, u_input_percent, tolerance_percent, scale
            )
        else:
            report["validation"] = {"verdict": "not_established",
                                    "reason": "no numerical uncertainty available"}
    return report
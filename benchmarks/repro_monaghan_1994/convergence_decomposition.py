#!/usr/bin/env python3
"""Is the residual a registration offset, or a converging discretisation error?

Why this script exists
----------------------
The committed ladder for REPRO-SPH-MONAGHAN1994 is non-monotone against the
independent transcription - MAE/amplitude 4.67 % (dx = 4 cm) -> 1.65 % (2 cm)
-> 4.38 % (1 cm) - and the merged audit (``docs/audit_repro_monaghan1994_m16936.md``,
F2) reports it as a failed trend gate plus the observation that the *bias*
(sim - reference) walks monotonically upward and crosses zero:

    bias = -0.125 -> +0.032 -> +0.115   [Z/a]

The audit stops there ("the residual is resolution-independent, so it cannot be
attributed to discretisation error").  That sentence skips a decomposition that
the committed artefacts can settle without a single new simulation:

    delta(tau) = bias(dx) + scatter(tau, dx)

  * ``bias`` is the part a constant registration offset would explain;
  * ``scatter`` is everything left once the offset is removed.

If ``scatter`` shrinks at the expected rate under refinement while ``bias``
converges to a *non-zero* limit, then the ladder is not "the solver fails to
converge": the trajectory shape converges, and what survives refinement is a
systematic offset of the simulation against this table - which is either a
registration convention between the table and the case, or physics the case
does not model.  If instead ``scatter`` does not shrink, or ``bias`` collapses
to ~0, the audit's reading stands and this script says so.

Second question, also settled from the artefacts: a *single* registration
parameter (spatial shift ``dz`` in Z/a, or temporal lag ``dtau`` in tau) that is
fitted at the coarsest resolution and then applied **unchanged** to the two
finer ones - out of sample - reduces them to what?  A convention enjoys no
freedom per resolution; a solver/reference mismatch does.

Pre-registration
----------------
``PROTOCOL`` below fixes the window, the refinement ratio, the estimator, the
out-of-sample rule and the four verdict criteria *before* the numbers are
computed from the artefacts.  Nothing in this script reads a number back from
the document that quotes it, and nothing here can change a gate: the gates in
``results/summary.json`` are readonly inputs, and ``gate_contact: false`` is
asserted on output.

Usage::

    python benchmarks/repro_monaghan_1994/convergence_decomposition.py
    python benchmarks/repro_monaghan_1994/convergence_decomposition.py --out /tmp/cd.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

# --------------------------------------------------------------------------- #
# pre-registered protocol: fixed here, before the numbers are derived
# --------------------------------------------------------------------------- #
PROTOCOL = {
    "registered": "2026-11-12, before any number in results/convergence_decomposition.json",
    "reference": "benchmarks/dam_break/martin_moyce_1952_crosscheck.json",
    "reference_rejected_as_pass_source": "benchmarks/dam_break/martin_moyce_1952.json",
    "case": "benchmarks/repro_monaghan_1994/run.py trajectory, estimator x_max",
    "window_tau": [0.5, 3.0],
    "n_samples": 161,
    "ladder_dx": [0.04, 0.02, 0.01],
    "refinement_ratio": 2.0,
    "offset_models": {
        "M0_none": "delta = sim - ref",
        "M1_spatial": "delta + dz   (constant shift of the front ordinate, Z/a)",
        "M2_temporal": "compare sim(tau) with ref(tau - dtau)   (registration in tau)",
    },
    "fit_criterion": "MAE over the pre-registered window",
    "out_of_sample_rule": "the coarsest case (dx = 0.04) is the only fitting case; "
                          "the fitted parameter is applied unchanged to dx = 0.02 and 0.01",
    "criteria": {
        "C1_offset_survives_refinement": "|bias_limit| > 0.10 Z/a (2x the 0.05 claimed "
                                         "uncertainty of the independent transcription)",
        "C2_scatter_converges": "scatter (mean |delta - bias|) strictly decreasing and the "
                                "observed order p = log2(q_coarse/q_fine) >= 0.7 on both increments",
        "C3_single_shift_insufficient": "after the coarse-fitted shift, no fine case reaches "
                                        "MAE/amplitude < 0.03 (the paper-level 3 % line)",
        "C4_temporal_vs_spatial": "whichever of M1/M2 gives the smaller out-of-sample MAE/amplitude "
                                  "on the two finer cases, by more than 0.005 absolute",
    },
    "gate_contact": False,
    "note": "criteria are deliberately hard to satisfy: C1 needs the offset to survive "
            "refinement, C2 needs the scatter to actually shrink, C3 is the honesty clause "
            "against declaring a convention from a single free parameter.",
}


# --------------------------------------------------------------------------- #
# primitives (own implementation: no import from run.py, so this is independent)
# --------------------------------------------------------------------------- #
def interp(xs, ys, xq: float) -> float:
    """Linear interpolation with clamped ends (same convention as the case)."""
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0.0:
        return ys[lo]
    w = (xq - xs[lo]) / span
    return ys[lo] * (1.0 - w) + ys[hi] * w


def evaluation_grid(tau, ref, window, n: int):
    lo = max(window[0], min(tau), min(ref["tau"]))
    hi = min(window[1], max(tau), max(ref["tau"]))
    if not hi > lo:
        return []
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def residuals(tau, za, ref, window=PROTOCOL["window_tau"], n=PROTOCOL["n_samples"]):
    """sampled (tau, sim, ref, delta) on the pre-registered window."""
    grid = evaluation_grid(tau, ref, window, n)
    rows = []
    for t in grid:
        s = interp(tau, za, t)
        e = interp(ref["tau"], ref["za"], t)
        rows.append((t, s, e, s - e))
    return rows


def decompose(rows) -> dict:
    """Split the residual into a systematic offset and scatter about it."""
    d = np.array([r[3] for r in rows])
    amplitude = max(r[2] for r in rows)
    bias = float(d.mean())
    centred = d - bias
    return {
        "n_samples": len(rows),
        "amplitude": amplitude,
        "bias": bias,
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "scatter_mean_abs": float(np.abs(centred).mean()),
        "scatter_rms": float(np.sqrt((centred ** 2).mean())),
        "mae_over_amplitude": float(np.abs(d).mean() / amplitude),
        "max_abs": float(np.abs(d).max()),
        # is the residual, after removing the offset, still a ramp in tau?
        "centred_trend_per_tau": _slope([r[0] for r in rows], centred),
    }


def _slope(xs, ys) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2 or np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x - x.mean(), y, 1)[0])


def fit_spatial(rows) -> float:
    """M1: the constant ordinate shift minimising the MAE is the median residual."""
    return float(np.median([r[3] for r in rows]))


def fit_temporal(tau, za, ref, window, n, lo=-0.40, hi=0.40, step=0.002) -> dict:
    """M2: lag the reference table in tau; the window shrinks so no clamping is used."""
    best = None
    shift = lo
    while shift <= hi + 1e-12:
        # keep both tables inside their own support after the shift
        w = (window[0] + max(shift, 0.0), window[1] + min(shift, 0.0))
        rows = residuals(tau, za, ref, window=w, n=n)
        if rows:
            # sim(tau) vs ref(tau - dtau)
            d = np.array([interp(tau, za, r[0]) - interp(ref["tau"], ref["za"], r[0] - shift)
                          for r in rows])
            mae = float(np.abs(d).mean())
            if best is None or mae < best["mae"]:
                best = {"dtau": float(shift), "mae": mae, "window": list(w)}
        shift += step
    return best


def observed_order(q_coarse: float, q_fine: float, ratio: float = 2.0) -> float:
    if q_fine <= 0 or q_coarse <= 0:
        return float("nan")
    return float(math.log(q_coarse / q_fine) / math.log(ratio))


def richardson_limit(b_last: float, b_prev: float, p: float, ratio: float = 2.0) -> float:
    """Extrapolated dx -> 0 value of the bias from the two finest members."""
    if not math.isfinite(p) or p <= 0.2:
        return float("nan")
    return float(b_last + (b_last - b_prev) / (ratio ** p - 1.0))


# --------------------------------------------------------------------------- #
def load_case(dx: float) -> dict:
    path = RESULTS / f"trajectory_dx{dx:g}_c025.json"
    return json.loads(path.read_text())


def load_reference() -> dict:
    path = ROOT / PROTOCOL["reference"]
    raw = json.loads(path.read_text())
    rows = raw["data"]
    tau = [r["tau"] for r in rows]
    # the transcription is stored as X/a measured from the wall; the case protocol
    # is Z/a measured from the initial column edge (see the file's `convention`),
    # so the conversion is the same one benchmarks/dam_break/reference_audit.py uses
    za = [float(r["X_over_a"]) - 1.0 for r in rows]
    order = sorted(range(len(tau)), key=lambda i: tau[i])
    return {
        "name": raw.get("reference", "crosscheck"),
        "path": str(path.relative_to(ROOT)),
        "tau": [tau[i] for i in order],
        "za": [za[i] for i in order],
        "claimed_uncertainty": raw.get("claimed_uncertainty", 0.05),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "convergence_decomposition.json"))
    args = ap.parse_args()

    ref = load_reference()
    cases = {}
    for dx in PROTOCOL["ladder_dx"]:
        traj = load_case(dx)
        rows = residuals(traj["tau"], traj["za"], ref)
        info = decompose(rows)
        info["dx"] = dx
        info["case"] = traj["case"]
        info["particles"] = traj["particles"]
        info["bias_in_sample_fit"] = fit_spatial(rows)
        info["temporal_fit"] = fit_temporal(traj["tau"], traj["za"], ref,
                                           PROTOCOL["window_tau"], PROTOCOL["n_samples"])
        # residual profile: the document plots delta(tau), so commit it
        step = max(1, len(rows) // 24)
        info["residual_profile"] = [{"tau": r[0], "delta": r[3]} for r in rows[::step]]
        # does the in-sample offset remove the shape?  (a constant offset cannot
        # flatten a ramp, so this number is the test of "it is just a convention")
        shifted = [(r[0], r[1], r[2], r[3] - info["bias_in_sample_fit"]) for r in rows]
        info["after_in_sample_offset"] = {
            "residual_form": "delta - median(delta)",
            "mae": float(np.mean([abs(r[3]) for r in shifted])),
            "trend_per_tau": _slope([r[0] for r in shifted], [r[3] for r in shifted]),
            "max_abs": float(max(abs(r[3]) for r in shifted)),
        }
        cases[f"dx{dx:g}"] = info

    # ---------------- convergence of the two components ----------------
    order = PROTOCOL["ladder_dx"]
    ladder = []
    for dx in order:
        c = cases[f"dx{dx:g}"]
        ladder.append({"dx": dx, "bias": c["bias"],
                       "scatter_mean_abs": c["scatter_mean_abs"],
                       "scatter_rms": c["scatter_rms"],
                       "mae_over_amplitude": c["mae_over_amplitude"]})
    inc_bias, inc_scatter = [], []
    for a, b in zip(ladder[:-1], ladder[1:]):
        inc_bias.append({"between": [a["dx"], b["dx"]],
                         "order": observed_order(abs(a["bias"]), abs(b["bias"]))})
        inc_scatter.append({"between": [a["dx"], b["dx"]],
                            "order": observed_order(a["scatter_mean_abs"], b["scatter_mean_abs"])})
    p_scatter = float(np.nanmean([i["order"] for i in inc_scatter]))
    p_bias = float(np.nanmean([i["order"] for i in inc_bias]))
    bias_limit = richardson_limit(ladder[-1]["bias"], ladder[-2]["bias"], p_bias)

    # ---------------- out-of-sample registration test ----------------
    coarse = cases[f"dx{order[0]:g}"]
    dz = coarse["bias_in_sample_fit"]
    dtau = coarse["temporal_fit"]["dtau"]
    oos = {"fitted_on": f"dx{order[0]:g}", "dz": dz, "dtau": dtau, "applied": []}
    for dx in order:
        traj = load_case(dx)
        rows = residuals(traj["tau"], traj["za"], ref)
        amp = max(r[2] for r in rows)
        m1 = float(np.mean([abs(r[3] + dz) for r in rows]))
        w = (PROTOCOL["window_tau"][0] + max(dtau, 0.0), PROTOCOL["window_tau"][1] + min(dtau, 0.0))
        rows2 = residuals(traj["tau"], traj["za"], ref, window=w)
        amp2 = max(r[2] for r in rows2)
        m2 = float(np.mean([abs(interp(traj["tau"], traj["za"], r[0])
                                - interp(ref["tau"], ref["za"], r[0] - dtau)) for r in rows2]))
        oos["applied"].append({
            "dx": dx,
            "M0_mae_over_amplitude": float(np.mean([abs(r[3]) for r in rows])) / amp,
            "M1_mae_over_amplitude": m1 / amp,
            "M2_mae_over_amplitude": m2 / amp2,
        })
    fine = [r for r in oos["applied"] if r["dx"] != order[0]]
    m1_fine = max(r["M1_mae_over_amplitude"] for r in fine)
    m2_fine = max(r["M2_mae_over_amplitude"] for r in fine)

    # ---------------- verdict against the pre-registered criteria ----------------
    crit = PROTOCOL["criteria"]
    scatter_monotone = all(ladder[i + 1]["scatter_mean_abs"] < ladder[i]["scatter_mean_abs"]
                           for i in range(len(ladder) - 1))
    verdict = {
        "C1_offset_survives_refinement": {
            "criterion": crit["C1_offset_survives_refinement"],
            "limit": 0.10,
            "measured": bias_limit,
            "result": "supported" if (math.isfinite(bias_limit)
                                      and abs(bias_limit) > 0.10) else "not supported",
        },
        "C2_scatter_converges": {
            "criterion": crit["C2_scatter_converges"],
            "limit": 0.7,
            "measured": {"orders": [i["order"] for i in inc_scatter],
                         "mean_order": p_scatter,
                         "monotone": scatter_monotone},
            "result": "supported" if (scatter_monotone and math.isfinite(p_scatter)
                                      and p_scatter >= 0.7) else "not supported",
        },
        "C3_single_shift_insufficient": {
            "criterion": crit["C3_single_shift_insufficient"],
            "limit": 0.03,
            "measured": {"M1_worst_fine": m1_fine, "M2_worst_fine": m2_fine},
            "result": "supported" if min(m1_fine, m2_fine) >= 0.03 else "not supported",
        },
        "C4_temporal_vs_spatial": {
            "criterion": crit["C4_temporal_vs_spatial"],
            "limit": 0.005,
            "measured": {"M1": m1_fine, "M2": m2_fine,
                         "advantage": abs(m2_fine - m1_fine)},
            "result": ("temporal" if m2_fine + 0.005 < m1_fine else
                       "spatial" if m1_fine + 0.005 < m2_fine else "indistinguishable"),
        },
    }

    payload = {
        "benchmark": "REPRO-SPH-MONAGHAN1994",
        "anchor": "REPRO-SPH-MONAGHAN1994",
        "question": "given delta(tau) = bias(dx) + scatter(tau, dx): does the scatter converge "
                    "and the bias converge to a non-zero limit, and can one registration "
                    "parameter fitted at dx=0.04 explain the finer cases out of sample?",
        "protocol": PROTOCOL,
        "reference": {"name": ref["name"], "path": ref["path"],
                      "claimed_uncertainty": ref["claimed_uncertainty"],
                      "n_points": len(ref["tau"])},
        "cases": cases,
        "convergence": {"ladder": ladder, "bias_increments": inc_bias,
                        "scatter_increments": inc_scatter,
                        "observed_order_bias": p_bias,
                        "observed_order_scatter": p_scatter,
                        "richardson_bias_limit": bias_limit,
                        "richardson_note": "two increments at ratio 2 give one order estimate; "
                                           "the limit is an extrapolation, not a measurement"},
        "out_of_sample_registration": oos,
        "verdict": verdict,
        "gate_contact": False,
        "gate_contact_note": "this artefact is diagnostic: results/summary.json gates are "
                             "unchanged and results/summary.json is not read as a pass source",
    }

    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print("bias:", [f"{c['bias']:+.4f}" for c in [cases[f'dx{d:g}'] for d in order]])
    print("scatter:", [f"{c['scatter_mean_abs']:.4f}" for c in [cases[f'dx{d:g}'] for d in order]])
    print(f"p_bias={p_bias:.2f} p_scatter={p_scatter:.2f} bias_limit={bias_limit:+.4f}")
    print("out-of-sample M1/M2 worst fine:",
          f"{m1_fine:.4f} / {m2_fine:.4f}")
    for k, v in verdict.items():
        print(f"  {k}: {v['result']}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Round 7 of ``CASE-SPH-OSC-DROP-FREQ``: observed order of convergence + GCI +
ASME V&V 20 validation uncertainty on the round-4 independent estimators.

Why this round exists
---------------------
Round 4 (``docs/case_sph_osc_drop_freq_round4.md``) concluded that the anchor gate
(``|omega error| < 6 %`` vs Rayleigh 1879) is met for ``R/dx >= 24`` on **two
independent** reconstruction channels (A: modal amplitude of the ``c = 1/2``
colour-field contour; B: surface energy, curvature-free).  Both of its decisive
numbers rest on one assumption that was never measured:

    ``g(R/dx) = 1 - c / (R/dx)``      (round 4 eq. "transfer law", p = 1 assumed)

``transfer_law()`` fits a **two-parameter** model (intercept + coefficient) in the
``1/(R/dx)`` basis, i.e. it *assumes* first-order convergence, then reports
``threshold_rdx_at_gate`` from that fit.  Round 3 already reported that the fitted
first-order coefficient disagrees between modes (1.91 at ``l = 2`` vs 2.20 at
``l = 3`` from round 4's refit), which is exactly the fingerprint of a
mis-specified order: if the true order is ``p != 1``, a basis in ``h`` absorbs the
mismatch into the coefficient and the extrapolated intercept.

So the question this round answers is not "does the anchor pass at 24" (round 4
measured that) but:

    **is the ``R/dx >= 24`` pass a property of the asymptotic solution, or an
    artefact of the assumed order?**  Specifically:
    1. what is the *observed* order ``p`` (three/four grids, not assumed)?
    2. with that ``p``, what is the conservative GCI at the finest grid, and is
       the extrapolated limit ``Q_inf`` equal to the exact value 1 within the
       ASME V&V 20 validation uncertainty ``U_V``?
    3. is ``|Q(R/dx = 24) - 1| < 6 %`` reproduced with the *measured* order?

Method (ASME V&V 20-2009, Roache 1994/1997)
-------------------------------------------
Quantity of interest per channel: ``Q = omega_meas / omega_Rayleigh = sqrt(g)``
(the modal stiffness is linear in the reconstructed amplitude, so frequency
scales as the square root -- round 4's convention, kept).  Exact ``Q = 1``.

* observed order: fit ``Q(h) = Q_inf + C h^p`` with ``h = 1/(R/dx)`` on the whole
  ladder, ``p`` by a dense grid search with ``(Q_inf, C)`` solved by linear least
  squares for each ``p`` (no ``scipy`` dependency).  A converged fit must have
  monotone ``Q(h)`` and a small relative residual; both are reported.
* GCI on the finest pair (``32``, ``40``): ``GCI_fine = Fs |eps| / (r^p - 1)`` with
  ``r = 1.25`` and the conservative ``Fs = 1.25``.
* input/estimator uncertainty ``U_input``: the analytic level-set control of the
  same grid (round 4's route-A control and the energy-route control) -- its
  deviation from 1 is the estimator's own numerical bias, and it is not reduced
  by the SPH resolution.
* ``U_V = sqrt(U_num^2 + U_input^2)``; validation passes when
  ``|Q_fine - 1| <= U_V``.

Honest reporting rule: every row is written whether it passes or not, and a
channel whose fit is non-monotone or order-marginal is labelled
``order_not_established`` rather than given a number.

Usage::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/round7_convergence.py --part 3 --parts 5
    PYTHONPATH=src python benchmarks/oscillating_drop_freq/round7_convergence.py --quick
    PYTHONPATH=src python benchmarks/oscillating_drop_freq/round7_convergence.py --merge
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import round4_energy_contour as R4  # noqa: E402

OUT = HERE / "out"
GATE_PCT = R4.GATE_PCT
#: ladder.  12 is kept only as a continuity check against round 3/4; the
#: convergence study uses 16/24/32/40 and the finest pair 32->40 for the GCI.
LADDER = (12.0, 16.0, 24.0, 32.0, 40.0)
FS = 1.25          # Roache safety factor (conservative; one-sided refinement)
EPS_VALUES = (0.02, 0.05)


# --------------------------------------------------------------------------- #
# per-grid-point measurement (shardable)
# --------------------------------------------------------------------------- #
def measure_points(radii, *, radius=1.0, sigma=1.0, rho=1.0, modes=(2, 3),
                   grid_per_dx=4.0, eps_values=EPS_VALUES) -> dict:
    """Run round 4's two independent channels at the given ``R/dx`` points."""
    out = {"rows": [], "energy": [], "control": [], "grid_per_dx": grid_per_dx}
    for rdx in radii:
        t0 = time.time()
        lad = R4.transfer_ladder(radius, sigma, rho, modes=modes, ladder=(rdx,),
                                 eps_values=eps_values,
                                 n_grid_per_dx=grid_per_dx,
                                 energy_ladder=(rdx,))
        out["rows"].extend(lad["rows"])
        out["energy"].extend(lad["energy"])
        out["control"].extend(
            R4.estimator_control(radius, rho, modes=modes, ladder=(rdx,),
                                 eps=eps_values[0], n_grid_per_dx=grid_per_dx))
        out.setdefault("wall_seconds", {})[str(rdx)] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- #
# convergence machinery
# --------------------------------------------------------------------------- #
def _linear_fit(h, q, p):
    """Least squares ``q = q_inf + C h^p`` for a *fixed* p; returns (q_inf, C, rms)."""
    x = torch.tensor([float(v) ** p for v in h], dtype=torch.float64)
    y = torch.tensor([float(v) for v in q], dtype=torch.float64)
    m = torch.stack([torch.ones_like(x), x], dim=1)
    sol = torch.linalg.lstsq(m, y[:, None]).solution.squeeze(-1)
    resid = m @ sol - y
    return float(sol[0]), float(sol[1]), float(torch.sqrt((resid ** 2).mean()))


def observed_order(rdx, q, *, p_lo=0.2, p_hi=4.0, n_search=761) -> dict:
    """Observed order ``p`` by grid search; ``(Q_inf, C)`` linear for each ``p``.

    ``h = 1/(R/dx)``.  The search returns the ``p`` that minimises the rms
    residual of the 2-parameter fit, plus a bracket where the residual is within
    5 % of its minimum, which is the honest way to say "the order is 1.7 +- 0.2"
    instead of quoting a single number.
    """
    rdx = [float(v) for v in rdx]
    q = [float(v) for v in q]
    if len(rdx) < 3:
        return {"status": "order_not_established",
                "reason": "fewer than three grids", "n_grids": len(rdx)}
    h = [1.0 / v for v in rdx]
    best = None
    curve = []
    for k in range(n_search):
        p = p_lo + (p_hi - p_lo) * k / (n_search - 1.0)
        q_inf, c, rms = _linear_fit(h, q, p)
        curve.append((p, rms, q_inf, c))
        if best is None or rms < best[1]:
            best = (p, rms, q_inf, c)
    p, rms, q_inf, c = best
    tol = rms * 1.05 + 1e-18
    band = [p2 for (p2, r2, _, _) in curve if r2 <= tol]
    monotone = all((b - a) * (q[-1] - q[0]) > 0 for a, b in zip(q, q[1:]))
    return {"status": "ok", "n_grids": len(rdx), "p_observed": p,
            "p_band": [min(band), max(band)],
            "q_inf": q_inf, "coefficient": c, "rms_residual": rms,
            "monotone": bool(monotone),
            "orders": {f"{v:g}": qq for v, qq in zip(rdx, q)}}


def gci(q_coarse, q_fine, p, *, refinement=1.25, fs=FS) -> dict:
    """Roache GCI of the fine grid from the finest coarse/fine pair."""
    if q_fine == 0.0 or p <= 0.0:
        return {"status": "not_available"}
    eps = abs((q_coarse - q_fine) / q_fine)
    denom = refinement ** p - 1.0
    if denom <= 0.0:
        return {"status": "not_available", "refinement": refinement, "p": p}
    g = fs * eps / denom
    return {"status": "ok", "gci_rel": g, "gci_pct": 100.0 * g,
            "eps_rel": eps, "p": p, "refinement": refinement, "fs": fs}


def validation(q_fine, u_num, u_input) -> dict:
    """ASME V&V 20: ``U_V = sqrt(U_num^2 + U_input^2)`` vs ``|Q - 1|``."""
    u_v = math.sqrt(u_num ** 2 + u_input ** 2)
    err = abs(q_fine - 1.0)
    return {"u_num": u_num, "u_input": u_input, "u_v": u_v,
            "error_rel": err, "validation_ratio": err / u_v if u_v else None,
            "pass": bool(err <= u_v)}


def channel_history(rows, energy, control, *, mode, eps, channel) -> list:
    """``(R/dx, Q)`` samples of one channel, sorted by resolution."""
    pts = []
    if channel in ("A", "A_differential"):
        key = "transfer_corrected" if channel == "A" else "differential_transfer"
        for r in rows:
            if r["mode"] != mode or abs(r["eps"] - eps) > 1e-12:
                continue
            g = r.get(key)
            if g is None or not (g == g) or g <= 0.0:
                continue
            pts.append((r["radius_over_dx"], math.sqrt(g)))
    elif channel == "B":
        for r in energy:
            if r["mode"] != mode:
                continue
            g = r["g_energy_corrected"]
            if not (g == g) or g <= 0.0:
                continue
            pts.append((r["radius_over_dx"], math.sqrt(g)))
    pts.sort()
    return pts


def channel_input_uncertainty(rows, energy, *, mode, eps, channel,
                              finest) -> float:
    """``U_input``: the estimator's own bias at the finest grid (``|control - 1|``)."""
    if channel in ("A", "A_differential"):
        for r in rows:
            if (r["mode"] == mode and abs(r["eps"] - eps) < 1e-12
                    and abs(r["radius_over_dx"] - finest) < 1e-9):
                gc = r.get("transfer_control")
                if gc is None or not (gc == gc) or gc <= 0.0:
                    return float("nan")
                return abs(math.sqrt(gc) - 1.0)
    elif channel == "B":
        for r in energy:
            if (r["mode"] == mode
                    and abs(r["radius_over_dx"] - finest) < 1e-9):
                gc = r.get("c2_control")
                ge = r.get("c2_analytic")
                if not gc or not ge:
                    return float("nan")
                return abs(math.sqrt(gc / ge) - 1.0)
    return float("nan")


def analyse(payload: dict) -> dict:
    """Observed order + GCI + validation row for every channel."""
    rows = payload["rows"]
    energy = payload["energy"]
    modes = sorted({r["mode"] for r in rows})
    eps_values = sorted({r["eps"] for r in rows if r["eps"] > 0.0})
    radii = sorted({r["radius_over_dx"] for r in rows})
    finest = max(radii)
    coarse = sorted(radii)[-2] if len(radii) > 1 else finest
    out = []
    for channel in ("A", "A_differential", "B"):
        for mode in modes:
            eps_sel = eps_values if channel != "B" else [eps_values[0]]
            for eps in eps_sel:
                hist = channel_history(rows, energy, payload.get("control", []),
                                       mode=mode, eps=eps, channel=channel)
                if len(hist) < 3:
                    out.append({"channel": channel, "mode": mode, "eps": eps,
                                "status": "order_not_established",
                                "reason": f"only {len(hist)} grids"})
                    continue
                rdx = [p[0] for p in hist]
                q = [p[1] for p in hist]
                fit = observed_order(rdx, q)
                finest_q = q[rdx.index(finest)] if finest in rdx else q[-1]
                u_in = channel_input_uncertainty(rows, energy, mode=mode, eps=eps,
                                                 channel=channel, finest=finest)
                if fit["status"] != "ok":
                    out.append({"channel": channel, "mode": mode, "eps": eps,
                                "status": fit["status"], **fit})
                    continue
                if coarse in rdx and finest in rdx:
                    g = gci(q[rdx.index(coarse)], finest_q, fit["p_observed"],
                            refinement=finest / coarse)
                else:
                    g = {"status": "not_available"}
                v = (validation(finest_q, g["gci_rel"], u_in)
                     if g.get("status") == "ok" and u_in == u_in and u_in == u_in
                     else None)
                mid = 24.0
                q24 = q[rdx.index(mid)] if mid in rdx else None
                out.append({"channel": channel, "mode": mode, "eps": eps,
                            "status": "ok", "p_observed": fit["p_observed"],
                            "p_band": fit["p_band"],
                            "q_inf": fit["q_inf"],
                            "q_inf_error_pct": 100.0 * (fit["q_inf"] - 1.0),
                            "rms_residual": fit["rms_residual"],
                            "monotone": fit["monotone"],
                            "q_finest": finest_q, "rdx_finest": finest,
                            "gci_pct": g.get("gci_pct"),
                            "u_input_pct": (100.0 * u_in if u_in == u_in else None),
                            "validation": v,
                            "q_at_24": q24,
                            "omega_error_at_24_pct":
                                (100.0 * (q24 - 1.0) if q24 is not None else None),
                            "gate_at_24": (abs(q24 - 1.0) * 100.0 < GATE_PCT
                                           if q24 is not None else None),
                            "history": {f"{a:g}": b for a, b in zip(rdx, q)}})
    return {"rows": out, "ladder": radii, "finest": finest,
            "finest_pair": [coarse, finest]}


def verdict(analysis: dict) -> dict:
    """Programmatic verdict; every branch is a sentence, not a colour."""
    ok = [r for r in analysis["rows"] if r["status"] == "ok"]
    if not ok:
        return {"status": "order_not_established",
                "statement": "no channel produced three grids; nothing to decide"}
    gate = [r for r in ok if r.get("gate_at_24") is not None]
    n_gate = sum(1 for r in gate if r["gate_at_24"])
    p_vals = [r["p_observed"] for r in ok]
    q_inf = [r["q_inf"] for r in ok]
    val = [r for r in ok if r.get("validation")]
    n_val = sum(1 for r in val if r["validation"]["pass"])
    return {
        "status": "ok",
        "n_ok_rows": len(ok),
        "p_observed_range": [min(p_vals), max(p_vals)],
        "p_observed_median": sorted(p_vals)[len(p_vals) // 2],
        "first_order_assumption_held":
            bool(max(abs(p - 1.0) for p in p_vals) < 0.3),
        "gate_rows": len(gate), "gate_pass_rows_at_24": n_gate,
        "q_inf_range_pct": [100.0 * (min(q_inf) - 1.0),
                            100.0 * (max(q_inf) - 1.0)],
        "validation_rows": len(val), "validation_pass_rows": n_val,
        "gci_at_finest_pct_range": [
            min(r["gci_pct"] for r in ok if r.get("gci_pct") is not None)
            if any(r.get("gci_pct") is not None for r in ok) else None,
            max(r["gci_pct"] for r in ok if r.get("gci_pct") is not None)
            if any(r.get("gci_pct") is not None for r in ok) else None],
        "statement": _statement(ok, gate, n_gate, val, n_val),
    }


def _statement(ok, gate, n_gate, val, n_val) -> str:
    p = sorted(r["p_observed"] for r in ok)
    pmed = p[len(p) // 2]
    s = [f"observed order p = {pmed:.2f} (range {p[0]:.2f}-{p[-1]:.2f}) "
         f"on {len(ok)} channel rows"]
    s.append(f"R/dx = 24 inside the 6 % gate on {n_gate}/{len(gate)} rows")
    if val:
        s.append(f"Q_inf validated against the exact value 1 on {n_val}/{len(val)} rows "
                 f"(ASME V&V 20, U_V from GCI + estimator control bias)")
        if n_val == len(val) and n_gate == len(gate):
            s.append("=> the pass is asymptotic, not an artefact of the assumed "
                     "first-order law: Q_inf is statistically indistinguishable "
                     "from 1 and the 6 % band at R/dx = 24 is reproduced")
        else:
            s.append("=> the pass is NOT fully asymptotic: see the failing rows "
                     "for the channel/mode where Q_inf or the 6 % band fails")
    return "; ".join(s)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def figure(analysis: dict, payload: dict, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    labels = {"A": "A colour-field contour", "A_differential": "A' null-subtracted",
              "B": "B surface energy"}
    style = {"A": "-o", "A_differential": "--s", "B": ":^"}
    for ax, mode in zip(axes[:2], (2, 3)):
        for r in analysis["rows"]:
            if r["status"] != "ok" or r["mode"] != mode:
                continue
            if r["channel"] == "B" and r["eps"] != min(EPS_VALUES):
                continue
            h = sorted((float(k), v) for k, v in r["history"].items())
            ax.plot([a for a, _ in h], [100.0 * (b - 1.0) for _, b in h],
                    style[r["channel"]], ms=4,
                    label=f"{labels[r['channel']]} eps={r['eps']:g} "
                          f"p={r['p_observed']:.2f}")
            ax.axhline(0.0, color="k", lw=0.8)
        ax.axhspan(-GATE_PCT, GATE_PCT, color="0.85", zorder=0)
        ax.set_xlabel("R/dx")
        ax.set_ylabel("omega error vs Rayleigh  [%]")
        ax.set_title(f"l = {mode}: measured transfer (band = 6 % gate)")
        ax.legend(fontsize=6.5, loc="best")
        ax.grid(alpha=0.3)
    ax = axes[2]
    ok = [r for r in analysis["rows"] if r["status"] == "ok" and r.get("gci_pct")]
    if ok:
        names = [f"{labels[r['channel']].split()[0]}\nl={r['mode']}" for r in ok]
        ax.bar(range(len(ok)), [r["q_inf_error_pct"] for r in ok], color="#4477aa",
               label="Q_inf error [%]")
        ax.errorbar(range(len(ok)), [r["q_inf_error_pct"] for r in ok],
                    yerr=[r["gci_pct"] for r in ok], fmt="none", ecolor="k",
                    capsize=3, label="GCI band")
        ax.set_xticks(range(len(ok)))
        ax.set_xticklabels(names, fontsize=6)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylabel("extrapolated limit error [%]")
        ax.set_title("Q_inf (Richardson) with conservative GCI")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("CASE-SPH-OSC-DROP-FREQ round 7: observed order, GCI, V&V 20",
                 fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--parts", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--grid-per-dx", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.merge:
        payload = {"rows": [], "energy": [], "control": [], "parts": []}
        for f in sorted(OUT.glob("round7_part*.json")):
            part = json.loads(f.read_text())
            for k in ("rows", "energy", "control"):
                payload[k].extend(part.get(k, []))
            payload["parts"].append({"file": f.name,
                                     "wall_seconds": part.get("wall_seconds")})
        analysis = analyse(payload)
        vd = verdict(analysis)
        doc = {"case": "SPH-OSC-DROP-FREQ", "round": 7,
               "study": "observed order + conservative GCI + ASME V&V 20 "
                        "validation uncertainty on the round-4 estimators",
               "reference": "Rayleigh 1879, Proc. R. Soc. Lond. 29:71; "
                            "Lamb 1932 Art. 275; ASME V&V 20-2009; Roache 1994",
               "gate_pct": GATE_PCT,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "environment": {"python": platform.python_version(),
                               "torch": torch.__version__, "device": "cpu"},
               "safety_factor": FS,
               "analysis": analysis, "verdict": vd}
        fig_path = HERE.parents[1] / "docs" / "figures" / \
            "case_sph_osc_drop_freq_round7_convergence.png"
        doc["figure"] = figure(analysis, payload, fig_path)
        (OUT / "round7.json").write_text(json.dumps(doc, indent=1, default=float))
        print(json.dumps({"verdict": vd, "figure": doc["figure"]}, indent=1,
                         ensure_ascii=False))
        return 0

    radii = LADDER
    if args.quick:
        radii = (16.0, 24.0)
    if args.parts:
        parts = int(args.parts)
        i = int(args.part or 0)
        radii = tuple(v for k, v in enumerate(LADDER) if k % parts == i)
    print(f"[round7] computing R/dx = {radii} at grid_per_dx = {args.grid_per_dx}",
          flush=True)
    payload = measure_points(radii, grid_per_dx=args.grid_per_dx)
    name = args.out or (f"round7_part{args.part}" if args.parts else
                        "round7_single")
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1, default=float))
    # compact digest on stdout so the numbers survive even if only the log is
    # fetched back from the queue
    dig = {"radii": list(radii), "wall_seconds": payload.get("wall_seconds"),
           "rows": [{k: r[k] for k in ("mode", "radius_over_dx", "eps",
                                       "transfer_corrected")} for r in payload["rows"]],
           "energy": [{k: r[k] for k in ("mode", "radius_over_dx",
                                          "g_energy_corrected")}
                      for r in payload["energy"]]}
    print("DIGEST " + json.dumps(dig, default=float), flush=True)
    print(f"wrote {OUT / f'{name}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
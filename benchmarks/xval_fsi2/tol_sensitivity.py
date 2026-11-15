"""Tolerance sensitivity of the FSI2 strong-coupling iteration.

Anchor ``XVAL-SPH-FSI2-BENCHMARK-AND-FORCE-FROM-FLUID``.

Question this answers
---------------------
``docs/xval_fsi2/STRONG_COUPLING_AND_ROD_STABILITY.md`` §5.3 records that the
interface iteration stops reducing its residual at ``~1e-2`` because the
load-transfer operator is not smooth in the interface position.  The transient
driver nevertheless defaulted to ``tol = 1e-3``.  Two consequences follow, and
both are measured here on the same case, the same ``dt`` and the same initial
state, changing *only* ``tol``:

1. **Cost** -- when the gate is below the floor, every step runs to ``max_iter``.
   Predicted: the ``1e-3`` arm uses strictly more sub-iterations per step.
2. **Meaning of the returned load** -- the returned load is the *last* iterate's.
   If the iteration has not terminated on its own criterion, that load is
   whatever the residual happened to be when the budget ran out.  Predicted: the
   Cd/Cl series differ materially between the arms, which would mean the
   withdrawn ``1e-3`` benchmark numbers were iteration artefacts, not physics.

Predeclared decision (before inspecting results)
------------------------------------------------
* If the ``1e-3`` arm never converges AND its Cd mean differs from the ``1e-2``
  arm by more than the 15 % acceptance band, the ``1e-3`` run is declared
  *not a valid measurement* and the driver default moves to the measured floor.
* If the two arms agree within 3 %, the tolerance was cosmetic and only a cost
  finding is recorded (no accuracy claim either way).
* Anything between is reported as an unresolved sensitivity band.

Usage::

    python benchmarks/xval_fsi2/tol_sensitivity.py \
        --strict artifacts/xval_fsi2/tol_1e3.json \
        --floor  artifacts/xval_fsi2/tol_1e2.json \
        --out-json docs/xval_fsi2/tol_sensitivity.json \
        --out-fig  docs/xval_fsi2/assets/fig_tol_sensitivity.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.fsi2_reference import summarise_body_forces  # noqa: E402


def _stats(payload: dict) -> dict:
    h = payload["history"]
    it = np.asarray(h["coupling_iterations"], dtype=float)
    # tolerate artefacts written before the double-append bug: the row-key
    # loop filled placeholder NaNs, the real residuals are the finite ones
    res = np.asarray(h.get("coupling_residual", []), dtype=float)
    res = res[np.isfinite(res)]
    conv = np.asarray(h["coupling_converged"], dtype=float)
    case = payload.get("case", {})
    return {
        "n_steps": int(len(h["t"])),
        "t_end": float(h["t"][-1]),
        "iterations_mean": float(it.mean()),
        "iterations_max": float(it.max()),
        "iterations_at_budget": float(np.mean(it >= it.max())),
        "converged_fraction": float(conv.mean()) if conv.size else None,
        "residual_median": float(np.nanmedian(res)) if res.size else None,
        "residual_p10": float(np.nanpercentile(res, 10)) if res.size else None,
        "residual_min": float(np.nanmin(res)) if res.size else None,
        "rho0": float(case.get("rho0", 1000.0)),
        "u_mean": float(case.get("inlet_u_max", 0.3)) / 1.5,
        "D": float(case.get("cylinder_d", 0.1)),
    }


def _series(payload: dict) -> dict:
    h = payload["history"]
    return {k: np.asarray(h[k], dtype=float)
            for k in ("body_drag_fit", "body_lift_fit", "tip_x", "tip_y")}


def analyse(strict: dict, floor: dict) -> dict:
    s_s, s_f = _stats(strict), _stats(floor)
    ser_s, ser_f = _series(strict), _series(floor)
    n = min(len(ser_s["body_drag_fit"]), len(ser_f["body_drag_fit"]))

    # per-channel agreement over the common window
    channels = {}
    for k in ser_s:
        a, b = ser_s[k][:n], ser_f[k][:n]
        denom = max(float(np.abs(b).mean()), 1e-30)
        channels[k] = {
            "rms_abs_diff": float(np.sqrt(np.mean((a - b) ** 2))),
            "rel_rms_diff": float(np.sqrt(np.mean((a - b) ** 2)) / denom),
            "strict_mean": float(a.mean()), "floor_mean": float(b.mean()),
            "strict_halfrange": float(0.5 * (a.max() - a.min())),
            "floor_halfrange": float(0.5 * (b.max() - b.min())),
        }

    # coefficients over the full common window (no warm-up: 200 steps is short)
    sc = s_s["rho0"], s_s["u_mean"], s_s["D"]
    cd_s = summarise_body_forces(ser_s["body_drag_fit"][:n],
                                ser_s["body_lift_fit"][:n], *sc)
    cd_f = summarise_body_forces(ser_f["body_drag_fit"][:n],
                                ser_f["body_lift_fit"][:n], *sc)
    pairs = {}
    for k in ("cd_mean", "cd_amplitude", "cl_amplitude"):
        a, b = cd_s.get(k), cd_f.get(k)
        if a is None or b is None or b == 0:
            continue
        pairs[k] = {"strict": float(a), "floor": float(b),
                    "rel_diff": float((a - b) / abs(b))}

    worst = max((abs(v["rel_diff"]) for v in pairs.values()), default=None)
    if worst is None:
        decision = "INCONCLUSIVE"
    elif worst > 0.15:
        decision = "STRICT_TOL_RUN_NOT_VALID"
    elif worst <= 0.03:
        decision = "TOLERANCE_COSMETIC_ONLY"
    else:
        decision = "UNRESOLVED_SENSITIVITY_BAND"

    cost = {
        "iterations_ratio": (s_s["iterations_mean"] / s_f["iterations_mean"]
                             if s_f["iterations_mean"] else None),
        "wall_clock_ratio": None,
    }
    ws, wf = strict.get("wall_clock_s"), floor.get("wall_clock_s")
    if ws and wf:
        cost["wall_clock_ratio"] = float(ws / wf)

    return {
        "strict": s_s, "floor": s_f, "channels": channels,
        "coefficients": pairs, "worst_rel_diff": worst,
        "decision": decision, "cost": cost,
    }


def make_figure(strict: dict, floor: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs, hf = strict["history"], floor["history"]
    ts, tf = np.asarray(hs["t"]), np.asarray(hf["t"])
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2))

    a = ax[0, 0]
    a.step(ts, hs["coupling_iterations"], where="mid", label="tol = 1e-3 (was default)")
    a.step(tf, hf["coupling_iterations"], where="mid", label="tol = 1e-2 (measured floor)")
    a.set(xlabel="t [s]", ylabel="sub-iterations", title="Interface iterations per step")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[0, 1]
    rs = np.asarray(hs.get("coupling_residual", []), dtype=float)
    rf = np.asarray(hf.get("coupling_residual", []), dtype=float)
    rs, rf = rs[np.isfinite(rs)], rf[np.isfinite(rf)]
    if rs.size and rf.size:
        a.semilogy(np.arange(rs.size), rs, ".", ms=2, label="tol = 1e-3")
        a.semilogy(np.arange(rf.size), rf, ".", ms=2, label="tol = 1e-2")
        a.axhline(1e-2, color="k", ls="--", lw=.8, label="measured floor ~1e-2")
        a.axhline(1e-3, color="r", ls=":", lw=.8, label="requested 1e-3 (unreachable)")
        a.legend(fontsize=8)
    a.set(xlabel="t [s]", ylabel=r"$\|r_k\|/\|r_0\|$",
          title="Interface residual at step exit"); a.grid(alpha=.3)

    a = ax[1, 0]
    a.plot(ts, hs["body_drag_fit"], lw=.9, label="tol = 1e-3")
    a.plot(tf, hf["body_drag_fit"], lw=.9, label="tol = 1e-2")
    a.set(xlabel="t [s]", ylabel=r"$F_x$ on body [N/m]",
          title="Whole-body drag, plane-fit arm"); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1, 1]
    a.plot(ts, hs["body_lift_fit"], lw=.9, label="tol = 1e-3")
    a.plot(tf, hf["body_lift_fit"], lw=.9, label="tol = 1e-2")
    a.set(xlabel="t [s]", ylabel=r"$F_y$ on body [N/m]",
          title="Whole-body lift, plane-fit arm"); a.legend(fontsize=8); a.grid(alpha=.3)

    fig.suptitle("FSI2 strong coupling: iteration tolerance vs the achievable floor "
                 "(dx = 0.02 m, 200 steps, same dt/IC)", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--out-json", default="docs/xval_fsi2/tol_sensitivity.json")
    ap.add_argument("--out-fig", default="docs/xval_fsi2/assets/fig_tol_sensitivity.png")
    args = ap.parse_args()

    strict = json.loads(Path(args.strict).read_text())
    floor = json.loads(Path(args.floor).read_text())
    res = analyse(strict, floor)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    print(f"{'':22s} {'tol=1e-3':>12s} {'tol=1e-2':>12s}")
    for k in ("iterations_mean", "iterations_at_budget", "converged_fraction",
              "residual_median", "residual_min", "n_steps", "t_end"):
        a, b = res["strict"].get(k), res["floor"].get(k)
        fmt = lambda v: "n/a" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v)
        print(f"{k:22s} {fmt(a):>12s} {fmt(b):>12s}")
    print("\ncoefficient agreement (common window):")
    for k, v in res["coefficients"].items():
        print(f"  {k:14s} strict={v['strict']:+.4f} floor={v['floor']:+.4f} "
              f"rel={100*v['rel_diff']:+.1f}%")
    print("\nchannel agreement:")
    for k, v in res["channels"].items():
        print(f"  {k:14s} rel_rms={100*v['rel_rms_diff']:.1f}%")
    print(f"\nworst_rel_diff={res['worst_rel_diff']}  DECISION={res['decision']}")
    make_figure(strict, floor, Path(args.out_fig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
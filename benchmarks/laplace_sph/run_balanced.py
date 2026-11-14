"""Round 3 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE: well-balanced capillary force.

Round 1 measured the static Young-Laplace jump (``xval_projection_laplace.json``);
round 2 mapped its knob sensitivity (``robustness.json``); the dynamic round
(``results_dynamic.json``) made the time loop stable with the adjoint operator
pair but left the drop alive with spurious currents ``max|u|/v_ref = 0.179``,
*above* Adami et al. (2012)'s ``0.1`` admissibility line, and held it there with
a viscosity knob that is not a modelled fluid property.

This round tests the diagnosis that ``projection_adjoint``'s header already
states: the projection cannot remove the *solenoidal* part of the lattice CSF
force, so that part is re-injected every step.  If the capillary acceleration is
first split, with the projection's own operator, into a discrete gradient plus a
remainder (``helmholtz_split``), then the gradient part is annihilated exactly
and the drop must rest at solver tolerance -- with the Laplace jump read off the
potential ``rho [q]`` (an instrument that never touches the pressure field).

Runs are sharded by resolution (``--level``); ``--merge`` assembles
``results_balanced.json`` and the figure.

    PYTHONPATH=src python benchmarks/laplace_sph/run_balanced.py --level 1
    PYTHONPATH=src python benchmarks/laplace_sph/run_balanced.py --merge --figure
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.laplace_isph import hex_block, csf_state_two_phase
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.projection_adjoint import (helmholtz_split, irrotationality_defect,
                                         relax_projection_balanced)

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "laplace_sph" / "results_balanced"
RESULTS = ROOT / "benchmarks" / "laplace_sph"

R = 0.020
SIGMA = 0.072
RHO = 1000.0
OUTER = 0.012
HF = 1.3
STEPS = 50
SAMPLE = 10
GATES = {
    "laplace_pct": 15.0,      # owner acceptance on the jump
    "laplace_pct_tight": 3.0,  # the round-1/2 claim level
    "adami_spurious": 0.10,   # Adami et al. (2012) admissibility
    "area_drift_pct": 2.0,
}
LEVELS = [
    {"level": 1, "dx": 0.0020, "steps": STEPS, "extra": "dt invariance + no-filter control"},
    {"level": 2, "dx": 0.0015, "steps": STEPS, "extra": "-"},
    {"level": 3, "dx": 0.0010, "steps": STEPS, "extra": "h/dx defect sweep + field map"},
]


def _defect_sweep(dx: float) -> dict:
    """Irrotationality defect of the CSF force vs kernel support, at one dx."""
    pos, mass, liquid, half = hex_block(R, dx, OUTER, RHO)
    vol = mass / RHO
    edge = 1.6 * dx
    pinned = (pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge)
    gauge = int(torch.nonzero(pinned)[0, 0])
    rows = []
    for hf in (1.3, 1.5, 1.7, 1.9):
        h = hf * dx
        pairs = cell_linked_pairs(pos, 2.0 * h)
        st = csf_state_two_phase(pos, mass, liquid, h, RHO, SIGMA,
                                 curvature_method="mls")
        sp = helmholtz_split(pos, st["acc"], vol, pairs, h, gauge=gauge,
                             rel_tol=1e-6, max_iter=6000)
        r = torch.linalg.vector_norm(pos, dim=1)
        core = r < 0.5 * R
        band = (r > 1.25 * R) & (r < 1.45 * R)
        rows.append({
            "h_over_dx": hf,
            "solenoidal_fraction": sp["solenoidal_fraction"],
            "orthogonality": sp["orthogonality"],
            "reconstruction_defect": sp["reconstruction_defect"],
            "dp_potential_Pa": float(RHO * (sp["potential"][core].mean()
                                            - sp["potential"][band].mean())),
            "cg_iterations": sp["cg_iterations"],
        })
        print("  h/dx=%.1f defect=%.4f dp_pot=%.4f"
              % (hf, sp["solenoidal_fraction"], rows[-1]["dp_potential_Pa"]),
              flush=True)
    return {"dx_m": dx, "R_over_dx": R / dx, "rows": rows}


def run_level(idx: int, steps_override=None) -> Path:
    spec = dict(LEVELS[idx - 1])
    if steps_override:
        # smoke runs only: the sharded production command uses the default STEPS
        spec["steps"] = steps_override
    dx = spec["dx"]
    rep = {
        "case": "CASE-LAPLACE-SPH-WELLBALANCED",
        "round": 3,
        "title": "Projection-SPH Young-Laplace: a well-balanced capillary force",
        "setup": {
            "radius_m": R, "dx_m": dx, "R_over_dx": R / dx, "h_over_dx": HF,
            "sigma_N_per_m": SIGMA, "rho0_kg_m3": RHO, "outer_carrier_m": OUTER,
            "steps": spec["steps"], "sample_every": SAMPLE,
            "dt_over_t_sigma": 0.1, "curvature": "mls",
            "cg_rel_tol": 1e-6, "force": "irrotational part of the CSF acceleration",
            "nu_factor": 0.0, "pin_outer": True,
        },
        "gates": GATES,
        "runs": [],
        "controls": [],
    }
    t0 = time.time()
    keep = (idx == 3)
    main = relax_projection_balanced(radius=R, dx=dx, sigma=SIGMA, rho0=RHO,
                                     outer=OUTER, h_factor=HF,
                                     steps=spec["steps"],
                                     sample_every=min(SAMPLE, spec["steps"]),
                                     keep_fields=keep)
    main["wall_s"] = time.time() - t0
    rep["runs"].append(main)
    print("L%d dx=%.4f balanced: umax/vref=%.3e dp=%+.3f%% dp_pot=%+.3f%% area=%.6f (%.0fs)"
          % (idx, dx, main["umax_over_vref_final"], main["dp_rel_error_pct"],
             main["dp_potential_rel_error_pct"], main["area_ratio_final"],
             main["wall_s"]), flush=True)

    if idx == 1:
        # (i) the same settings without the split: the round-3 failure mode,
        #     inviscid, so the honesty of the comparison does not rest on a knob.
        c = relax_projection_balanced(radius=R, dx=dx, sigma=SIGMA, rho0=RHO,
                                      outer=OUTER, h_factor=HF, steps=20,
                                      sample_every=5, balanced=False)
        c["label"] = "no split, inviscid (control)"
        rep["controls"].append(c)
        print("  control no-split: umax/vref=%.3e diverged=%s"
              % (c["umax_over_vref_final"], c["diverged"]), flush=True)
        # (ii) step-size invariance: a true equilibrium must not depend on dt.
        for dtf in (0.05, 0.2):
            c2 = relax_projection_balanced(radius=R, dx=dx, sigma=SIGMA, rho0=RHO,
                                           outer=OUTER, h_factor=HF, steps=20,
                                           sample_every=10, dt_factor=dtf)
            c2["label"] = "dt_factor=%.2f" % dtf
            rep["controls"].append(c2)
            print("  dt=%g: umax/vref=%.3e dp=%+.3f%%"
                  % (dtf, c2["umax_over_vref_final"], c2["dp_rel_error_pct"]),
                  flush=True)
    if idx == 3:
        rep["defect_sweep"] = _defect_sweep(dx)

    out = OUT / ("level%d.json" % idx)
    out.write_text(json.dumps(rep, indent=1))
    return out


def merge(figure: bool) -> Path:
    blob = {
        "case": "CASE-LAPLACE-SPH-WELLBALANCED",
        "round": 3,
        "title": "Projection-SPH Young-Laplace: a well-balanced capillary force",
        "setup": {"radius_m": R, "sigma_N_per_m": SIGMA, "rho0_kg_m3": RHO,
                  "outer_carrier_m": OUTER, "h_over_dx": HF,
                  "dt_over_t_sigma": 0.1, "cg_rel_tol": 1e-6,
                  "v_ref_m_per_s": math.sqrt(SIGMA / (RHO * R)),
                  "dp_theory_Pa": SIGMA / R},
        "gates": GATES,
        "runs": [],
        "controls": [],
        "levels": {},
    }
    for idx, spec in enumerate(LEVELS, start=1):
        f = OUT / ("level%d.json" % idx)
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        blob["runs"].extend(d["runs"])
        blob["controls"].extend(d.get("controls", []))
        blob["levels"][str(idx)] = {"dx_m": spec["dx"], "extra": spec["extra"]}
        if "defect_sweep" in d:
            blob["defect_sweep"] = d["defect_sweep"]
        if d["runs"] and d["runs"][0].get("fields"):
            blob["fields"] = d["runs"][0]["fields"]
            blob["fields"]["dx_m"] = spec["dx"]
    runs = [r for r in blob["runs"] if not r["diverged"]]
    blob["summary"] = {
        "levels_run": len(blob["runs"]),
        "worst_abs_dp_error_pct": max(abs(r["dp_rel_error_pct"]) for r in runs)
        if runs else None,
        "worst_abs_dp_potential_error_pct": max(abs(r["dp_potential_rel_error_pct"])
                                               for r in runs) if runs else None,
        "worst_umax_over_vref": max(r["umax_over_vref_final"] for r in runs)
        if runs else None,
        "best_umax_over_vref": min(r["umax_over_vref_final"] for r in runs)
        if runs else None,
        "worst_area_drift_pct": max(abs(r["area_ratio_final"] - 1.0) * 100.0
                                    for r in runs) if runs else None,
    }
    blob["verdict"] = {
        "laplace_15pct": bool(runs) and abs(blob["summary"]["worst_abs_dp_error_pct"])
        <= GATES["laplace_pct"],
        "laplace_3pct": bool(runs) and abs(blob["summary"]["worst_abs_dp_error_pct"])
        <= GATES["laplace_pct_tight"],
        "adami_spurious": bool(runs) and blob["summary"]["worst_umax_over_vref"]
        < GATES["adami_spurious"],
        "area_drift": bool(runs) and blob["summary"]["worst_area_drift_pct"]
        < GATES["area_drift_pct"],
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "results_balanced.json"
    path.write_text(json.dumps(blob, indent=1))
    if figure:
        make_figure(blob, OUT / "balanced.png")
    return path


def make_figure(blob: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)

    # (a) spurious-current time series, round 3 (damped, gate FAIL) vs round 4
    ax = fig.add_subplot(gs[0, 0])
    dyn = RESULTS / "results_dynamic.json"
    if dyn.exists():
        d = json.loads(dyn.read_text())
        best = d.get("best") or {}
        if best.get("history"):
            t = [h["t_over_t_sigma"] for h in best["history"]]
            y = [h["umax_over_vref"] for h in best["history"]]
            ax.semilogy(t, y, "o-", color="tab:red", lw=1.4,
                        label="round 3 damped (nu knob)")
    for r in blob["runs"]:
        t = [h["t_over_t_sigma"] for h in r["history"] if "umax_over_vref" in h]
        y = [h["umax_over_vref"] for h in r["history"] if "umax_over_vref" in h]
        ax.semilogy(t, y, "-", lw=1.5,
                    label="balanced, R/dx=%.0f" % r["R_over_dx"])
    ax.axhline(GATES["adami_spurious"], color="k", ls="--", lw=1.0,
               label="Adami 2012 gate 0.1")
    ax.set_xlabel(r"$t/t_\sigma$")
    ax.set_ylabel(r"$\max|u|/v_{\rm ref}$")
    ax.set_title("(a) spurious currents")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (b) Laplace jump error, both instruments
    ax = fig.add_subplot(gs[0, 1])
    rd = [r["R_over_dx"] for r in blob["runs"]]
    ax.plot(rd, [r["dp_rel_error_pct"] for r in blob["runs"]], "o-",
            label=r"pressure solve $\Delta p$")
    ax.plot(rd, [r["dp_potential_rel_error_pct"] for r in blob["runs"]], "s--",
            label=r"potential $\rho[q]$")
    ax.axhline(3.0, color="tab:green", ls=":", label="3 %")
    ax.axhline(-3.0, color="tab:green", ls=":")
    ax.axhline(15.0, color="tab:red", ls=":", label="15 % acceptance")
    ax.axhline(-15.0, color="tab:red", ls=":")
    ax.set_xlabel(r"$R/\Delta x$")
    ax.set_ylabel("relative error / %")
    ax.set_title(r"(b) $\Delta p$ vs $\sigma/R$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) irrotationality defect vs kernel support
    ax = fig.add_subplot(gs[0, 2])
    sw = blob.get("defect_sweep")
    if sw:
        hf = [r["h_over_dx"] for r in sw["rows"]]
        ax.plot(hf, [r["solenoidal_fraction"] for r in sw["rows"]], "d-",
                label="discarded fraction")
        ax.set_xlabel(r"$h/\Delta x$")
        ax.set_ylabel(r"$\|a_{\rm sol}\|/\|a_{\rm csf}\|$")
        ax.set_title("(c) force the projection cannot see")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # (d)+(e) final fields
    fl = blob.get("fields")
    if fl:
        import numpy as np
        pos = np.asarray(fl["pos"])
        sp = np.asarray(fl["speed_over_vref"])
        pr = np.asarray(fl["pressure"])
        for k, (val, ttl, cmap, lab) in enumerate([
                (np.log10(np.maximum(sp, 1e-12)), r"(d) $\log_{10}\max|u|/v_{\rm ref}$",
                 "magma", "log10"),
                (pr, "(e) pressure / Pa", "coolwarm", "p")]):
            ax = fig.add_subplot(gs[1, k])
            sc = ax.scatter(pos[:, 0] * 1e2, pos[:, 1] * 1e2, c=val, s=3, cmap=cmap)
            fig.colorbar(sc, ax=ax, shrink=0.85)
            ax.set_aspect("equal")
            ax.set_xlabel("x / cm")
            ax.set_ylabel("y / cm")
            ax.set_title(ttl, fontsize=9)
        # (f) radial pressure profile vs analytic step
        ax = fig.add_subplot(gs[1, 2])
        r = np.linalg.norm(pos, axis=1)
        order = np.argsort(r)
        ax.plot(r[order] * 1e2, pr[order], ".", ms=2, label="SPH pressure")
        ax.axhline(SIGMA / R, color="tab:red", ls="--", label=r"analytic $\sigma/R$")
        ax.axvline(R * 1e2, color="k", ls=":", lw=1, label="R")
        ax.set_xlabel("r / cm")
        ax.set_ylabel("p / Pa")
        ax.set_title("(f) radial pressure, R/dx=%d" % int(R / fl["dx_m"]), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Projection-SPH Young-Laplace with a well-balanced capillary force "
                 "(round 3)", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("figure ->", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--steps", type=int, default=None,
                    help="override the per-level step count (smoke runs)")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()
    global OUT
    if args.outdir:
        OUT = Path(args.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.level:
        run_level(args.level, args.steps)
    if args.merge:
        print("merged ->", merge(args.figure))


if __name__ == "__main__":
    main()
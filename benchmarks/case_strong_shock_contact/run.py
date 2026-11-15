#!/usr/bin/env python
"""CASE-SPH-STRONG-SHOCK-CONTACT — pool entry (TensorSPH).

Second benchmark family for :mod:`tensorsph.compressible` (the Sod shock tube is
the first).  The run is a gamma-law Riemann problem with **stationary outer
states** and a **1000:1 pressure ratio**:

    (rho, u, p) = (1.0, 0.0, 1000.0)  for x < 0
    (rho, u, p) = (1.0, 0.0,    1.0)  for x > 0        x in [-1, 1], t_end = 0.02

which produces a left rarefaction fan (density drop 1.0 -> 0.576), a pressure
plateau p* = 461.5, a contact discontinuity with a 10.3x density jump
(0.576 -> 5.925) and a strong right shock (shock Mach ~ 4.6).  Sod's tube is a
10:1 driver with a 1.3-Mach shock and a 3.4x contact jump, so this family is a
different regime, not a re-run: it is the regime where the pressure plateau has
to survive a large contact jump -- exactly the quantity that the Sod family does
not test.

Why *stationary* outer states (and not Toro's tests 2-5)
-------------------------------------------------------
The 1-D family in :mod:`tensorsph.compressible.sod` is a **Lagrangian** box with
reflecting (mirror-ghost) walls and a fixed particle mass.  A Riemann problem
whose outer states move towards/away from the boundaries has mass crossing the
box boundary in the Eulerian exact solution; a fixed-mass box cannot host it.
:func:`tensorsph.compressible.riemann_topology.reflective_box_budget` decides
this from the exact solution alone, and this driver *refuses to grade* a
configuration it declares non-representable (measured for Toro's test 4: net
boundary influx 5.41 vs a box mass of 6.00, i.e. 90 %, and the shipped solver's
L1(rho) on that problem is 39 % -- a boundary-condition artefact, not a scheme
error).  See ``--probe-nonrepresentable``.

Usage
-----
    python benchmarks/case_strong_shock_contact/run.py            # grade + write result.json
    python benchmarks/case_strong_shock_contact/run.py --figures  # also write the report figures
    python benchmarks/case_strong_shock_contact/run.py --no-write

Reference: E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
Dynamics*, 2nd ed., Springer (1999), ch. 4 (exact solver + wave topology);
shock/contact grading follows the conventions of the Sod family
(``docs/validation/compressible_sod/REPORT.md``).  The exact yardstick is
cross-checked against the independent HLLC finite-volume solver
:mod:`tensorsph.compressible.fv_reference`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorsph.compressible import fv_reference as FV  # noqa: E402
from tensorsph.compressible import riemann_topology as RT  # noqa: E402
from tensorsph.compressible import sod as S  # noqa: E402

CASE = "CASE-SPH-STRONG-SHOCK-CONTACT"
REFERENCE = ("Toro 1999, Riemann Solvers and Numerical Methods for Fluid Dynamics, "
             "ch. 4 (exact solver); Sod 1978 JCP 27(1):1-31 for the grading convention")
STATES = dict(gamma=1.4, rho_L=1.0, p_L=1000.0, u_L=0.0,
              rho_R=1.0, p_R=1.0, u_R=0.0)
DOMAIN = dict(x_left=-1.0, x_right=1.0, x_diaphragm=0.0, t_end=0.02)
#: Production resolution of the record (same N and h/dx as the Sod family, so the
#: two families are quoted at the same cost) and the convergence ladder.
N_PRIMARY = 800
LADDER = (400, 800, 1600)
H_FACTOR = 2.5
CFL = 0.15
GATE_PCT = 5.0
OWNER_TOLERANCE_PCT = 15.0
#: resolutions used by the independent yardstick cross-check (HLLC, first order)
FV_CELLS = 8000

FIG_DIR = os.path.join(ROOT, "docs", "figures", "strong_shock_contact")


def build_config(n: int, t_end: float = None) -> "S.SodSPHConfig":
    """The predeclared configuration; every run takes its knobs from here."""
    return S.SodSPHConfig(
        n_particles=n, h_factor=H_FACTOR, cfl=CFL, t_end=DOMAIN["t_end"] if t_end is None else t_end,
        x_left=DOMAIN["x_left"], x_right=DOMAIN["x_right"],
        x_diaphragm=DOMAIN["x_diaphragm"], **STATES)


def topology() -> "RT.WaveTopology":
    return RT.topology(STATES["gamma"], STATES["rho_L"], STATES["p_L"], STATES["u_L"],
                       STATES["rho_R"], STATES["p_R"], STATES["u_R"])


def run_resolution(n: int) -> dict:
    """Run the SPH solver at resolution ``n`` and grade it against the exact solution."""
    cfg = build_config(n)
    res = S.run_sod_sph(cfg)
    snap = res["snapshots"][-1]
    grade = RT.grade_riemann(snap["x"], snap["rho"], snap["u"], snap["p"], topology(),
                             snap["t"], cfg.x_diaphragm, cfg.h, cfg.x_left, cfg.x_right)
    return {
        "n_particles": n,
        "h": cfg.h,
        "dx": cfg.dx,
        "steps": res["steps"],
        "t_final": res["t_final"],
        "mass_rel_change": (res["mass_final"] - res["mass_initial"]) / res["mass_initial"],
        "energy_rel_change": res["energy_rel_change"],
        "grading": grade,
        "profiles": {k: np.asarray(snap[k], dtype=float).tolist()
                     for k in ("x", "rho", "u", "p")},
    }


def yardstick_cross_check() -> dict:
    """Independent route to the same exact solution: first-order HLLC finite volume.

    The analytic profile is the yardstick for *every* error number of this case;
    the HLLC solver shares nothing with it (different discretisation, different
    code path), so agreement is evidence that the yardstick itself is right.
    """
    topo = topology()
    fv = FV.run_hllc(n_cells=FV_CELLS, t_end=DOMAIN["t_end"], cfl=0.4,
                     gamma=STATES["gamma"], x_left=DOMAIN["x_left"],
                     x_right=DOMAIN["x_right"], x_diaphragm=DOMAIN["x_diaphragm"],
                     rho_L=STATES["rho_L"], p_L=STATES["p_L"], u_L=STATES["u_L"],
                     rho_R=STATES["rho_R"], p_R=STATES["p_R"], u_R=STATES["u_R"])
    snap = {"x": np.asarray(fv["x"]), "rho": np.asarray(fv["rho"]),
            "u": np.asarray(fv["u"]), "p": np.asarray(fv["p"])}
    grade = RT.grade_riemann(snap["x"], snap["rho"], snap["u"], snap["p"], topo,
                             float(fv["t"]), DOMAIN["x_diaphragm"],
                             (DOMAIN["x_right"] - DOMAIN["x_left"]) / FV_CELLS,
                             DOMAIN["x_left"], DOMAIN["x_right"])
    return {
        "solver": "first-order HLLC finite volume (src/tensorsph/compressible/fv_reference.py)",
        "n_cells": FV_CELLS,
        "steps": int(fv["steps"]),
        "t_final": float(fv["t"]),
        "max_abs_rel_err_gated": grade["max_abs_rel_err"],
        "rel_err": grade["rel_err"],
        "l1": grade["l1"],
        "note": ("the contact is smeared by the first-order flux (L1(rho) is dominated "
                 "by that smearing); the plateau *values* and the wave positions are "
                 "what confirm the yardstick"),
    }


def nonrepresentable_probe() -> dict:
    """Toro's test 4 (two strong shocks, inflow at both boundaries) -- refused, with evidence.

    Recorded so that the refusal is a measured statement rather than an opinion:
    the representability budget is 90 %, and running it anyway shows the failure
    signature (the star pressure collapses; the mass that the Eulerian solution
    draws in through the boundaries is simply absent).
    """
    st = dict(gamma=1.4, rho_L=5.99924, p_L=460.8938, u_L=19.5975,
              rho_R=5.99242, p_R=46.0950, u_R=-6.19633)
    topo = RT.topology(st["gamma"], st["rho_L"], st["p_L"], st["u_L"],
                       st["rho_R"], st["p_R"], st["u_R"])
    t_end, x0 = 0.035, 0.4
    budget = RT.reflective_box_budget(topo, t_end, x0, 0.0, 1.0)
    cfg = S.SodSPHConfig(n_particles=800, h_factor=H_FACTOR, cfl=CFL, t_end=t_end,
                         x_left=0.0, x_right=1.0, x_diaphragm=x0,
                         **{k: v for k, v in st.items() if k != "gamma"},
                         gamma=st["gamma"])
    res = S.run_sod_sph(cfg)
    snap = res["snapshots"][-1]
    grade = RT.grade_riemann(snap["x"], snap["rho"], snap["u"], snap["p"], topo,
                             snap["t"], x0, cfg.h, cfg.x_left, cfg.x_right)
    return {
        "problem": "Toro (1999) test 4: two strong shocks, contact nearly stationary",
        "states": st,
        "t_end": t_end,
        "representability": budget,
        "graded_anyway": {"l1": grade["l1"],
                          "star_pressure_rel_err": grade["rel_err"]["p_star"],
                          "max_abs_rel_err_gated": grade["max_abs_rel_err"]},
        "verdict": ("NOT a valid measurement of the scheme: the exact solution holds "
                    "the box mass plus the inflow, the Lagrangian box cannot.  This is "
                    "why the second family uses stationary outer states."),
    }


def make_figures(result: dict) -> list:
    """Field comparisons + convergence + contact-spike zoom (report figures)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    paths = []
    prim = result["primary"]
    topo = topology()
    x = np.asarray(prim["profiles"]["x"])
    t = prim["grading"]["t"]
    from tensorsph.wcsph.sod_shock import exact_riemann_profile
    order = np.argsort(x)
    x = x[order]
    rho = np.asarray(prim["profiles"]["rho"])[order]
    u = np.asarray(prim["profiles"]["u"])[order]
    p = np.asarray(prim["profiles"]["p"])[order]
    rho_e, u_e, p_e = exact_riemann_profile(
        x, t, gamma=STATES["gamma"], rho_L=STATES["rho_L"], p_L=STATES["p_L"],
        u_L=STATES["u_L"], rho_R=STATES["rho_R"], p_R=STATES["p_R"],
        u_R=STATES["u_R"], x0=DOMAIN["x_diaphragm"])

    # 1) three fields vs the exact solution
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, got, ref, lab, ylab in (
            (axes[0], rho, rho_e, "SPH", r"$\rho$"),
            (axes[1], u, u_e, "SPH", r"$u$ [m/s]"),
            (axes[2], p, p_e, "SPH", r"$p$")):
        ax.plot(x, ref, "k-", lw=1.2, label="exact (Toro ch. 4)")
        ax.plot(x, got, "C0.", ms=2.4, label=f"{lab} N={prim['n_particles']}")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(f"{CASE}: pressure ratio 1000:1, stationary outer states, "
                      f"t = {t:.3f} s, N = {prim['n_particles']}")
    axes[-1].set_xlabel("x [m]")
    path = os.path.join(FIG_DIR, "profiles.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    # 2) contact-zone zoom (pressure plateau + spurious spike)
    pos = prim["grading"]["exact_wave_positions"]
    lo = pos["x_rarefaction_tail"]
    hi = pos["x_wave_right"] + 0.05
    sel = (x >= lo) & (x <= hi)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x[sel], p_e[sel], "k-", lw=1.2, label="exact")
    ax.plot(x[sel], p[sel], "C3.-", ms=3, label="SPH")
    ax.axhline(topo.p_star, ls="--", c="gray", lw=0.8,
               label=f"exact plateau $p^*$ = {topo.p_star:.1f}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("p")
    ax.set_title("star region: pressure plateau and the contact-zone spike "
                 f"(spike = {prim['grading']['star_envelope_pressure_spike_over_p_star']:.2f} $p^*$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(FIG_DIR, "star_region.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    # 3) convergence of the gated quantities
    gated = result["gated_quantities"]
    ns = [row["n_particles"] for row in result["resolutions"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in gated:
        ax.loglog(ns, [abs(row["grading"]["rel_err"][key]) * 100 for row in result["resolutions"]],
                  "o-", label=key)
    ax.axhline(GATE_PCT, ls="--", c="r", lw=1, label=f"own gate {GATE_PCT:g} %")
    ax.axhline(OWNER_TOLERANCE_PCT, ls=":", c="k", lw=1, label="owner tolerance 15 %")
    ax.set_xlabel("N particles")
    ax.set_ylabel("relative error [%]")
    ax.set_title("gated quantities vs resolution")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    path = os.path.join(FIG_DIR, "convergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--probe-nonrepresentable", action="store_true",
                    help="also record the refused Toro test 4 configuration")
    args = ap.parse_args(argv)

    topo = topology()
    budget = RT.reflective_box_budget(topo, DOMAIN["t_end"], DOMAIN["x_diaphragm"],
                                      DOMAIN["x_left"], DOMAIN["x_right"])
    if not budget["representable"]:
        print(json.dumps(budget, indent=2))
        raise SystemExit("configuration is not representable in a reflecting box")

    resolutions = [run_resolution(n) for n in LADDER]
    primary = next(r for r in resolutions if r["n_particles"] == N_PRIMARY)

    result = {
        "case": CASE,
        "reference": REFERENCE,
        "states": STATES,
        "domain": DOMAIN,
        "production": {"n_particles": N_PRIMARY, "h_factor": H_FACTOR, "cfl": CFL,
                       "gamma": STATES["gamma"], "convergence_ladder": list(LADDER)},
        "topology": topo.as_dict(),
        "representability": budget,
        "gated_quantities": list(RT.GATED_QUANTITIES),
        "diagnostic_quantities": list(RT.DIAGNOSTIC_QUANTITIES),
        "gate_pct": GATE_PCT,
        "owner_tolerance_pct": OWNER_TOLERANCE_PCT,
        "resolutions": resolutions,
        "primary": primary,
        "yardstick_cross_check": yardstick_cross_check(),
        "pass_gate": bool(primary["grading"]["max_abs_rel_err"] * 100.0 < GATE_PCT),
        "pass_owner_tolerance": bool(primary["grading"]["max_abs_rel_err"] * 100.0
                                     < OWNER_TOLERANCE_PCT),
    }
    if args.probe_nonrepresentable:
        result["nonrepresentable_probe"] = nonrepresentable_probe()

    g = primary["grading"]
    print(f"{CASE}  N={N_PRIMARY}  steps={primary['steps']}  t={primary['t_final']:.5f}")
    print(f"{'quantity':20s} {'got':>12s} {'exact':>12s} {'rel.err':>9s}")
    for key in RT.GATED_QUANTITIES:
        print(f"{key:20s} {g['got'][key]:12.5f} {g['exact'][key]:12.5f} "
              f"{g['rel_err'][key] * 100:8.3f}%")
    print(f"max |rel err| (gated) = {g['max_abs_rel_err'] * 100:.3f} %  "
          f"gate {GATE_PCT:g} %: {result['pass_gate']}  "
          f"owner 15 %: {result['pass_owner_tolerance']}")
    print(f"L1(rho,u,p) = {g['l1']['rho'] * 100:.2f} %, {g['l1']['u'] * 100:.2f} %, "
          f"{g['l1']['p'] * 100:.2f} %   "
          f"contact-zone pressure spike = {g['star_envelope_pressure_spike_over_p_star']:.2f} p*")
    print("diagnostics (not gated): " + ", ".join(
        f"{k} {v * 100:+.1f}%" for k, v in g["diagnostic_rel_err"].items()))
    yc = result["yardstick_cross_check"]
    print(f"yardstick cross-check (HLLC {FV_CELLS} cells): max gated |rel| = "
          f"{yc['max_abs_rel_err_gated'] * 100:.3f} %")

    if args.figures:
        for path in make_figures(result):
            print("wrote", os.path.relpath(path, ROOT))

    if not args.no_write:
        path = os.path.join(HERE, "result.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=1)
        print("wrote", os.path.relpath(path, ROOT))
    return 0 if result["pass_gate"] else 1


if __name__ == "__main__":
    sys.exit(main())
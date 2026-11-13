"""Round 2 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE — the error budget of the claim.

Round 1 (``docs/xval_projection_laplace_oss.md``, m17594) reported
``dp = sigma/R`` to **-0.21 %** for this repo's projection-SPH route at
``R/dx = 20``, and cross-validated it against the PySPH operators.  Every one of
those numbers was taken with a *single, fixed* experimental set-up:

    interface centred on a lattice site, ``outer = 12 dx`` of carrier,
    ``h = 1.3 dx``, jump read as ``<p>_(r<0.5R) - <p>_(1.25R<r<1.45R)``.

A static droplet is one physical solution, and the Young-Laplace law contains
``sigma`` and ``R`` only; nothing in the physics selects a lattice alignment,
a container size or a sampling band.  So a single-set-up number cannot separate
*method accuracy* from *lattice luck*: if the answer moves by 10 % when the
interface is translated by ``dx/2``, the honest error bar of the claim is 10 %,
not 0.2 %.

This script measures that budget directly.  Each axis varies **one** thing and
holds the rest at the round-1 baseline, and every axis is judged by a gate that
was declared before the numbers were inspected (see ``GATES``):

======  ===========================================================  =========
axis    what is varied                                                gate
======  ===========================================================  =========
A       interface position inside the lattice cell (12 offsets)       15 %
B       lattice orientation vs interface (0/5/10/15/20/30 deg)       15 %
C       Dirichlet-gauge distance ``outer/dx`` (2.5 .. 12)             5 %
D       estimator: core fraction x outer band (3 x 4)                 5 %
E       kernel support ``h/dx`` (1.15 .. 1.6)                         5 %
F       resolution ``R/dx`` = 10 .. 60 + observed order              5 % @>=20
======  ===========================================================  =========

Two independent instruments are carried through every row, because the jump can
be read without ever touching the pressure field:

* ``dp_pressure`` -- ``<p>_core - <p>_band`` from the projection solve;
* ``dp_flux``     -- ``-int rho a_r dA / (2 pi R)``, the radial integral of the
  CSF traction alone (the Laplace law *is* the integral of the force across the
  interface, so this route is the model's own momentum budget).

If the two disagree, or if either depends on a knob that the physics cannot see,
the round-1 number was not a property of the method.

Run::

    PYTHONPATH=src python3 scripts/xval_projection_laplace/robustness.py --quick
    PYTHONPATH=src python3 scripts/xval_projection_laplace/robustness.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.laplace_isph import laplace_isph_case

torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

RADIUS = 0.02
SIGMA = 0.072
RHO0 = 1000.0
OUTER = 0.012           # the round-1 baseline carrier thickness
DX_BASE = 0.001         # R/dx = 20, the round-1 finest level
CURVATURE = "mls"       # the shipped route used by round 1

# Declared before the numbers were inspected (owner gate = 15 %).
GATES = {
    "single_config_abs_err_pct": 15.0,   # every row of every axis
    "translation_spread_pct": 15.0,      # physics: a static drop is translation invariant
    "orientation_spread_pct": 15.0,
    "gauge_spread_pct": 5.0,             # a gauge must not move the answer
    "estimator_spread_pct": 5.0,
    "h_factor_spread_pct": 5.0,
    "resolution_abs_err_pct_at_20": 5.0,
}


def _row(**kw):
    """One equilibrium, reduced to the numbers the audit judges."""
    kw.setdefault("radius", RADIUS)
    kw.setdefault("dx", DX_BASE)
    kw.setdefault("outer", OUTER)
    kw.setdefault("curvature_method", CURVATURE)
    t0 = time.time()
    rep = laplace_isph_case(**kw)
    return {
        "dp_pressure_Pa": rep["dp_measured_Pa"],
        "dp_pressure_err_pct": rep["dp_error_pct"],
        "dp_flux_Pa": rep["dp_flux_Pa"],
        "dp_flux_err_pct": (rep["dp_flux_over_theory"] - 1.0) * 100.0,
        "kappaR": rep["kappaR_delta_weighted"],
        "delta_integral_dn": rep["delta_s_integral_dn"],
        "residual_median_over_mean_acc": rep["residual_acc_median_over_mean_csf_acc"],
        "n_particles": rep["n_particles"],
        "cg_iterations": rep["cg_iterations"],
        "R_over_dx": rep["R_over_dx"],
        "h_over_dx": rep["h_m"] / rep["dx_m"],
        "center_m": rep["center_m"],
        "lattice_angle_deg": rep["lattice_angle_deg"],
        "wall_s": time.time() - t0,
    }


def _spread(rows, key="dp_pressure_err_pct"):
    vals = [r[key] for r in rows]
    return {"min_pct": min(vals), "max_pct": max(vals),
            "spread_pct": max(vals) - min(vals),
            "mean_pct": sum(vals) / len(vals),
            "max_abs_pct": max(abs(v) for v in vals)}


def _axis(name, vary, rows):
    return {"axis": name, "varied": vary, "rows": rows, "spread": _spread(rows),
            "spread_flux": _spread(rows, "dp_flux_err_pct")}


# --------------------------------------------------------------------------- #
# axes
# --------------------------------------------------------------------------- #
def axis_translation(dx=DX_BASE, fractions=None):
    """A: move the interface inside the lattice cell (hex: a1=(dx,0), a2=(dx/2,dy))."""
    dy = dx * math.sqrt(3.0) / 2.0
    if fractions is None:
        fractions = [(0, 0), (1 / 3, 0), (1 / 2, 0), (0, 1 / 3), (0, 1 / 2),
                     (1 / 4, 1 / 4), (1 / 3, 1 / 3), (1 / 2, 1 / 2),
                     (2 / 3, 0), (0, 2 / 3), (1 / 3, 1 / 2), (1 / 2, 1 / 3)]
    rows = []
    for fx, fy in fractions:
        r = _row(dx=dx, center=(fx * dx, fy * dy))
        r["offset_x_over_dx"] = fx
        r["offset_y_over_dy"] = fy
        rows.append(r)
    return _axis("translation", "interface offset inside the lattice cell", rows)


def axis_orientation(dx=DX_BASE, degrees=None):
    """B: rotate the lattice relative to the interface (hexagonal period = 60 deg)."""
    if degrees is None:
        degrees = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    rows = [_row(dx=dx, angle=math.radians(d)) for d in degrees]
    return _axis("orientation", "lattice basis rotation vs the interface", rows)


def axis_gauge(dx=DX_BASE, outer_over_dx=None):
    """C: distance of the Dirichlet (ambient) gauge from the interface."""
    if outer_over_dx is None:
        outer_over_dx = [2.5, 4.0, 6.0, 9.0, 12.0]
    rows = []
    for f in outer_over_dx:
        r = _row(dx=dx, outer=f * dx)
        r["outer_over_dx"] = f
        rows.append(r)
    return _axis("gauge", "Dirichlet gauge distance outer/dx", rows)


def axis_estimator(dx=DX_BASE, core_fracs=None, bands=None):
    """D: how the jump is read out of the pressure field."""
    if core_fracs is None:
        core_fracs = (0.3, 0.5, 0.7)
    if bands is None:
        bands = ((1.15, 1.35), (1.25, 1.45), (1.40, 1.60), (1.55, 1.75))
    rows = []
    for cf, band in itertools.product(core_fracs, bands):
        r = _row(dx=dx, core_frac=cf, out_band=band)
        r["core_frac"] = cf
        r["out_band_over_R"] = list(band)
        rows.append(r)
    return _axis("estimator", "core fraction x outer sampling band", rows)


def axis_h_factor(dx=DX_BASE, factors=None):
    """E: kernel support, ``h/dx`` (a discretisation knob, not a physical one)."""
    if factors is None:
        factors = [1.15, 1.3, 1.45, 1.6]
    rows = []
    for hf in factors:
        r = _row(dx=dx, h_factor=hf)
        rows.append(r)
    return _axis("h_factor", "kernel support h/dx", rows)


def axis_resolution(dx_list=None, offsets=((0.0, 0.0), (0.5, 0.5))):
    """F: resolution ladder, each level both on-lattice and worst-case offset.

    The second sub-arm answers the question a ladder alone cannot: does the
    closed-loop error shrink with refinement, or is it a floor set by the
    lattice alignment?
    """
    if dx_list is None:
        dx_list = [0.002, 0.0013333333, 0.001, 0.0006666667, 0.0005, 0.0003333333]
    rows = []
    for dx in dx_list:
        dy = dx * math.sqrt(3.0) / 2.0
        for fx, fy in offsets:
            r = _row(dx=dx, center=(fx * dx, fy * dy))
            r["offset_x_over_dx"] = fx
            r["offset_y_over_dy"] = fy
            rows.append(r)
    return _axis("resolution", "R/dx (and lattice offset at each level)", rows)


def observed_order(rows, key="dp_pressure_err_pct", min_r=15.0):
    """Least-squares slope of log|err| vs log(R/dx) over the refined part."""
    pts = [(math.log(r["R_over_dx"]), math.log(abs(r[key]) / 100.0))
           for r in rows if r["R_over_dx"] >= min_r]
    if len(pts) < 2:
        return None
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    if abs(den) < 1e-14:
        return None
    return {"slope": (n * sxy - sx * sy) / den, "n_points": n,
            "note": ("slope <= 0 means the error does not decay inside this "
                     "range (measurement floor, not discretisation convergence)")}


# --------------------------------------------------------------------------- #
# audit driver
# --------------------------------------------------------------------------- #
def run_audit(quick: bool = False):
    t0 = time.time()
    payload = {
        "case": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE round 2 (error budget)",
        "reference": "2-D Young-Laplace dp = sigma/R (circle: 1/R and 0)",
        "params": {"radius_m": RADIUS, "sigma_N_per_m": SIGMA,
                   "rho0_kg_per_m3": RHO0, "dx_base_m": DX_BASE,
                   "outer_m": OUTER, "curvature_method": CURVATURE,
                   "dp_theory_Pa": SIGMA / RADIUS},
        "gates": GATES,
        "torch": torch.__version__,
        "axes": {},
    }
    if quick:
        payload["axes"]["translation"] = axis_translation(
            fractions=[(0, 0), (1 / 2, 0), (0, 1 / 2), (1 / 2, 1 / 2)])
        payload["axes"]["orientation"] = axis_orientation(degrees=[0.0, 15.0, 30.0])
        payload["axes"]["gauge"] = axis_gauge(outer_over_dx=[2.5, 6.0, 12.0])
        payload["axes"]["estimator"] = axis_estimator(
            core_fracs=(0.5,), bands=((1.15, 1.35), (1.25, 1.45)))
        payload["axes"]["h_factor"] = axis_h_factor(factors=[1.15, 1.3])
        payload["axes"]["resolution"] = axis_resolution(dx_list=[0.002, 0.001])
    else:
        payload["axes"]["translation"] = axis_translation()
        payload["axes"]["orientation"] = axis_orientation()
        payload["axes"]["gauge"] = axis_gauge()
        payload["axes"]["estimator"] = axis_estimator()
        payload["axes"]["h_factor"] = axis_h_factor()
        payload["axes"]["resolution"] = axis_resolution()

    res = payload["axes"]["resolution"]["rows"]
    on_lattice = [r for r in res if r["offset_x_over_dx"] == 0.0]
    off_lattice = [r for r in res if r["offset_x_over_dx"] != 0.0]
    payload["observed_order_on_lattice"] = observed_order(on_lattice)
    payload["observed_order_off_lattice"] = observed_order(off_lattice)
    payload["observed_order_flux"] = observed_order(on_lattice, key="dp_flux_err_pct")

    all_rows = [r for ax in payload["axes"].values() for r in ax["rows"]]
    worst = max(all_rows, key=lambda r: abs(r["dp_pressure_err_pct"]))
    payload["summary"] = {
        "n_equilibria": len(all_rows),
        "max_abs_err_pct": abs(worst["dp_pressure_err_pct"]),
        "max_abs_err_where": {"center_m": worst["center_m"],
                              "lattice_angle_deg": worst["lattice_angle_deg"],
                              "R_over_dx": worst["R_over_dx"]},
        "max_abs_flux_err_pct": max(abs(r["dp_flux_err_pct"]) for r in all_rows),
        "gate_check": {
            "single_config": abs(worst["dp_pressure_err_pct"])
            <= GATES["single_config_abs_err_pct"],
            "translation_spread": payload["axes"]["translation"]["spread"]["spread_pct"]
            <= GATES["translation_spread_pct"],
            "orientation_spread": payload["axes"]["orientation"]["spread"]["spread_pct"]
            <= GATES["orientation_spread_pct"],
            "gauge_spread": payload["axes"]["gauge"]["spread"]["spread_pct"]
            <= GATES["gauge_spread_pct"],
            "estimator_spread": payload["axes"]["estimator"]["spread"]["spread_pct"]
            <= GATES["estimator_spread_pct"],
            "h_factor_spread": payload["axes"]["h_factor"]["spread"]["spread_pct"]
            <= GATES["h_factor_spread_pct"],
            "resolution_at_20": all(
                abs(r["dp_pressure_err_pct"]) <= GATES["resolution_abs_err_pct_at_20"]
                for r in res if r["R_over_dx"] >= 20.0),
        },
        "wall_s": time.time() - t0,
    }
    return payload


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax1 = payload["axes"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))

    a = axes[0, 0]
    tr = ax1["translation"]["rows"]
    a.plot([r["offset_x_over_dx"] for r in tr],
           [r["dp_pressure_err_pct"] for r in tr], "o-", label="pressure jump")
    a.plot([r["offset_x_over_dx"] for r in tr],
           [r["dp_flux_err_pct"] for r in tr], "s--", label="traction flux")
    a.axhspan(-15, 15, color="0.9", zorder=0, label="+-15% gate")
    a.set_xlabel("interface offset inside the cell  [dx]")
    a.set_ylabel("error vs sigma/R  [%]")
    a.set_title("A: lattice translation")
    a.legend(fontsize=7)

    b = axes[0, 1]
    for key, label, style in (("gauge", "C: gauge distance outer/dx", "o-"),
                              ("h_factor", "E: kernel support h/dx", "s-"),
                              ("orientation", "B: lattice angle [deg]", "^--")):
        rows = ax1[key]["rows"]
        xs = ([r["outer_over_dx"] for r in rows] if key == "gauge"
              else [r["h_over_dx"] for r in rows] if key == "h_factor"
              else [r["lattice_angle_deg"] for r in rows])
        b.plot(xs, [r["dp_pressure_err_pct"] for r in rows], style, label=label)
    b.axhspan(-15, 15, color="0.9", zorder=0)
    b.set_xscale("log")
    b.set_xlabel("knob value")
    b.set_ylabel("error vs sigma/R  [%]")
    b.set_title("B/C/E: knobs the physics cannot see")
    b.legend(fontsize=7)

    c = axes[1, 0]
    rows = ax1["estimator"]["rows"]
    core = sorted({r["core_frac"] for r in rows})
    bands = sorted({tuple(r["out_band_over_R"]) for r in rows})
    grid = [[None] * len(bands) for _ in core]
    for r in rows:
        grid[core.index(r["core_frac"])][bands.index(tuple(r["out_band_over_R"]))] = \
            r["dp_pressure_err_pct"]
    im = c.imshow(grid, cmap="coolwarm", vmin=-8, vmax=8, aspect="auto")
    c.set_xticks(range(len(bands)))
    c.set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in bands], fontsize=8)
    c.set_yticks(range(len(core)))
    c.set_yticklabels([f"{v:g}R" for v in core])
    c.set_xlabel("outer sampling band  [x R]")
    c.set_ylabel("core radius")
    c.set_title("D: how the jump is read out  [% error]")
    for i in range(len(core)):
        for j in range(len(bands)):
            if grid[i][j] is not None:
                c.text(j, i, f"{grid[i][j]:+.2f}", ha="center", va="center",
                       fontsize=8)
    fig.colorbar(im, ax=c, fraction=0.046)

    d = axes[1, 1]
    res = ax1["resolution"]["rows"]
    for fx, label, style in ((0.0, "on-lattice", "o-"),
                             (0.5, "offset dx/2,dy/2", "s--")):
        rows = [r for r in res if r["offset_x_over_dx"] == fx]
        d.plot([r["R_over_dx"] for r in rows],
               [abs(r["dp_pressure_err_pct"]) for r in rows], style, label=label)
    d.axhline(15, color="0.4", lw=1.0, label="15% gate")
    d.axhline(3, color="0.4", lw=0.8, ls=":", label="3% line")
    d.set_xscale("log")
    d.set_yscale("log")
    d.set_xlabel("R/dx")
    d.set_ylabel("|error| vs sigma/R  [%]")
    d.set_title("F: resolution ladder (two lattice alignments)")
    d.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = run_audit(quick=args.quick)
    stem = "robustness_quick" if args.quick else "robustness"
    png = make_figure(payload, OUT / f"{stem}.png")
    payload["figure"] = png
    path = OUT / f"{stem}.json"
    path.write_text(json.dumps(payload, indent=2))
    s = payload["summary"]
    print(f"equilibria: {s['n_equilibria']}  wall: {s['wall_s']:.1f}s")
    for name, ax in payload["axes"].items():
        print(f"  {name:12s} spread {ax['spread']['spread_pct']:6.2f} %  "
              f"max|err| {ax['spread']['max_abs_pct']:6.2f} %  "
              f"(flux spread {ax['spread_flux']['spread_pct']:6.2f} %)")
    print("gates:", json.dumps(s["gate_check"]))
    print("worst:", json.dumps(s["max_abs_err_where"]),
          f"{s['max_abs_err_pct']:.2f} %")
    print("order on/off lattice:",
          json.dumps(payload["observed_order_on_lattice"]),
          json.dumps(payload["observed_order_off_lattice"]))
    print("wrote", path, png)


if __name__ == "__main__":
    main()
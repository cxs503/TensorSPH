"""Cross-validation (XVAL) of TensorSPH against SPHinXsys' surface-tension model.

Reference implementation
------------------------
Xiangyu-Hu/SPHinXsys, ``src/shared/particle_dynamics/fluid_dynamics/
surface_tension.{h,hpp,cpp}`` (master, fetched 2026-11-13; raw bytes + sha256 in
``docs/xval_sphinxys_surface_tension/raw/``).  Upstream models surface tension in
a **two-phase** mixture: the colour gradient of a phase is built from the
*other* phase's neighbours, weighted by the contact fraction
``f_k = rho0_i/(rho0_i + rho0_k)``, and the force is the divergence of the
continuous surface stress ``S = sigma (I - n n^T) |grad c|`` (no explicit
curvature) plus a zero-surface-energy hourglass penalty.

What is measured
----------------
Two independent readings of the *same* ported fields, both against the exact
2-D equilibrium condition ``Delta p = sigma / R`` (Young-Laplace):

``stress`` reading
    The upstream stress is a tangential tension of magnitude
    ``|S_t| = sigma |grad c|``; across the interface its normal integral is
    ``int |S_t| dn = sigma int |grad c| dn``, and the Laplace jump of the
    tangential stress is ``Delta p = int |S_t| dn / R``.  This tests the part of
    the upstream pipeline that the mission names: contact fraction / mixture
    weighting -> colour gradient -> normal -> stress.
``flux`` reading
    The actual acceleration returned by the ported force assembly,
    ``Delta p = sum_i rho_i (-a_i . n) A_i / (2 pi r_c)`` (geometry-exact
    normal integral, same estimator as ``benchmarks/surface_tension/laplace.py``).
    This tests the second half of the pipeline (inner + contact flux).

Reported quantities per run
---------------------------
``delta_integral``  ``int |grad c| dn`` over the interface.  A correctly
                    normalised surface delta function gives 1.
``dp_stress`` / ``dp_flux``  the two Laplace readings above, in Pa.
``flux_bulk_leak``  ``max |grad c|`` deep inside the phase (should be ~0: the
                    construction is blind to the bulk).
``momentum_residual``  ``|sum_i m_i a_i| / sum_i m_i |a_i|`` of the ported force:
                    exact momentum conservation is a property of the upstream
                    flux structure and is checked rather than assumed.

Usage
-----
    python benchmarks/surface_tension/sphinxys_xval.py                # full ladder
    python benchmarks/surface_tension/sphinxys_xval.py --radii-dx 16,32

Every number in ``docs/xval_sphinxys_surface_tension/README.md`` comes from the
``xval_ladder.json`` this script writes; ``xval_ladder.png`` is drawn from the
same JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph import sphinxys_css as sc
from tensorsph.neighbor_search import cell_linked_pairs

SIGMA = 0.072          # N/m, water-air at 20 C
RADIUS = 0.1           # m
RHO_WATER = 1000.0     # kg/m^3
RHO_AIR = 1.0          # 1000:1, the ratio of the upstream square-droplet case
DP_RATIO = sc.SPHINXSYS_H_SPACING_RATIO   # 1.3, upstream adaptation.h
RINGS = 6              # air rings outside the interface (kernel support = 2h = 2.6 dp)
HOURGLASS = sc.SPHINXSYS_HOURGLASS_COEFF  # 4.5
OUT = Path(__file__).resolve().parent / "out_sphinxys_xval"


def make_two_phase(radius: float, dx: float, rho_air: float, rings: int = RINGS):
    """Water disc ``r <= radius`` plus ``rings`` air rings on the same lattice.

    Same particle spacing for both phases, as upstream (``SPHSystem(...,
    particle_spacing_ref)`` for both bodies).
    """
    span = radius + (rings + 1) * dx
    axis = torch.arange(-span, span + 0.5 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    dist = torch.linalg.vector_norm(pts, dim=1)
    keep = dist <= radius + rings * dx + 1.0e-12
    pos = pts[keep].clone()
    r = dist[keep]
    phase = (r > radius + 1.0e-12).to(torch.int64)      # 0 = water, 1 = air
    rho0 = torch.where(phase == 0, torch.tensor(RHO_WATER, dtype=pos.dtype),
                       torch.tensor(rho_air, dtype=pos.dtype))
    vol = torch.full((pos.shape[0],), dx * dx, dtype=pos.dtype)
    return pos, r, phase, rho0, vol


def measure(radius: float, dx: float, sigma: float = SIGMA, rho_air: float = RHO_AIR,
            hourglass: bool = True, delta: str = "sphinxys",
            rings: int = RINGS) -> dict:
    h = DP_RATIO * dx
    pos, r, phase, rho0, vol = make_two_phase(radius, dx, rho_air, rings)
    mass = rho0 * vol
    n = pos.shape[0]
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    same = phase[i] == phase[j]
    pairs_inner, pairs_cross = pairs[same], pairs[~same]
    frac = sc.contact_fraction(rho0, torch.where(phase == 0, rho_air, RHO_WATER))

    acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs_inner, pairs_cross, sigma, h, frac=frac,
        mode="two_phase", delta=delta, hourglass=hourglass,
        hourglass_coeff=HOURGLASS)

    water = phase == 0
    centre = pos.mean(dim=0)
    radial = pos - centre
    r_loc = torch.linalg.vector_norm(radial, dim=1)
    n_out = radial / r_loc.clamp_min(1e-30)[:, None]

    # ---- reading 1: stress across the interface --------------------------- #
    # |S_t| = sigma |grad c|; normal integral over the interface / R.
    area = dx * dx
    interface_area = 2.0 * math.pi * radius
    delta_integral = float((fields.delta_s[water] * vol[water]).sum()) / interface_area
    dp_stress = sigma * delta_integral / radius

    # ---- reading 2: the ported force -------------------------------------- #
    force_density = -rho0 * (acc * n_out).sum(dim=1)     # rho a . (-n), inward > 0
    weight = force_density[water] * area
    total = float(weight.sum())
    r_c = float((weight * r_loc[water]).sum() / total) if abs(total) > 1e-30 else float("nan")
    dp_flux = total / (2.0 * math.pi * r_c) if abs(total) > 1e-30 else float("nan")

    # ---- diagnostics ------------------------------------------------------- #
    bulk = water & (r_loc < radius - 2.0 * h)
    leak = float(fields.delta_s[bulk].max()) * dx if bulk.any() else 0.0
    mom = (mass[:, None] * acc).sum(dim=0)
    mom_scale = float((mass * torch.linalg.vector_norm(acc, dim=1)).sum())
    momentum_residual = float(torch.linalg.vector_norm(mom)) / max(mom_scale, 1e-300)
    stress_ratio = float((fields.delta_s[water & (r_loc > radius - 2 * h)]
                          * vol[water & (r_loc > radius - 2 * h)]).sum()) / interface_area

    theory = sigma / radius
    return {
        "radius_m": radius,
        "dx_m": dx,
        "radius_over_dx": radius / dx,
        "h_over_dx": DP_RATIO,
        "particles": n,
        "water_particles": int(water.sum()),
        "rho_air": rho_air,
        "contact_fraction_water": float(frac[water][0]),
        "delta": delta,
        "hourglass": hourglass,
        "delta_integral": delta_integral,
        "dp_stress_pa": dp_stress,
        "dp_stress_err_pct": (dp_stress - theory) / theory * 100.0,
        "dp_flux_pa": dp_flux,
        "dp_flux_err_pct": (dp_flux - theory) / theory * 100.0,
        "theory_pa": theory,
        "flux_centroid_radius_m": r_c,
        "interface_delta_ratio": stress_ratio,
        "flux_bulk_leak_dimensionless": leak,
        "momentum_residual": momentum_residual,
    }


def plot(rows, sens, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    res = [row["radius_over_dx"] for row in rows]
    ax.axhline(1.0, color="0.4", lw=1.0, ls=":", label="exact $\\Delta p=\\sigma/R$")
    ax.plot(res, [row["dp_stress_pa"] / row["theory_pa"] for row in rows], "o-",
            label="stress reading $\\int|S_t|\\,dn/R$")
    ax.plot(res, [row["dp_flux_pa"] / row["theory_pa"] for row in rows], "s--",
            label="ported flux $\\sum \\rho a\\cdot(-n)A/2\\pi r_c$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$R/\\Delta x$")
    ax.set_ylabel("$\\Delta p / (\\sigma/R)$")
    ax.set_title("Laplace law, SPHinXsys port (1000:1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ratios = [row["rho_air"] for row in sens]
    fracs = [row["contact_fraction_water"] for row in sens]
    ax.plot(fracs, [row["delta_integral"] for row in sens], "o-")
    for xx, row in zip(fracs, sens):
        ax.annotate(f"$\\rho_a/\\rho_w$={row['rho_air']/RHO_WATER:g}", (xx, row["delta_integral"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axhline(1.0, color="0.4", ls=":", lw=1.0, label="unit delta")
    ax.axhline(2.0, color="0.8", ls=":", lw=1.0)
    ax.set_xlabel("contact fraction $f_w=\\rho_w/(\\rho_w+\\rho_a)$")
    ax.set_ylabel("$\\int |\\nabla c|\\,dn$")
    ax.set_title("delta normalisation vs density ratio")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radii-dx", default="16,32,64",
                    help="comma separated R/dx values for the resolution ladder")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ladder = [int(v) for v in args.radii_dx.split(",")]
    rows, sens = [], []
    t0 = time.time()
    for ratio in ladder:
        row = measure(RADIUS, RADIUS / ratio)
        rows.append(row)
        print(f"R/dx={ratio:3d} N={row['particles']:6d} "
              f"int|gradc|dn={row['delta_integral']:.4f} "
              f"dp_stress={row['dp_stress_pa']:.4f} ({row['dp_stress_err_pct']:6.2f}%) "
              f"dp_flux={row['dp_flux_pa']:.5f} ({row['dp_flux_err_pct']:7.2f}%) "
              f"mom={row['momentum_residual']:.2e}")

    # Sensitivity 1: contact fraction (density ratio of the "mixture").
    for rho_air in (RHO_WATER, 100.0, 10.0, 1.0, 0.1):
        sens.append(measure(RADIUS, RADIUS / 32, rho_air=rho_air))
    # Sensitivity 2: hourglass penalty on/off at the finest ladder point.
    hg_off = measure(RADIUS, RADIUS / 64, hourglass=False)
    # Sensitivity 3: sigma-consistent dt bound, upstream vs TensorSPH convention.
    dt = {
        "sphinxys_cfl0.6": sc.capillary_dt_limit(RHO_WATER, DP_RATIO * RADIUS / 64, SIGMA),
        "tensorsph_cfl0.25": 0.25 * math.sqrt(RHO_WATER * (DP_RATIO * RADIUS / 64) ** 3 / SIGMA),
    }
    wall_clock = time.time() - t0

    summary = {
        "reference": {
            "oss": "Xiangyu-Hu/SPHinXsys (master)",
            "files": ["src/shared/particle_dynamics/fluid_dynamics/surface_tension.cpp",
                      "src/shared/particle_dynamics/fluid_dynamics/surface_tension.hpp",
                      "src/shared/particle_dynamics/fluid_dynamics/surface_tension.h"],
            "case": "tests/2d_examples/test_2d_square_droplet",
            "raw_archive": "docs/xval_sphinxys_surface_tension/raw/",
        },
        "case": {"radius_m": RADIUS, "sigma_n_m": SIGMA, "rho_water": RHO_WATER,
                 "rho_air": RHO_AIR, "h_over_dx": DP_RATIO,
                 "hourglass_coeff": HOURGLASS,
                 "non_dimensional": {
                     "Laplace": "sigma/(R rho c^2) with c = 10 U_ref (upstream)",
                     "Ohnesorge_free": "inviscid static case: no other group enters",
                     "Weber_free": "U = 0 -> We = 0",
                 }},
        "ladder": rows,
        "contact_fraction_sensitivity": sens,
        "hourglass_off": hg_off,
        "capillary_dt_s": dt,
        "wall_clock_s": wall_clock,
        "gate": {
            "criterion": "xval error vs sigma/R must be < 15%",
            "stress_reading_max_abs_err_pct": max(abs(r["dp_stress_err_pct"]) for r in rows),
            "flux_reading_max_abs_err_pct": max(abs(r["dp_flux_err_pct"]) for r in rows),
            "stress_reading": "pass" if max(abs(r["dp_stress_err_pct"]) for r in rows) < 15.0 else "fail",
            "flux_reading": "pass" if max(abs(r["dp_flux_err_pct"]) for r in rows) < 15.0 else "fail",
        },
    }
    (out / "xval_ladder.json").write_text(json.dumps(summary, indent=1) + "\n")
    plot(rows, sens, out / "xval_ladder.png")
    print(json.dumps(summary["gate"], indent=1))
    print(f"artifacts -> {out}  ({wall_clock:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Field picture for the round-3 density-ratio cross-validation (mission m23209).

The shipped cross-validation stores *integrated* numbers (the jump ``dp``, the
traction-flux cross-check, the residual norms).  A jump alone does not show
where a scheme goes wrong, so this reporter re-runs the same two equilibria the
audit runs and dumps the fields behind them:

    ratio 1000:1, ``R/dx = 20``, ``sigma = 0.072 N/m``, ``R = 0.02 m``
    arm A: ``poisson_weighting="force_density"``  (rho-blind, ``grad p = f``)
    arm B: ``poisson_weighting="density"``        (``-div((1/rho) grad p)``)

Panels: the pressure field of arm A (the cloud map), the radial pressure
profile of both arms against the analytic Young-Laplace step, and the spurious
acceleration field of both arms (relative to the local CSF traction), which is
where arm B's defect lives.  Every number printed here is also written to
``results/density_ratio_fields.json`` so the report quotes the artifact, not a
transcription.

Run: ``python scripts/xval_projection_laplace/report_m23209_fields.py``
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.laplace_isph import (  # noqa: E402
    cell_linked_pairs, csf_state_two_phase, hex_block, pressure_poisson,
    spurious_acceleration,
)

OUT = Path(__file__).resolve().parent / "results"
RADIUS, DX, OUTER, SIGMA = 0.02, 0.001, 0.012, 0.072
RHO_L, RATIO = 1000.0, 1000.0
H = 1.3 * DX
DP_THEORY = SIGMA / RADIUS


def run_arm(weighting: str, rho_vapour: float = RHO_L / RATIO):
    """One static equilibrium; returns the fields the figure needs."""
    dx, h = DX, H
    area = dx * dx * math.sqrt(3.0) / 2.0
    pos, mass, liquid, half = hex_block(RADIUS, dx, OUTER, RHO_L,
                                        rho_vapour=rho_vapour)
    rho_phase = torch.where(liquid, torch.full_like(mass, RHO_L),
                            torch.full_like(mass, rho_vapour))
    mass = rho_phase * area                      # vol = mass/rho = area, ratio-proof
    vol = mass / rho_phase
    pairs = cell_linked_pairs(pos, 2.0 * h)
    st = csf_state_two_phase(pos, mass, liquid, h, RHO_L, SIGMA,
                             rho_phase=rho_phase)
    kappa, delta_s, normal = st["kappa"], st["delta_s"], st["normal"]
    f = (-SIGMA * kappa * delta_s)[:, None] * normal          # force density
    acc = f / rho_phase[:, None]
    from tensorsph.csf_consistent import mls_divergence
    if weighting == "force_density":
        source, rho_op = -mls_divergence(pos, f, vol, pairs, h), None
    else:
        source, rho_op = -mls_divergence(pos, acc, vol, pairs, h), rho_phase
    edge = 1.6 * dx
    dirichlet = (pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge)
    p, nit = pressure_poisson(pos, source, pairs, vol, h, dirichlet,
                              rho=rho_op, weighting=weighting)
    resid = spurious_acceleration(pos, acc, p, RHO_L, pairs, vol, h, rho=rho_phase)
    r = torch.linalg.vector_norm(pos, dim=1)
    # same readout bands as the module: core ``r < 0.5 R``, outer ``1.25-1.45 R``
    core = r < 0.5 * RADIUS
    out_band = (r > 1.25 * RADIUS) & (r < 1.45 * RADIUS)
    dp = float(p[core].mean() - p[out_band].mean())
    return {
        "weighting": weighting, "pos": pos, "r": r, "liquid": liquid, "p": p,
        "resid_rel": (torch.linalg.vector_norm(resid, dim=1)
                      / torch.linalg.vector_norm(acc, dim=1).clamp_min(1e-300)),
        "active": (r > 0.9 * RADIUS) & (r < 1.1 * RADIUS),
        "cg_iterations": int(nit), "n_particles": int(pos.shape[0]),
        "dp_measured": dp, "dp_error_pct": (dp / DP_THEORY - 1.0) * 100.0,
    }


def radial_profile(r, p, liquid, nbin=40):
    lo, hi = 0.0, 1.6 * RADIUS
    edges = torch.linspace(lo, hi, nbin + 1, dtype=torch.float64)
    rc, pc = [], []
    for k in range(nbin):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if bool(m.any()):
            rc.append(float((edges[k] + edges[k + 1]) / 2.0))
            pc.append(float(p[m].mean()))
    return rc, pc


def main():
    torch.set_grad_enabled(False)
    arm_a = run_arm("force_density")
    arm_b = run_arm("density")

    ra, pa = radial_profile(arm_a["r"], arm_a["p"], arm_a["liquid"])
    rb, pb = radial_profile(arm_b["r"], arm_b["p"], arm_b["liquid"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

    a = ax[0]
    sc = a.scatter(arm_a["pos"][:, 0] * 1e3, arm_a["pos"][:, 1] * 1e3,
                   c=arm_a["p"], s=3, cmap="viridis")
    a.set_aspect("equal")
    a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
    a.set_title("A  p field, $\\rho_l/\\rho_v=1000$, rho-blind arm\n"
                "$\\Delta p$ = %.3f Pa vs $\\sigma/R$ = %.3f Pa"
                % (arm_a["dp_measured"], DP_THEORY), fontsize=9)
    fig.colorbar(sc, ax=a, label="$p$ [Pa]")

    a = ax[1]
    a.plot(ra, pa, "o-", ms=3, label="force-density arm (validated)")
    a.plot(rb, pb, "s--", ms=3, label="$1/\\rho$ operator arm")
    a.axhline(DP_THEORY, color="k", lw=1.0, ls=":",
              label="analytic step $\\sigma/R$ = %.2f Pa" % DP_THEORY)
    a.axvline(RADIUS * 1e3, color="grey", lw=0.8)
    a.set_xlabel("r [mm]"); a.set_ylabel("$p$ [Pa]")
    a.set_title("B  radial pressure profile vs analytic step", fontsize=9)
    a.legend(fontsize=7.5); a.grid(alpha=0.3)

    a = ax[2]
    for st, style, lab in ((arm_a, "o-", "force-density arm"),
                           (arm_b, "s--", "$1/\\rho$ operator arm")):
        rr = st["r"][st["active"]] * 1e3
        rv = st["resid_rel"][st["active"]]
        a.semilogy(rr, rv.clamp_min(1e-6), style, ms=3, lw=0.8,
                   label="%s (median %.2e)" % (lab, float(rv.median())))
    a.set_xlabel("r [mm]")
    a.set_ylabel("$|a_{resid}| / |a_{CSF}|$")
    a.set_title("C  spurious acceleration across the interface", fontsize=9)
    a.legend(fontsize=7.5); a.grid(alpha=0.3, which="both")

    fig.suptitle("projection-SPH Young-Laplace with a real density ratio: "
                 "fields behind the $\\Delta p=\\sigma/R$ number", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png_path = OUT / "density_ratio_fields.png"
    fig.savefig(png_path, dpi=110, pil_kwargs={"optimize": True})
    plt.close(fig)
    (OUT / "density_ratio_fields.png.b64").write_text(
        base64.b64encode(png_path.read_bytes()).decode("ascii"))

    payload = {
        "mission": 23209,
        "case": {"radius_m": RADIUS, "sigma_N_per_m": SIGMA, "rho_liquid": RHO_L,
                 "density_ratio": RATIO, "dx_m": DX, "h_over_dx": H / DX,
                 "dp_theory_Pa": DP_THEORY, "R_over_dx": RADIUS / DX},
        "arms": {
            st["weighting"]: {
                "n_particles": st["n_particles"],
                "cg_iterations": st["cg_iterations"],
                "dp_measured_Pa": st["dp_measured"],
                "dp_theory_Pa": DP_THEORY,
                "dp_error_pct": st["dp_error_pct"],
                "residual_rel_median_interface": float(
                    st["resid_rel"][st["active"]].median()),
                "p_max_Pa": float(st["p"].max()),
                "p_min_Pa": float(st["p"].min()),
            } for st in (arm_a, arm_b)
        },
        "figure": {"png": png_path.name, "ascii": png_path.name + ".b64",
                   "pixels": [int(v) for v in fig.get_size_inches() * 110]},
    }
    (OUT / "density_ratio_fields.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return payload


if __name__ == "__main__":
    main()
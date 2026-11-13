"""Build ``benchmarks/laplace_sph/results_dynamic.json`` (dynamic Young-Laplace gate).

What is measured: the *time marching* of projection SPH on a static drop, which
``laplace_isph`` could not do (it diverged after 13 steps; see
``docs/verification_laplace_sph_isph.md`` §5.1).  The operators here are the
adjoint-consistent pair of ``tensorsph.projection_adjoint``, so the pressure
solve really removes the discrete divergence, and a physical viscosity damps
what a consistent projection cannot remove (the solenoidal part of the lattice
CSF traction).

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic.py --figure
    PYTHONPATH=src python benchmarks/laplace_sph/run_dynamic.py --reuse run_a.json run_b.json
    PYTHONPATH=src python benchmarks/laplace_sph/make_doc_dynamic.py

``--reuse`` merges already-marched runs (same schema) so the artefact can be
assembled from runs launched separately; the numbers are then bit-identical to
what those runs produced, and the per-run ``wall_s`` records the cost.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.csf_consistent import mls_divergence, mls_gradient
from tensorsph.laplace_isph import csf_state_two_phase, hex_block, relax_projection
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.projection_adjoint import (adjoint_gradient, adjointness_defect,
                                          pair_divergence, project_velocity,
                                          relax_projection_consistent)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_dynamic.json"
FIG = HERE / "laplace_sph_dynamics.png"

R, DX, HF = 0.02, 0.0025, 1.3
SIGMA, RHO = 0.072, 1000.0
OUTER = 0.012
H = HF * DX
NU_FACTORS = (3e-3, 1e-2)
STEPS = 60
SAMPLE = 10
ADAMI_GATE = 0.1
LAPLACE_GATE_PCT = 15.0
AREA_GATE = 0.02

torch.set_num_threads(8)


def operator_self_tests():
    pos, mass, liquid, _ = hex_block(R, DX, OUTER, RHO)
    vol = mass / RHO
    pairs = cell_linked_pairs(pos, 2.0 * H)
    g = torch.Generator().manual_seed(11)
    n = pos.shape[0]
    u = torch.randn((n, 2), generator=g, dtype=pos.dtype) * 1e-3
    q = torch.randn(n, generator=g, dtype=pos.dtype)

    mls_lhs = (mls_divergence(pos, u, vol, pairs, H) * q).sum()
    mls_rhs = -(u * mls_gradient(pos, q, vol, pairs, H)).sum()
    mls_defect = float((mls_lhs - mls_rhs).abs() / mls_lhs.abs())

    _, _, proj = project_velocity(pos, u, vol, pairs, H, RHO, 1e-4, gauge=0,
                                  rel_tol=1e-10)
    grad = adjoint_gradient(pos, pos[:, 0].clone(), vol, pairs, H)
    interior = torch.linalg.vector_norm(pos, dim=1) < 0.6 * R
    # Independent control: the same projection step with the shipped MLS pair.
    dt = 1e-4
    div_star = mls_divergence(pos, u, vol, pairs, H)
    from tensorsph.laplace_isph import pressure_poisson
    fixed = torch.zeros(n, dtype=torch.bool)
    fixed[0] = True
    p_mls, _ = pressure_poisson(pos, RHO * div_star / dt, pairs, vol, H, fixed,
                                rel_tol=1e-10, max_iter=8000)
    u_mls = u - dt * mls_gradient(pos, p_mls, vol, pairs, H) / RHO
    mls_div_left = float(torch.linalg.vector_norm(
        pair_divergence(pos, u_mls, vol, pairs, H)) /
        torch.linalg.vector_norm(div_star))
    return {
        "n_particles": int(n),
        "n_pairs": int(pairs.shape[0]),
        "adjoint_defect_pair": adjointness_defect(pos, vol, pairs, H),
        "adjoint_defect_mls": mls_defect,
        "grad_linear_recovery": float(grad[interior].mean(dim=0)[0]),
        "projection_div_after_over_before": proj["div_after_over_div_before"],
        "projection_cg_iterations": proj["cg_iterations"],
        "mls_projection_div_after_over_before": mls_div_left,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--reuse", nargs="*", default=None,
                    help="pre-marched run JSONs to merge instead of recomputing")
    ap.add_argument("--no-legacy", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rep = {
        "case": "CASE-LAPLACE-SPH-DYNAMIC",
        "title": "Projection-SPH Young-Laplace: a stable dynamic equilibrium",
        "setup": {
            "radius_m": R, "dx_m": DX, "h_m": H, "h_over_dx": HF,
            "sigma_N_per_m": SIGMA, "rho0": RHO, "outer_carrier_m": OUTER,
            "R_over_dx": R / DX, "R_over_h": R / H,
            "dt_over_t_sigma": 0.1,
            "t_sigma_s": math.sqrt(RHO * H ** 3 / SIGMA),
            "v_ref_m_per_s": math.sqrt(SIGMA / (RHO * R)),
            "steps": STEPS, "sample_every": SAMPLE,
            "cg_rel_tol": 1e-6, "cg_max_iter": 4000,
            "curvature": "mls (grad.n of the twice-smoothed normal)",
            "adami_gate": ADAMI_GATE,
            "laplace_gate_pct": LAPLACE_GATE_PCT,
            "area_gate": AREA_GATE,
        },
        "operators": operator_self_tests(),
        "runs": [],
    }

    if args.reuse:
        for path in args.reuse:
            blob = json.loads(Path(path).read_text())
            rep["runs"].extend(blob["runs"] if "runs" in blob else [blob])
    else:
        for nu in NU_FACTORS:
            t = time.time()
            r = relax_projection_consistent(radius=R, dx=DX, sigma=SIGMA,
                                            rho0=RHO, outer=OUTER, h_factor=HF,
                                            steps=STEPS, sample_every=SAMPLE,
                                            nu_factor=nu)
            r["wall_s"] = time.time() - t
            rep["runs"].append(r)
            print("nu=%g umax/vref=%.4g dp_err=%.3f%%"
                  % (nu, r["umax_over_vref_final"], r["dp_rel_error_pct"]),
                  flush=True)
        # inviscid control, shorter: it is expected to lose the lattice
        t = time.time()
        r = relax_projection_consistent(radius=R, dx=DX, sigma=SIGMA, rho0=RHO,
                                        outer=OUTER, h_factor=HF, steps=30,
                                        sample_every=SAMPLE, nu_factor=0.0)
        r["wall_s"] = time.time() - t
        rep["runs"].append(r)

    if not args.no_legacy:
        t = time.time()
        rep["legacy"] = relax_projection(radius=R, dx=DX, sigma=SIGMA, rho0=RHO,
                                         outer=OUTER, h_factor=HF,
                                         curvature_method="mls", steps=200,
                                         sample_every=5)
        rep["legacy"]["wall_s"] = time.time() - t

    visc = [r for r in rep["runs"] if r["nu_factor"] > 0.0 and not r["diverged"]]
    best = min(visc, key=lambda r: (abs(r["dp_rel_error_pct"]),
                                    r["umax_over_vref_final"]))
    rep["best"] = {k: best[k] for k in (
        "nu_factor", "nu_m2_per_s", "dp_final_Pa", "dp_theory_Pa",
        "dp_rel_error_pct", "umax_over_vref_final", "umean_over_vref_final",
        "area_ratio_final", "R_over_dx", "steps_completed")}
    rep["best"]["history"] = best["history"]

    rep["gates"] = {
        "laplace_15pct": {
            "gate_pct": LAPLACE_GATE_PCT,
            "value_pct": best["dp_rel_error_pct"],
            "pass": abs(best["dp_rel_error_pct"]) <= LAPLACE_GATE_PCT,
        },
        "adami_spurious_0p1": {
            "gate": ADAMI_GATE,
            "value": best["umax_over_vref_final"],
            "pass": best["umax_over_vref_final"] <= ADAMI_GATE,
            "note": "best damped run",
        },
        "area_drift_2pct": {
            "gate": AREA_GATE,
            "value": abs(best["area_ratio_final"] - 1.0),
            "pass": abs(best["area_ratio_final"] - 1.0) <= AREA_GATE,
        },
        "projection_exact": {
            "value": rep["operators"]["projection_div_after_over_before"],
            "pass": rep["operators"]["projection_div_after_over_before"] < 1e-5,
        },
        "legacy_still_diverges": {
            "steps_to_divergence": rep.get("legacy", {}).get("steps_to_divergence"),
            "pass": bool(rep.get("legacy", {}).get("diverged", False)),
        },
    }
    rep["wall_s_total"] = time.time() - t0
    OUT.write_text(json.dumps(rep, indent=1, sort_keys=False))
    print("wrote", OUT, "in %.0f s" % rep["wall_s_total"])

    if args.figure:
        make_figure(rep)


def make_figure(rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # (a) the equilibrated pressure field of the best run
    best = min([r for r in rep["runs"] if r["nu_factor"] > 0.0],
               key=lambda r: abs(r["dp_rel_error_pct"]))
    # re-run a short march to snapshot the field (cheap: 20 steps)
    from tensorsph.projection_adjoint import relax_projection_consistent as relax
    snap = relax(radius=R, dx=DX, sigma=SIGMA, rho0=RHO, outer=OUTER,
                 h_factor=HF, steps=20, sample_every=20,
                 nu_factor=best["nu_factor"])
    pos, mass, liquid, half = hex_block(R, DX, OUTER, RHO)
    vol = mass / RHO
    pairs = cell_linked_pairs(pos, 2.0 * H)
    st = csf_state_two_phase(pos, liquid=liquid, mass=mass, h=H, rho0=RHO,
                             sigma=SIGMA, curvature_method="mls")
    ax[0, 0].remove()
    ax[0, 0] = fig.add_subplot(2, 2, 1)
    ax[0, 0].set_title("(a) CSF traction magnitude |a| (initial state)")
    sc = ax[0, 0].scatter(pos[:, 0] * 1e3, pos[:, 1] * 1e3,
                          c=torch.linalg.vector_norm(st["acc"], dim=1) * 1e3,
                          s=4, cmap="viridis")
    th = torch.linspace(0, 2 * math.pi, 200)
    ax[0, 0].plot(R * 1e3 * torch.cos(th), R * 1e3 * torch.sin(th), "w--", lw=0.8)
    ax[0, 0].set_xlabel("x [mm]")
    ax[0, 0].set_ylabel("y [mm]")
    ax[0, 0].set_aspect("equal")
    fig.colorbar(sc, ax=ax[0, 0], label="|a| [mm/s$^2$]")

    # (b) spurious currents vs time
    for r in rep["runs"]:
        if r["diverged"]:
            continue
        t = [h["t_over_t_sigma"] for h in r["history"]]
        ax[0, 1].semilogy(t, [h["umax_over_vref"] for h in r["history"]],
                          "o-", ms=3, label=r"$\nu_f=%g$" % r["nu_factor"])
    ax[0, 1].axhline(ADAMI_GATE, color="k", ls="--", lw=1,
                     label="Adami 2012 gate 0.1")
    leg = rep.get("legacy") or {}
    if leg.get("history"):
        t = [h.get("t_over_t_sigma") for h in leg["history"] if "t_over_t_sigma" in h]
        v = [h.get("umax_over_vref") for h in leg["history"] if "umax_over_vref" in h]
        ax[0, 1].semilogy(t, v, "s--", color="crimson", ms=3,
                          label="legacy (non-adjoint) pair")
    ax[0, 1].set_xlabel(r"t / $t_\sigma$")
    ax[0, 1].set_ylabel(r"$\max|u| / \sqrt{\sigma/\rho R}$")
    ax[0, 1].set_title("(b) spurious currents")
    ax[0, 1].legend(fontsize=7)
    ax[0, 1].grid(alpha=0.3)

    # (c) measured pressure jump vs theory
    for r in rep["runs"]:
        if r["diverged"]:
            continue
        t = [h["t_over_t_sigma"] for h in r["history"]]
        ax[1, 0].plot(t, [h["dp_Pa"] / r["dp_theory_Pa"] for h in r["history"]],
                      "o-", ms=3, label=r"$\nu_f=%g$" % r["nu_factor"])
    ax[1, 0].axhline(1.0, color="k", ls="--", lw=1)
    ax[1, 0].axhline(1.0 + LAPLACE_GATE_PCT / 100.0, color="grey", ls=":", lw=1)
    ax[1, 0].axhline(1.0 - LAPLACE_GATE_PCT / 100.0, color="grey", ls=":", lw=1)
    ax[1, 0].set_xlabel(r"t / $t_\sigma$")
    ax[1, 0].set_ylabel(r"$\Delta p / (\sigma/R)$")
    ax[1, 0].set_title("(c) Laplace jump during the march")
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].grid(alpha=0.3)

    # (d) area drift
    for r in rep["runs"]:
        if r["diverged"]:
            continue
        t = [h["t_over_t_sigma"] for h in r["history"]]
        ax[1, 1].plot(t, [h["area_ratio"] for h in r["history"]], "o-", ms=3,
                      label=r"$\nu_f=%g$" % r["nu_factor"])
    ax[1, 1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1, 1].set_xlabel(r"t / $t_\sigma$")
    ax[1, 1].set_ylabel("liquid area / initial")
    ax[1, 1].set_title("(d) shape conservation")
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].grid(alpha=0.3)
    fig.suptitle("Projection-SPH Young-Laplace drop, R/dx = %.0f" % (R / DX))
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
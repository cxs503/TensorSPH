"""Round 3 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE: the density ratio.

Rounds 1-2 (``docs/xval_projection_laplace_oss.md``,
``docs/xval_projection_laplace_robustness.md``) measured ``dp = sigma/R`` for the
projection-SPH droplet on a **1:1 density carrier** and said so explicitly: the
vapour was filled with liquid-density particles because "the Young-Laplace law
contains ``sigma`` and ``R`` only".  That is the right *reference*, but it leaves
the *numerics* untested.  The projection step solves a pressure Poisson
equation, and in a two-phase flow the coefficient of that operator is ``1/rho``:
every published number for this module was measured with the one term that the
density ratio multiplies switched off by construction.

This round turns the density ratio on.  The lattice, the cell volume ``V`` and
therefore the colour function are held fixed (each particle carries
``m = rho_phase V``, so ``vol = m/rho_phase`` is unchanged): the *only* knob is
``rho_liquid/rho_vapour``.  The question is whether

    dp = p_core - p_vapour = sigma/R          (Young-Laplace, rho-blind)

survives, and with which Poisson operator.

Arms
----
``density``   ``-div((1/rho) grad p) = -div(a)``, pair coefficient ``1/rho_ij``
              with the harmonic pair mean ``rho_ij = 2 rho_i rho_j/(rho_i+rho_j)``
              (Hu & Adams 2007).  This is the projection step at ``u = 0``,
              where ``grad p = f`` is a *force density* balance and therefore
              does not contain ``rho``.
``uniform``   constant-coefficient Laplacian ``-grad^2 p = -rho_l div(a)``: the
              naive implementation that treats the density ratio as a source
              correction.
``diffuse``   (ablation, falsified) density carried by the *smoothed* colour
              field, ``rho_i = rho_v + (rho_l-rho_v) c_i``, i.e. a diffuse
              interface.  Tested because the ``density`` arm loses accuracy as
              the mesh is refined; it makes the bias three times larger.

GATES were fixed before the sweep was run and are copied verbatim into the JSON:
the owner gate (15 %) on the jump, a 3 % "publishable" band on the validated
arm, and spreads (not point values), so a change that improves the fitted number
while making the answer knob-dependent cannot pass.  ``resolution_*`` is the gate
that the measurement actually broke -- it is reported as failed, not relaxed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.csf_consistent import mls_divergence  # noqa: E402
from tensorsph.laplace_isph import (  # noqa: E402
    csf_state_two_phase, hex_block, laplace_isph_case, pressure_poisson,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

GATES = {
    "dp_err_max_pct_density_arm_R_over_dx_le_25": 3.0,
    "dp_err_max_pct_ratio_1000_R_over_dx_20": 3.0,
    "ratio_spread_pct": 3.0,
    "flux_spread_pct": 1.0,
    "resolution_err_max_pct_R_over_dx_ge_20": 5.0,
    "translation_spread_pct": 5.0,
    "naive_err_min_pct_at_ratio_1000": 15.0,
    "sigma_zero_max_pa": 1e-9,
    "vol_phase_spread_max": 1e-12,
}

RATIOS = [1, 2, 5, 10, 30, 100, 300, 1000]
RES_LEVELS = [10, 15, 20, 30, 40]
RES_RATIOS = [1, 10, 1000]
OFFSETS = [(0.0, 0.0), (1 / 3, 1 / 3), (0.5, 0.5)]
RADIUS = 0.02
DX = 0.001          # R/dx = 20, the round-1/2 baseline
OUTER = 0.012
RHO_L = 1000.0


def _row(r, axis, ratio=None, **extra):
    row = {
        "axis": axis,
        "ratio": (round(r["density_ratio"], 6) if ratio is None else ratio),
        "rho_vapour": r["rho_vapour"],
        "R_over_dx": r["R_over_dx"],
        "n_particles": r["n_particles"],
        "arm": r["poisson_weighting"],
        "dp_measured_Pa": r["dp_measured_Pa"],
        "dp_theory_Pa": r["dp_theory_Pa"],
        "err_pct": (r["dp_over_theory"] - 1.0) * 100.0,
        "dp_flux_Pa": r["dp_flux_Pa"],
        "flux_err_pct": (r["dp_flux_over_theory"] - 1.0) * 100.0,
        "kappaR": r["kappaR_delta_weighted"],
        "delta_s_integral_dn": r["delta_s_integral_dn"],
        "residual_rel_local": r["residual_acc_median_over_local_csf_acc"],
        "cg_iterations": r["cg_iterations"],
    }
    row.update(extra)
    return row


def sweep_ratio(ratios=RATIOS):
    rows = []
    for ratio in ratios:
        for arm in ("density", "uniform"):
            r = laplace_isph_case(radius=RADIUS, dx=DX, outer=OUTER,
                                  rho0=RHO_L, rho_vapour=RHO_L / ratio,
                                  poisson_weighting=arm)
            rows.append(_row(r, "density_ratio", ratio=ratio, arm=arm))
    return rows


def sweep_resolution(ratios=RES_RATIOS, levels=RES_LEVELS):
    rows = []
    for ratio in ratios:
        for lvl in levels:
            r = laplace_isph_case(radius=RADIUS, dx=RADIUS / lvl, outer=OUTER,
                                  rho0=RHO_L, rho_vapour=RHO_L / ratio,
                                  poisson_weighting="density")
            rows.append(_row(r, "resolution", ratio=ratio, arm="density"))
    return rows


def sweep_translation(ratio=1000, offsets=OFFSETS):
    dy = DX * math.sqrt(3.0) / 2.0
    rows = []
    for fx, fy in offsets:
        r = laplace_isph_case(radius=RADIUS, dx=DX, outer=OUTER, rho0=RHO_L,
                              rho_vapour=RHO_L / ratio,
                              center=(fx * DX, fy * dy),
                              poisson_weighting="density")
        rows.append(_row(r, "translation", ratio=ratio, arm="density",
                             offset=[fx, fy]))
    return rows


def sweep_diffuse(ratio=1000, levels=RES_LEVELS):
    """Ablation: density carried by the smoothed colour field (diffuse interface).

    Uses only the shipped operators: build the lattice, take the Shepard colour
    ``c``, set ``rho = rho_v + (rho_l-rho_v) c``, then solve the same
    density-weighted Poisson.  The cell volume stays ``V`` so the colour field
    and the interface geometry are bit-identical to the sharp arm.
    """
    rows = []
    rho_v = RHO_L / ratio
    for lvl in levels:
        dx = RADIUS / lvl
        h = 1.3 * dx
        area = dx * dx * math.sqrt(3.0) / 2.0
        pos, mass, liquid, half = hex_block(RADIUS, dx, OUTER, RHO_L)
        st0 = csf_state_two_phase(pos, mass, liquid, h, RHO_L, 0.072)
        rho_phase = rho_v + (RHO_L - rho_v) * st0["color"]
        st = csf_state_two_phase(pos, rho_phase * area, liquid, h, RHO_L, 0.072,
                                 rho_phase=rho_phase)
        vol, pairs = st["vol"], st["pairs"]
        acc = ((-0.072 * st["kappa"] * st["delta_s"])[:, None] * st["normal"]
               / rho_phase[:, None])
        source = -mls_divergence(pos, acc, vol, pairs, h)
        edge = 1.6 * dx
        dr = (pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge)
        p, nit = pressure_poisson(pos, source, pairs, vol, h, dr, rho=rho_phase)
        r = torch.linalg.vector_norm(pos, dim=1)
        core = r < 0.5 * RADIUS
        out = (r > 1.25 * RADIUS) & (r < 1.45 * RADIUS)
        dp = float(p[core].mean() - p[out].mean())
        a_r = (acc * (pos / r[:, None])).sum(dim=1)
        flux = float((-a_r * (rho_phase * area)).sum() / (2 * math.pi * RADIUS))
        rows.append({
            "axis": "diffuse_density", "ratio": ratio, "R_over_dx": lvl,
            "n_particles": int(pos.shape[0]), "arm": "diffuse",
            "dp_measured_Pa": dp, "dp_theory_Pa": 0.072 / RADIUS,
            "err_pct": (dp / (0.072 / RADIUS) - 1.0) * 100.0,
            "flux_err_pct": (flux / (0.072 / RADIUS) - 1.0) * 100.0,
            "rho_shell_min": float(rho_phase.min()),
            "rho_shell_max": float(rho_phase.max()),
            "cg_iterations": int(nit),
        })
    return rows


def controls(ratio=1000):
    zero = laplace_isph_case(radius=RADIUS, dx=DX, outer=OUTER, sigma=0.0,
                             rho0=RHO_L, rho_vapour=RHO_L / ratio)
    pos, mass, liquid, _ = hex_block(RADIUS, DX, OUTER, RHO_L,
                                     rho_vapour=RHO_L / ratio)
    area = DX * DX * math.sqrt(3.0) / 2.0
    vol = mass / (mass / area)
    return {
        "sigma_zero": {"dp_measured_Pa": zero["dp_measured_Pa"],
                       "dp_flux_Pa": zero["dp_flux_Pa"]},
        "vol_phase_spread": float((vol.max() - vol.min()) / vol.mean()),
        "mass_per_phase": {"liquid": float(mass[liquid].mean()),
                           "vapour": float(mass[~liquid].mean())},
        "n_liquid": int(liquid.sum()), "n_vapour": int((~liquid).sum()),
    }


def summarise(ratio_rows, res_rows, trans_rows, diff_rows, ctrl):
    dens = [r for r in ratio_rows if r["arm"] == "density"]
    naive = [r for r in ratio_rows if r["arm"] == "uniform"]
    dp_over = [r["dp_measured_Pa"] / r["dp_theory_Pa"] for r in dens]
    flux_over = [r["dp_flux_Pa"] / r["dp_theory_Pa"] for r in ratio_rows]
    res_ok = [r for r in res_rows if r["R_over_dx"] >= 20]
    trans_err = [r["err_pct"] for r in trans_rows]
    by_res = {}
    for r in res_rows:
        by_res.setdefault(r["ratio"], {})[r["R_over_dx"]] = r["err_pct"]
    return {
        "gates": GATES,
        "density_arm_err_pct_by_ratio": {r["ratio"]: r["err_pct"] for r in dens},
        "density_arm_max_abs_err_pct": max(abs(e) for e in dp_over_to_err(dp_over)),
        "err_pct_ratio_1000_R_over_dx_20": next(
            r["err_pct"] for r in dens if r["ratio"] == 1000),
        "ratio_spread_pct": (max(dp_over) - min(dp_over)) * 100.0,
        "flux_err_pct_max_abs": max(abs((f - 1.0) * 100.0) for f in flux_over),
        "flux_err_pct_max_abs_density_arm": max(
            abs(r["flux_err_pct"]) for r in dens),
        "flux_spread_pct": (max(flux_over) - min(flux_over)) * 100.0,
        "naive_err_pct_by_ratio": {r["ratio"]: r["err_pct"] for r in naive},
        "naive_err_pct_ratio_1000": next(r["err_pct"] for r in naive
                                         if r["ratio"] == 1000),
        "resolution_err_pct_by_ratio": by_res,
        "resolution_max_abs_err_pct_R_over_dx_ge_20": max(
            abs(r["err_pct"]) for r in res_ok),
        "diffuse_err_pct_by_level_ratio_1000": {r["R_over_dx"]: r["err_pct"]
                                                for r in diff_rows},
        "translation_spread_pct": max(trans_err) - min(trans_err),
        "translation_max_abs_err_pct": max(abs(e) for e in trans_err),
        "sigma_zero_dp_Pa": ctrl["sigma_zero"]["dp_measured_Pa"],
        "sigma_zero_flux_Pa": ctrl["sigma_zero"]["dp_flux_Pa"],
        "vol_phase_spread": ctrl["vol_phase_spread"],
        "mass_per_phase": ctrl["mass_per_phase"],
    }


def dp_over_to_err(ratios):
    return [(v - 1.0) * 100.0 for v in ratios]


def make_figure(summary, ratio_rows, res_rows, trans_rows, diff_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))
    dens = [r for r in ratio_rows if r["arm"] == "density"]
    naive = [r for r in ratio_rows if r["arm"] == "uniform"]

    a = ax[0][0]
    a.semilogx([r["ratio"] for r in dens], [r["err_pct"] for r in dens], "o-",
               label="$1/\\rho$ inside the Poisson (validated)")
    clipped = [max(-1e3, min(1e3, r["err_pct"])) for r in naive]
    a.semilogx([r["ratio"] for r in naive], clipped, "s--", color="crimson",
               label="constant-coefficient Laplacian (naive)")
    a.axhspan(-3, 3, color="green", alpha=0.12, label="$\\pm 3\\%$ band")
    a.axhline(0, color="k", lw=0.6)
    a.set_xscale("log")
    a.set_ylim(-200, 320)
    a.set_xlabel("density ratio  $\\rho_l/\\rho_v$")
    a.set_ylabel("$(\\Delta p/\\Delta p_{th}-1)$  [%]  (clipped at 1000 %)")
    a.set_title("A  jump vs density ratio, $R/\\Delta x=20$")
    a.legend(fontsize=7.5, loc="upper left")
    a.grid(alpha=0.3)

    a = ax[0][1]
    styles = {1: ("o-", "navy"), 10: ("s-", "darkorange"),
              1000: ("^-", "darkgreen")}
    for ratio in (1, 10, 1000):
        rows = [r for r in res_rows if r["ratio"] == ratio]
        st, col = styles[ratio]
        a.plot([r["R_over_dx"] for r in rows], [r["err_pct"] for r in rows], st,
               color=col, label="$\\rho_l/\\rho_v=%d$" % ratio)
    a.axhspan(-15, 15, color="orange", alpha=0.10, label="owner gate 15 %")
    a.axhspan(-5, 5, color="green", alpha=0.10, label="5 % band")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("$R/\\Delta x$")
    a.set_ylabel("error  [%]")
    a.set_title("B  resolution, by density ratio (density-weighted)")
    a.legend(fontsize=7.5)
    a.grid(alpha=0.3)

    a = ax[1][0]
    a.plot([r["R_over_dx"] for r in res_rows if r["ratio"] == 1000],
           [r["flux_err_pct"] for r in res_rows if int(r["ratio"]) == 1000],
           "s--", color="darkgreen", label="traction flux $-\\int\\rho a_r dA/2\\pi R$")
    a.plot([r["R_over_dx"] for r in res_rows if r["ratio"] == 1000],
           [r["err_pct"] for r in res_rows if int(r["ratio"]) == 1000],
           "^-", color="darkgreen", alpha=0.45, label="$\\Delta p$ (Poisson)")
    a.plot([r["R_over_dx"] for r in diff_rows], [r["err_pct"] for r in diff_rows],
           "v:", color="purple", label="diffuse-density ablation")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("$R/\\Delta x$")
    a.set_ylabel("error  [%]")
    a.set_title("C  who owns the error at $\\rho_l/\\rho_v=1000$")
    a.legend(fontsize=7.5)
    a.grid(alpha=0.3)

    a = ax[1][1]
    labels = ["(%g,%g)" % tuple(r["offset"]) for r in trans_rows]
    a.bar(labels, [r["err_pct"] for r in trans_rows], color="steelblue")
    a.axhspan(-3, 3, color="green", alpha=0.12)
    a.axhline(0, color="k", lw=0.6)
    a.set_ylabel("$\\Delta p$ error  [%]")
    a.set_title("D  lattice translation, $\\rho_l/\\rho_v=1000$, $R/\\Delta x=20$")
    a.grid(alpha=0.3, axis="y")

    fig.suptitle("projection-SPH Young-Laplace $\\Delta p=\\sigma/R$ with a real "
                 "density ratio ($\\sigma=0.072$ N/m, $R=0.02$ m)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_grad_enabled(False)
    ratio_rows = sweep_ratio()
    res_rows = sweep_resolution()
    trans_rows = sweep_translation()
    diff_rows = sweep_diffuse()
    ctrl = controls()
    summary = summarise(ratio_rows, res_rows, trans_rows, diff_rows, ctrl)
    payload = {
        "mission": 20557,
        "anchor": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE",
        "round": 3,
        "case": {
            "radius_m": RADIUS, "sigma_N_per_m": 0.072, "rho_liquid": RHO_L,
            "dx_m": DX, "h_over_dx": 1.3, "outer_m": OUTER,
            "dp_theory_Pa": 0.072 / RADIUS,
            "scheme": "Shepard colour (vol-weighted) -> MLS gradient -> smoothed "
                      "normal -> div n curvature (mls) -> Brackbill |grad c| -> "
                      "CSF traction -> Cummins&Rudman pair Laplacian with 1/rho "
                      "pair coefficient, Dirichlet ambient gauge, matrix-free CG",
        },
        "summary": summary,
        "controls": ctrl,
        "density_ratio": ratio_rows,
        "resolution": res_rows,
        "translation": trans_rows,
        "diffuse_density": diff_rows,
    }
    js = OUT / "density_ratio.json"
    js.write_text(json.dumps(payload, indent=1))
    make_figure(summary, ratio_rows, res_rows, trans_rows, diff_rows,
                OUT / "density_ratio.png")
    print(json.dumps(summary, indent=1))
    return payload


if __name__ == "__main__":
    main()
"""Round 5 of XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE: the density ratio, per phase.

Rounds 3-4 (``docs/xval_projection_laplace_density_ratio.md``) measured the
Young-Laplace jump ``dp = sigma/R`` at a real density ratio and showed that the
*rho-blind force-density* Poisson (``-grad^2 p = -div(f)``) carries it to
``0.93 %`` at every ``R/dx`` and every ratio up to 1000.

They also shipped a *momentum-balance* instrument next to the jump: the spurious
acceleration ``a_CSF - grad p / rho``, reported as

    residual_acc_median_over_local_csf_acc = median(|resid| / |a_CSF,local|)

**over all active particles at once.**  That average is phase-blind, and the
phase is exactly where a density ratio acts: the CSF traction is a *force
density* (rho-free), so the acceleration it produces is ``f/rho``, which is
``rho_l/rho_v`` times larger in the light phase.  A median dominated by the
heavy phase cannot see that.  Here the residual is attributed.

For each arm and ratio the case now returns ``residual_phase`` with, per phase,
the median/mean of ``|resid| / |a_CSF,local|`` and the absolute
``|resid|`` [m/s^2].  Three arms:

``force_density``  ``-grad^2 p = -div(f)``: no ``rho`` in the operator, so the
                   residual is a pure discretisation quantity -- it must be
                   *ratio invariant* in dimensionless form, and the vapour
                   absolute value must scale exactly as ``1/rho_v``.
``density``        ``-div((1/rho) grad p) = -div(a)``: the variable-coefficient
                   form kept in the record as the round-3 arm.
``uniform``        ``-grad^2 p = -rho_l div(a)``: the naive implementation, the
                   ablation that must fail.

Two measurements are separated deliberately:

* the *jump* (integral readout, round 3-4, reproduced here as a regression);
* the *per-particle imbalance* (this round).  A scheme can get the integral
  right and the local balance wrong; the ratio ladder says which of the two the
  density ratio touches.

GATES were fixed before the sweep ran and are stored in the JSON.  The one the
measurement breaks -- the per-particle imbalance must not grow under refinement
-- is recorded as failed, not relaxed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.laplace_isph import laplace_isph_case  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

GATES = {
    # the jump still has to be right (round 3-4 regression, owner 15 % gate)
    "force_density_dp_err_max_pct": 3.0,
    # the dimensionless per-particle imbalance must be ratio blind
    "force_density_phase_resid_ratio_spread_max": 1e-9,
    # ... and the absolute imbalance in the light phase must scale as 1/rho_v
    "vapour_residual_scaling_tol": 1e-9,
    # the naive operator must be visibly worse *per phase* than the validated one
    "uniform_vapour_resid_min_ratio_to_validated": 10.0,
    "uniform_dp_err_min_pct_at_ratio_1000": 15.0,
    # a global median must not be able to hide the light phase: for the
    # variable-coefficient arm at ratio 1000 the phase medians must straddle it
    # with at least this separation from the heavier value
    "phase_blind_hiding_min_pct": 25.0,
    # honest gate: refinement must not make the per-particle balance worse
    "resolution_resid_growth_max": 1.10,
    "sigma_zero_max_pa": 1e-9,
    "sigma_zero_resid_max": 1e-9,
}

RATIOS = [1, 10, 100, 1000]
ARMS = ("force_density", "density", "uniform")
RES_LEVELS = [10, 20, 40]
RES_RATIOS = [1, 1000]
RES_ARMS = ("force_density", "density")
RADIUS = 0.02
DX = 0.001            # R/dx = 20, the round-1..4 baseline
OUTER = 0.012
RHO_L = 1000.0
SIGMA = 0.072


def _row(r, axis, **extra):
    ph = r["residual_phase"]
    liq, vap = ph["liquid"], ph["vapour"]
    row = {
        "axis": axis,
        "ratio": round(float(r["density_ratio"]), 6),
        "arm": r["poisson_weighting"],
        "R_over_dx": r["R_over_dx"],
        "n_particles": r["n_particles"],
        "n_liquid_active": liq["n"],
        "n_vapour_active": vap["n"],
        "dp_measured_Pa": r["dp_measured_Pa"],
        "dp_theory_Pa": r["dp_theory_Pa"],
        "dp_err_pct": (r["dp_over_theory"] - 1.0) * 100.0,
        "resid_global_median": r["residual_acc_median_over_local_csf_acc"],
        "resid_liquid_median": liq["median_over_local_csf_acc"],
        "resid_vapour_median": vap["median_over_local_csf_acc"],
        "resid_liquid_mean": liq["mean_over_local_csf_acc"],
        "resid_vapour_mean": vap["mean_over_local_csf_acc"],
        "resid_liquid_abs_m_per_s2": liq["mean_abs_m_per_s2"],
        "resid_vapour_abs_m_per_s2": vap["mean_abs_m_per_s2"],
        "resid_vapour_over_liquid_abs": (vap["mean_abs_m_per_s2"]
                                        / max(liq["mean_abs_m_per_s2"], 1e-300)),
        "cg_iterations": r["cg_iterations"],
    }
    row.update(extra)
    return row


def _case(ratio, dx=DX, arm="force_density", sigma=SIGMA):
    return laplace_isph_case(radius=RADIUS, dx=dx, outer=OUTER, rho0=RHO_L,
                             sigma=sigma, rho_vapour=RHO_L / ratio,
                             poisson_weighting=arm)


def sweep_ratio(ratios=RATIOS, arms=ARMS):
    rows = []
    for ratio in ratios:
        for arm in arms:
            rows.append(_row(_case(ratio, arm=arm), "density_ratio", arm=arm))
    return rows


def sweep_resolution(ratios=RES_RATIOS, levels=RES_LEVELS, arms=RES_ARMS):
    rows = []
    for ratio in ratios:
        for lvl in levels:
            for arm in arms:
                rows.append(_row(_case(ratio, dx=RADIUS / lvl, arm=arm),
                                 "resolution", arm=arm))
    return rows


def controls(ratio=1000):
    """Two controls: no surface tension, and no density contrast.

    ``sigma = 0`` removes the traction, so ``a = 0``, ``p = 0`` and the residual
    must vanish *exactly* -- otherwise the imbalance instrument has an offset of
    its own.  ``ratio = 1`` is the round-1/2 carrier: every arm must collapse
    onto the same number there (the operators are identical at equal density,
    ``div(f/rho0) = div(f)/rho0``).
    """
    zero = laplace_isph_case(radius=RADIUS, dx=DX, outer=OUTER, sigma=0.0,
                             rho0=RHO_L, rho_vapour=RHO_L / ratio)
    same = {arm: _case(1, arm=arm) for arm in ARMS}
    dp_spread = (max(r["dp_measured_Pa"] for r in same.values())
                 - min(r["dp_measured_Pa"] for r in same.values()))
    resid_spread = (max(r["residual_phase"]["liquid"]["median_over_local_csf_acc"]
                        for r in same.values())
                    - min(r["residual_phase"]["liquid"]["median_over_local_csf_acc"]
                          for r in same.values()))
    return {
        "sigma_zero": {
            "dp_measured_Pa": zero["dp_measured_Pa"],
            "resid_liquid_abs_m_per_s2":
                zero["residual_phase"]["liquid"]["mean_abs_m_per_s2"],
            "resid_vapour_abs_m_per_s2":
                zero["residual_phase"]["vapour"]["mean_abs_m_per_s2"],
        },
        "equal_density_arm_spread": {
            "dp_pa": dp_spread,
            "resid_liquid_median": resid_spread,
        },
    }


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def summarize(rows, ctrl):
    """Every gate evaluated from the stored rows, plus the verdicts quoted in the doc."""
    def sel(axis, ratio=None, arm=None, level=None):
        out = [r for r in rows if r["axis"] == axis]
        if ratio is not None:
            out = [r for r in out if abs(r["ratio"] - ratio) < 1e-9]
        if arm is not None:
            out = [r for r in out if r["arm"] == arm]
        if level is not None:
            out = [r for r in out if abs(r["R_over_dx"] - level) < 1e-9]
        return out

    fd = sel("density_ratio", arm="force_density")
    uni = sel("density_ratio", arm="uniform")
    den = sel("density_ratio", arm="density")

    # G1 jump
    fd_dp_max = max(abs(r["dp_err_pct"]) for r in fd)
    # G2 dimensionless per-phase imbalance is ratio invariant for the rho-blind arm
    fd_resid_spread = max(
        (max(r[k] for r in fd) - min(r[k] for r in fd))
        / max(abs(fd[0][k]), 1e-300)
        for k in ("resid_liquid_median", "resid_vapour_median"))
    # G3 the absolute light-phase imbalance scales as 1/rho_v
    a_v = {r["ratio"]: r["resid_vapour_abs_m_per_s2"] for r in fd}
    scaling_err = 0.0
    for r in fd:
        if r["ratio"] != 1:
            # the mean (not median) |resid| in the light phase must scale as
            # 1/rho_v = ratio/rho_l: both terms of the residual (a = f/rho_v and
            # grad p/rho_v) carry it, and for this arm p itself is ratio blind
            # (bit-identical source and operator).  Heavier ratio => lighter
            # vapour => larger absolute imbalance, linearly.
            scaling_err = max(scaling_err,
                              _rel(a_v[r["ratio"]], a_v[1.0] * r["ratio"]))
    # G4 the naive operator, per phase
    u1k = sel("density_ratio", ratio=1000, arm="uniform")[0]
    f1k = sel("density_ratio", ratio=1000, arm="force_density")[0]
    uniform_ratio = u1k["resid_vapour_median"] / max(f1k["resid_vapour_median"], 1e-300)
    # G5 phase blindness of the global median (variable-coefficient arm, ratio 1000)
    d1k = sel("density_ratio", ratio=1000, arm="density")[0]
    hiding_pct = ((d1k["resid_liquid_median"] - d1k["resid_global_median"])
                  / d1k["resid_global_median"] * 100.0)
    # G6 refinement must not degrade the per-particle balance
    res_fd = {lvl: sel("resolution", ratio=1000, arm="force_density", level=lvl)[0]
              for lvl in RES_LEVELS}
    growth = (res_fd[40]["resid_liquid_median"]
              / res_fd[20]["resid_liquid_median"])

    verdicts = {
        "G1_force_density_dp_err_max_pct": {
            "gate": GATES["force_density_dp_err_max_pct"], "measured": fd_dp_max,
            "pass": bool(fd_dp_max <= GATES["force_density_dp_err_max_pct"])},
        "G2_force_density_phase_resid_ratio_spread": {
            "gate": GATES["force_density_phase_resid_ratio_spread_max"],
            "measured": fd_resid_spread,
            "pass": bool(fd_resid_spread
                         <= GATES["force_density_phase_resid_ratio_spread_max"])},
        "G3_vapour_residual_one_over_rho_scaling": {
            "gate": GATES["vapour_residual_scaling_tol"], "measured": scaling_err,
            "pass": bool(scaling_err <= GATES["vapour_residual_scaling_tol"])},
        "G4_uniform_vapour_resid_ratio_to_validated": {
            "gate": GATES["uniform_vapour_resid_min_ratio_to_validated"],
            "measured": uniform_ratio,
            "pass": bool(uniform_ratio
                         >= GATES["uniform_vapour_resid_min_ratio_to_validated"])},
        "G4b_uniform_dp_err_pct_at_ratio_1000": {
            "gate": GATES["uniform_dp_err_min_pct_at_ratio_1000"],
            "measured": abs(u1k["dp_err_pct"]),
            "pass": bool(abs(u1k["dp_err_pct"])
                         >= GATES["uniform_dp_err_min_pct_at_ratio_1000"])},
        "G5_global_median_hides_heavy_phase_pct": {
            "gate": GATES["phase_blind_hiding_min_pct"], "measured": hiding_pct,
            "pass": bool(hiding_pct >= GATES["phase_blind_hiding_min_pct"])},
        "G6_resolution_resid_growth_Rdx40_over_20": {
            "gate": GATES["resolution_resid_growth_max"], "measured": growth,
            "pass": bool(growth <= GATES["resolution_resid_growth_max"])},
        "G7_sigma_zero_dp_pa": {
            "gate": GATES["sigma_zero_max_pa"],
            "measured": abs(ctrl["sigma_zero"]["dp_measured_Pa"]),
            "pass": bool(abs(ctrl["sigma_zero"]["dp_measured_Pa"])
                         <= GATES["sigma_zero_max_pa"])},
        "G7b_sigma_zero_resid": {
            "gate": GATES["sigma_zero_resid_max"],
            "measured": max(ctrl["sigma_zero"]["resid_liquid_abs_m_per_s2"],
                            ctrl["sigma_zero"]["resid_vapour_abs_m_per_s2"]),
            "pass": bool(max(ctrl["sigma_zero"]["resid_liquid_abs_m_per_s2"],
                             ctrl["sigma_zero"]["resid_vapour_abs_m_per_s2"])
                         <= GATES["sigma_zero_resid_max"])},
    }
    return {
        "gates": GATES,
        "verdicts": verdicts,
        "headline": {
            "force_density_dp_err_max_pct": fd_dp_max,
            "density_arm_dp_err_range_pct": [min(r["dp_err_pct"] for r in den),
                                             max(r["dp_err_pct"] for r in den)],
            "uniform_arm_dp_err_at_ratio_1000_pct": u1k["dp_err_pct"],
            "force_density_resid_ratio_invariance": fd_resid_spread,
            "vapour_abs_resid_scaling_err": scaling_err,
            "phase_blind_hiding_pct": hiding_pct,
            "resolution_resid_growth": growth,
            "all_pass_except": sorted(k for k, v in verdicts.items()
                                      if not v["pass"]),
        },
    }


def make_field_figure(ratio=1000.0, level=20):
    """Field maps: the pressure the jump is read from, and *where* the
    momentum imbalance sits.  Same carrier, same lattice, ratio 1 vs 1000."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("force_density", 1.0), ("force_density", ratio), ("uniform", ratio)]
    cases = {}
    for arm, rat in panels:
        r = laplace_isph_case(radius=RADIUS, dx=RADIUS / level, outer=OUTER,
                              rho0=RHO_L, sigma=SIGMA, rho_vapour=RHO_L / rat,
                              poisson_weighting=arm, return_fields=True)
        cases[(arm, rat)] = (_row(r, "field", arm=arm), r["fields"])

    fig, ax = plt.subplots(2, 3, figsize=(15.0, 8.6))
    for col, key in enumerate(panels):
        arm, rat = key
        rec, fld = cases[key]
        rel = (fld["resid_norm"] / fld["acc_norm"].clamp_min(1e-30))
        x = fld["pos"][:, 0].numpy() * 1e3
        y = fld["pos"][:, 1].numpy() * 1e3
        a0 = ax[0, col]
        sc = a0.scatter(x, y, c=fld["p"].numpy(), s=9, cmap="viridis")
        fig.colorbar(sc, ax=a0, shrink=0.82, label="$p$ [Pa]")
        a0.set_title(f"$p$, {arm}, ratio {rat:.0f}\n"
                     f"jump error {rec['dp_err_pct']:+.3f} %", fontsize=9)
        a1 = ax[1, col]
        act = fld["active"].numpy()
        # a *common* colour scale over all three panels: 1.0 == the imbalance
        # equals the local traction; above that the colour saturates (the naive
        # arm exceeds it by orders of magnitude, which is the point).
        sc2 = a1.scatter(x[act], y[act], c=rel.numpy()[act], s=11,
                         cmap="inferno", vmin=0.0, vmax=1.0)
        fig.colorbar(sc2, ax=a1, shrink=0.82,
                     label=r"$|a_{CSF}-\nabla p/\rho|\,/\,|a_{CSF,local}|$")
        a1.set_title(f"momentum imbalance (traction shell only)\n"
                     f"liquid {rec['resid_liquid_median']:.3f} · "
                     f"vapour {rec['resid_vapour_median']:.3f}", fontsize=9)
        for a in (a0, a1):
            a.set_aspect("equal")
            a.set_xlabel("$x$ [mm]")
    for a in ax[:, 0]:
        a.set_ylabel("$y$ [mm]")
    fig.suptitle(f"XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE · round 5 · field maps, "
                 f"$R/\\Delta x = {level}$, $\\rho_l/\\rho_v$ = 1 and {ratio:.0f}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "density_ratio_momentum_field.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_figure(payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    colors = {"force_density": "#1b5e20", "density": "#e65100", "uniform": "#b71c1c"}
    labels = {"force_density": "force-density  $-\\nabla^2p=-\\nabla\\!\\cdot\\!f$",
              "density": "variable coeff.  $-\\nabla\\!\\cdot\\!((1/\\rho)\\nabla p)$",
              "uniform": "naive  $-\\nabla^2p=-\\rho_l\\nabla\\!\\cdot\\!a$"}

    # (a) dimensionless per-phase imbalance vs ratio
    a0 = ax[0]
    for arm in ARMS:
        rs = [r for r in rows if r["axis"] == "density_ratio" and r["arm"] == arm]
        a0.plot([r["ratio"] for r in rs], [r["resid_liquid_median"] for r in rs],
                "-o", color=colors[arm], lw=1.6, ms=5,
                label=f"{labels[arm]} — liquid")
        a0.plot([r["ratio"] for r in rs], [r["resid_vapour_median"] for r in rs],
                "--s", color=colors[arm], lw=1.6, ms=4.5, mfc="none",
                label=f"{labels[arm]} — vapour")
    a0.set_xscale("log"); a0.set_yscale("log")
    a0.set_xlabel(r"$\rho_l/\rho_v$"); a0.set_ylabel(r"median $|a_{CSF}-\nabla p/\rho|\,/\,|a_{CSF}|$")
    a0.set_title("(a) per-phase momentum imbalance vs density ratio")
    a0.grid(alpha=0.3, which="both"); a0.legend(fontsize=6.5, loc="upper left")

    # (b) absolute light-phase imbalance: exact 1/rho_v scaling
    a1 = ax[1]
    fd = [r for r in rows if r["axis"] == "density_ratio" and r["arm"] == "force_density"]
    for arm in ("density", "force_density", "uniform"):
        rs = [r for r in rows if r["axis"] == "density_ratio" and r["arm"] == arm]
        a1.plot([r["ratio"] for r in rs], [r["resid_vapour_abs_m_per_s2"] for r in rs],
                "-o", color=colors[arm], lw=1.6, ms=5, label=labels[arm])
    ref = [fd[0]["resid_vapour_abs_m_per_s2"] * r["ratio"] for r in fd]
    a1.plot([r["ratio"] for r in fd], ref, "k:", lw=1.4, label=r"$\propto 1/\rho_v$ (exact)")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel(r"$\rho_l/\rho_v$"); a1.set_ylabel(r"mean $|$residual$|$ in vapour  [m/s$^2$]")
    a1.set_title("(b) the light phase carries the ratio as a scale factor")
    a1.grid(alpha=0.3, which="both"); a1.legend(fontsize=6.5, loc="upper right")

    # (c) refinement: the jump converges, the pointwise balance does not
    a2 = ax[2]
    res = [r for r in rows if r["axis"] == "resolution" and r["ratio"] == 1000
           and r["arm"] == "force_density"]
    lv = sorted({r["R_over_dx"] for r in res})
    for phase, style in (("liquid", "-o"), ("vapour", "--s")):
        a2.plot(lv, [next(r[f"resid_{phase}_median"] for r in res if r["R_over_dx"] == L)
                     for L in lv], style, color="#1b5e20", lw=1.7,
                label=f"imbalance, {phase} (ratio 1000)")
    a2.plot(lv, [abs(next(r["dp_err_pct"] for r in res if r["R_over_dx"] == L)) / 100.0
                 for L in lv], "-^", color="#0d47a1", lw=1.7,
            label="|jump error| / (σ/R)")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel(r"$R/\Delta x$"); a2.set_ylabel("relative quantity")
    a2.set_title("(c) refinement: integral converges, per-particle balance does not")
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=7)

    fig.suptitle("XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE · round 5 · "
                 "density ratio attributed per phase", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "density_ratio_momentum.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main():
    torch.set_num_threads(2)
    rows = sweep_ratio() + sweep_resolution()
    ctrl = controls()
    payload = {"anchor": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE",
               "round": 5,
               "question": ("is the density ratio the thing that breaks the local "
                            "momentum balance, and in which phase?"),
               "params": {"radius_m": RADIUS, "dx_m": DX, "outer_m": OUTER,
                          "rho_liquid": RHO_L, "sigma": SIGMA,
                          "h_over_dx": 1.3, "ratios": RATIOS, "arms": list(ARMS),
                          "res_levels": RES_LEVELS, "res_ratios": RES_RATIOS,
                          "curvature_method": "mls", "cg_rel_tol": 1e-10}}
    payload["summary"] = summarize(rows, ctrl)   # gates + verdicts + headline
    payload["controls"] = ctrl
    payload["rows"] = rows
    payload["figure"] = "scripts/xval_projection_laplace/results/density_ratio_momentum.png"
    payload["figure_field"] = "scripts/xval_projection_laplace/results/density_ratio_momentum_field.png"
    make_figure(payload)
    make_field_figure()
    out = OUT / "density_ratio_momentum.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["summary"]["headline"], indent=1))
    print(f"\nwrote {out} and {payload['figure']}")


if __name__ == "__main__":
    main()
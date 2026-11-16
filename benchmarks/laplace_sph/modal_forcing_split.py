"""Mode-2 capillary dynamics of the projection-SPH drop: is the well-balanced split admissible?

Why this exists
---------------
Round 3 of the anchor ``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE``
(``docs/xval_projection_laplace_wellbalanced.md``) made a static 2-D drop a fixed
point of the projection-SPH time loop to solver tolerance by feeding the momentum
update only the *discrete gradient part* ``G q`` of the CSF acceleration and
dropping the remainder ``a_sol = a - G q``.  Its own section 5 scopes the claim:

    "Scope is the equilibrium branch.  Neither capillary waves nor droplet
     oscillation are covered."

This script pays that debt with a one-step, analytic-frequency measurement
instead of a long march.

The physics (2-D drop, mode n, both phases of equal density, rigid wall at R_c)
------------------------------------------------------------------------------
Take ``r(theta, t) = R (1 + eps cos n theta)``.  The perimeter excess gives the
surface energy, an irrotational potential flow (``r^n`` inside, a combination
annihilated at the wall outside) gives the kinetic energy:

    KE = rho pi R^2 k^{2n} / (n (k^{2n} - 1)) * etadot^2 ,   k = R_c / R
    PE = sigma pi n^2 R / 2 * eps^2

    =>  omega^2 = sigma n^3 (k^{2n} - 1) / (2 rho R^3 k^{2n})

Two limits are the references used below (``n = 2``):

* ``k -> inf`` (unconfined, equal densities both sides): ``omega^2 = 4 sigma/(rho R^3)``
* ``k = 1.6`` (the shipped carrier: 12 mm of padding on a 20 mm drop):
  ``omega^2 = 3.39 sigma/(rho R^3)``, i.e. 8 % lower in ``omega`` -- the pinned
  carrier layer is a rigid wall, so the confined value is the right reference
  for the shipped set-up and the unconfined one is the upper end of the band.

The measurement (one step, no marching)
---------------------------------------
From rest the scheme does ``u* = dt a_arm`` then ``u = P u*`` (adjoint
projection).  The modal average ``<u . rhat cos 2theta>`` over the near field
must equal ``-omega^2 eps R dt`` (linear theory: ``u_r = etadot cos 2theta``),
which defines a measured frequency per arm:

    omega_meas = sqrt( -<u . rhat cos 2theta> / (eps R dt) ).

Three arms share one lattice, one colour/curvature/force dog and one projector:

``full``          the shipped CSF acceleration (round-1/2 path, i.e. what
                  ``relax_projection_consistent`` integrates)
``irrotational``  ``G q`` -- the round-3 well-balanced arm
``solenoidal``    the discarded remainder ``a - G q``

``full`` is scored against the analytic ``omega``; ``irrotational`` is scored on
how much of ``full``'s modal response survives.  A formulation that preserves
capillary *dynamics* must keep that fraction near 1.

Hypotheses, declared before any number was read
-----------------------------------------------
H1  ``omega_meas(full)`` lies within 15 % of the analytic band at every case: the
    shipped CSF + projection machinery reproduces the linear Rayleigh frequency
    of a 2-D drop mode ``n = 2``.
H2  ``R2 = omega_meas(irrotational) / omega_meas(full) >= 0.95``: the dropped
    solenoidal part carries no mode-2 response, so the round-3 caveat can be
    lifted.  (If instead ``R2 ~ 0`` the split is a *fixed-point* device: because
    ``u* = dt G q`` implies ``p = (rho/dt) q`` exactly, the projector annihilates
    the balanced arm's own force identically and *any* shape -- deformed or not
    -- is a frozen fixed point.  That is then a statement about the formulation,
    not about resolution.)

Gates (declared before the numbers; echoed into the JSON)
---------------------------------------------------------
``frequency_15pct``    |omega(full) vs analytic band| <= 15 % at every case (H1)
``dynamics_retention`` R2 >= 0.95 at every case (H2)
``split_reconstruction`` ||a - (a_irr + a_sol)|| / ||a|| <= 1e-12
``split_orthogonality``  |<a_irr, a_sol>| / ||a||^2 <= 1e-06
``static_control``     on the circle the balanced arm's ``max|u|/v_ref`` <= 1e-5
                       (reproduces the round-3 static result)

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_sph/modal_forcing_split.py --run --level 10
    PYTHONPATH=src python benchmarks/laplace_sph/modal_forcing_split.py --merge --figure
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorsph.laplace_isph import csf_state_two_phase, hex_block
from tensorsph.projection_adjoint import helmholtz_split, project_velocity

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "results_modal"

R_EQ = 0.02                      # equivalent radius, m
SIGMA, RHO0 = 0.072, 1000.0
OUTER = 0.012                    # carrier padding -> wall at k = 1 + OUTER/R = 1.6
HF = 1.3
DT_FACTOR = 0.1
MODE = 2
NEAR = 1.5                       # modal sums restricted to r < NEAR * R_eq

ASPECTS = (1.0, 1.02, 1.05, 1.10, 1.20, 1.50)
LEVELS = (8, 10, 16)

GATES_DECLARED = {
    "frequency_15pct": {"op": "<=", "value": 15.0, "unit": "percent",
                        "scope": "|omega(full) vs analytic band|, every case (H1)"},
    "dynamics_retention": {"op": ">=", "value": 0.95, "unit": "ratio",
                           "scope": "R2 = omega(irrotational)/omega(full), every case (H2)"},
    "split_reconstruction": {"op": "<=", "value": 1e-12, "scope": "every case"},
    "split_orthogonality": {"op": "<=", "value": 1e-06, "scope": "every case"},
    "static_control": {"op": "<=", "value": 1e-05,
                       "scope": "max|u|/v_ref of the balanced arm on the circle"},
}


def omega_theory(radius: float, mode: int = MODE, sigma: float = SIGMA,
                 rho: float = RHO0, k: float | None = None) -> float:
    """``omega^2 = sigma n(n^2-1) (k^{2n}-1) / (2 rho R^3 k^{2n})``; ``k=None`` = unconfined.

    The ``(n^2 - 1)`` (not ``n^2``) comes from the incompressibility constraint:
    the geometric perimeter excess of ``r = R(1+eps cos n theta)`` is
    ``pi n^2 eta^2/(2R)``, but the area is only conserved at O(eta^2) after the
    constant part of the pressure adjusts, which removes exactly one ``n^2``.
    The factor 2 in the denominator is the equal-density carrier: the outer
    irrotational flow carries as much kinetic energy as the inner one (checked
    by the pair of integrals in the module docstring).  Same form as the classic
    Rayleigh value ``sigma n(n^2-1)/(rho R^3)`` for a drop in vacuum, halved.
    """
    if k is None:
        return math.sqrt(sigma * mode * (mode ** 2 - 1) / (2.0 * rho * radius ** 3))
    k2n = k ** (2 * mode)
    return math.sqrt(sigma * mode * (mode ** 2 - 1) * (k2n - 1.0)
                     / (2.0 * rho * radius ** 3 * k2n))


def ellipse_block(aspect: float, dx: float):
    """Triangular lattice block whose *liquid* label is an ellipse of equal area.

    ``a/b = aspect`` with ``pi a b = pi R_eq^2`` keeps the liquid particle count
    (and hence ``R_eq/dx``) fixed across the sweep; only the shape moves.  The
    lattice and carrier block are those of ``laplace_isph.hex_block``.
    """
    a = R_EQ * math.sqrt(aspect)
    b = R_EQ / math.sqrt(aspect)
    pos, mass, _liquid, half = hex_block(a, dx, OUTER, RHO0)
    liquid = (pos[:, 0] / a) ** 2 + (pos[:, 1] / b) ** 2 < 1.0
    return pos, mass, liquid, half, a, b


def _mode_weights(pos):
    r = torch.linalg.vector_norm(pos, dim=1).clamp_min(1e-30)
    rhat = pos / r[:, None]
    c2 = rhat[:, 0] ** 2 - rhat[:, 1] ** 2      # cos 2 theta
    near = r < NEAR * R_EQ
    return rhat, c2, near


def _modal_mean(field, vol, rhat, c2, near):
    """Volume-weighted mean of the ``cos 2theta`` component of ``field . rhat``."""
    w = vol * c2
    num = float((w[near] * (field * rhat).sum(dim=1)[near]).sum())
    den = float((w[near] * c2[near]).sum())
    return num / den


def run_case(aspect: float, level: int, rel_tol: float = 1e-8,
             cg_max_iter: int = 20000, verbose: bool = False):
    t0 = time.time()
    dx = R_EQ / level
    h = HF * dx
    pos, mass, liquid, half, a, b = ellipse_block(aspect, dx)
    vol = mass / RHO0
    n = pos.shape[0]
    t_sigma = math.sqrt(RHO0 * h ** 3 / SIGMA)
    dt = DT_FACTOR * t_sigma
    v_ref = math.sqrt(SIGMA / (RHO0 * R_EQ))
    eps = (a - b) / (a + b)                 # mode amplitude of the ellipse

    st = csf_state_two_phase(pos, mass, liquid, h, RHO0, SIGMA,
                             curvature_method="mls")
    pairs, active = st["pairs"], st["active"]
    a_full = st["acc"]

    edge = 1.6 * dx
    pinned = ((pos[:, 0].abs() > half - edge) | (pos[:, 1].abs() > half - edge))
    gauge = int(torch.nonzero(pinned)[0, 0]) if bool(pinned.any()) else 0

    sp = helmholtz_split(pos, a_full, vol, pairs, h, gauge=gauge,
                         rel_tol=rel_tol, max_iter=cg_max_iter)
    a_irr, a_sol = sp["irrotational"], sp["solenoidal"]

    rhat, c2, near = _mode_weights(pos)

    def projected(a_arm):
        u, p, info = project_velocity(pos, dt * a_arm, vol, pairs, h, RHO0, dt,
                                      gauge=gauge, rel_tol=rel_tol,
                                      max_iter=cg_max_iter)
        return _modal_mean(u, vol, rhat, c2, near) / dt, info, u

    acc_full = _modal_mean(a_full, vol, rhat, c2, near)
    acc_irr = _modal_mean(a_irr, vol, rhat, c2, near)
    acc_sol = _modal_mean(a_sol, vol, rhat, c2, near)

    acc_p_full, i_full, u_full = projected(a_full)
    acc_p_irr, i_irr, u_irr = projected(a_irr)
    acc_p_sol, i_sol, _u_sol = projected(a_sol)

    def omega_from(acc):
        val = -acc / (eps * R_EQ) if eps > 0 else float("nan")
        return math.sqrt(val) if val > 0 else float("nan")

    w_full, w_full_raw = omega_from(acc_p_full), omega_from(acc_full)
    w_irr, w_sol = omega_from(acc_p_irr), omega_from(acc_p_sol)

    k_wall = (a + OUTER) / a
    w_conf = omega_theory(R_EQ, k=k_wall)
    w_unconf = omega_theory(R_EQ, k=None)

    rec = {
        "aspect_a_over_b": float(aspect),
        "R_eq_over_dx": float(level),
        "dx_m": dx, "h_m": h, "dt_s": dt, "t_sigma_s": t_sigma,
        "semi_axis_a_m": a, "semi_axis_b_m": b,
        "mode_amplitude_eps": eps, "wall_ratio_k": k_wall,
        "n_particles": int(n), "n_liquid": int(liquid.sum()),
        "n_active": int(active.sum()), "n_near": int(near.sum()),
        "omega_theory_wall_rad_s": w_conf,
        "omega_theory_unconf_rad_s": w_unconf,
        "omega_meas_full_rad_s": w_full,
        "omega_meas_full_unprojected_rad_s": w_full_raw,
        "omega_meas_irrotational_rad_s": w_irr,
        "omega_meas_solenoidal_rad_s": w_sol,
        "modal_accel_full_m_s2": acc_full,
        "modal_accel_irrotational_m_s2": acc_irr,
        "modal_accel_solenoidal_m_s2": acc_sol,
        "modal_accel_full_projected_m_s2": acc_p_full,
        "modal_accel_irrotational_projected_m_s2": acc_p_irr,
        "v2_full_over_vref": _modal_mean(u_full, vol, rhat, c2, near) / v_ref,
        "v2_irrotational_over_vref": _modal_mean(u_irr, vol, rhat, c2, near) / v_ref,
        "umax_over_vref_full": float(torch.linalg.vector_norm(u_full, dim=1).max() / v_ref),
        "umax_over_vref_irrotational": float(torch.linalg.vector_norm(u_irr, dim=1).max() / v_ref),
        "solenoidal_fraction": sp["solenoidal_fraction"],
        "reconstruction_defect": sp["reconstruction_defect"],
        "orthogonality": sp["orthogonality"],
        "div_after_over_div_before_full": i_full["div_after_over_div_before"],
        "div_after_over_div_before_irrotational": i_irr["div_after_over_div_before"],
        "cg_iterations": {"split": sp["cg_iterations"],
                          "project_full": i_full["cg_iterations"],
                          "project_irrotational": i_irr["cg_iterations"],
                          "project_solenoidal": i_sol["cg_iterations"]},
        "wall_s": time.time() - t0,
    }
    rec["R2_velocity"] = (abs(rec["v2_irrotational_over_vref"])
                          / abs(rec["v2_full_over_vref"])
                          if abs(rec["v2_full_over_vref"]) > 0 else float("nan"))
    rec["R2_frequency"] = (w_irr / w_full
                           if (w_full == w_full and w_full > 0) else float("nan"))
    rec["omega_full_vs_wall_pct"] = (w_full / w_conf - 1.0) * 100.0
    rec["omega_full_vs_unconf_pct"] = (w_full / w_unconf - 1.0) * 100.0
    rec["omega_irr_vs_wall_pct"] = (w_irr / w_conf - 1.0) * 100.0
    if verbose:
        print(json.dumps({k: rec[k] for k in (
            "aspect_a_over_b", "R_eq_over_dx", "solenoidal_fraction",
            "omega_theory_wall_rad_s", "omega_meas_full_rad_s",
            "omega_meas_irrotational_rad_s", "omega_full_vs_wall_pct",
            "omega_full_vs_unconf_pct", "R2_velocity")}, indent=1))
    return rec


def summarise(runs):
    def worst(key):
        vals = [abs(r[key]) for r in runs if r[key] == r[key]]
        return max(vals) if vals else float("nan")

    # the circle (eps = 0) is the *static* control, not a frequency case: its
    # mode-2 frequency is undefined and it is scored only by ``static_control``.
    deform = [r for r in runs if r["mode_amplitude_eps"] > 1e-9]
    band_err = max(min(abs(r["omega_full_vs_wall_pct"]),
                       abs(r["omega_full_vs_unconf_pct"])) for r in deform)
    ret = min(r["R2_velocity"] for r in deform)
    circle = [r for r in runs if abs(r["aspect_a_over_b"] - 1.0) < 1e-12]
    circle_res = max((r["umax_over_vref_irrotational"] for r in circle),
                     default=float("nan"))

    def gate(name, measured, threshold, op):
        ok = bool(measured <= threshold) if op == "<=" else bool(measured >= threshold)
        return {"gate": name, "measured": measured, "threshold": threshold,
                "op": op, "passed": ok}

    gates = [
        gate("frequency_15pct", band_err, 15.0, "<="),
        gate("dynamics_retention", ret, 0.95, ">="),
        gate("split_reconstruction", worst("reconstruction_defect"), 1e-12, "<="),
        gate("split_orthogonality", worst("orthogonality"), 1e-6, "<="),
        gate("static_control", circle_res, 1e-5, "<="),
    ]
    retention_vs_eps = {}
    for lv in sorted({r["R_eq_over_dx"] for r in runs}):
        pts = sorted((r["mode_amplitude_eps"], r["R2_velocity"])
                     for r in runs if r["R_eq_over_dx"] == lv)
        retention_vs_eps[str(int(lv))] = [[float(x), float(y)] for x, y in pts]
    return {
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates),
        "h1_frequency_ok": bool(band_err <= 15.0),
        "h2_retention_ok": bool(ret >= 0.95),
        "worst_frequency_err_vs_band_pct": band_err,
        "n_deformed_cases": len(deform),
        "worst_frequency_err_vs_wall_pct": worst("omega_full_vs_wall_pct"),
        "worst_frequency_err_vs_unconf_pct": worst("omega_full_vs_unconf_pct"),
        "min_dynamics_retention": ret,
        "retention_vs_eps": retention_vs_eps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aspect", type=float, default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.run:
        aspects = (args.aspect,) if args.aspect is not None else ASPECTS
        levels = (args.level,) if args.level is not None else LEVELS
        for level in levels:
            for aspect in aspects:
                rec = run_case(aspect, level, verbose=True)
                (OUTDIR / f"a{aspect:g}_n{level}.json").write_text(
                    json.dumps(rec, indent=1))
        return

    if args.merge or args.figure:
        runs = [json.loads(p.read_text()) for p in sorted(OUTDIR.glob("a*_n*.json"))]
        out = {
            "case": "XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE / mode-2 dynamics of the well-balanced split",
            "mode": MODE, "near_field_R_over_Req": NEAR,
            "setup": {"R_eq_m": R_EQ, "sigma_N_per_m": SIGMA, "rho0_kg_m3": RHO0,
                      "outer_m": OUTER, "h_over_dx": HF, "dt_over_t_sigma": DT_FACTOR,
                      "aspects": sorted({r["aspect_a_over_b"] for r in runs}),
                      "levels": sorted({r["R_eq_over_dx"] for r in runs})},
            "gates_declared": GATES_DECLARED,
            "runs": runs,
        }
        out.update(summarise(runs))
        (OUTDIR / "results_modal.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items()
                          if k in ("gates", "all_passed", "h1_frequency_ok",
                                   "h2_retention_ok",
                                   "worst_frequency_err_vs_band_pct",
                                   "min_dynamics_retention")}, indent=1))
        if args.figure:
            make_figure(out)


def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = res["runs"]
    levels = sorted({r["R_eq_over_dx"] for r in runs})
    fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.1))

    for lv in levels:
        sub = sorted([r for r in runs if r["R_eq_over_dx"] == lv],
                     key=lambda r: r["aspect_a_over_b"])
        xs = [r["aspect_a_over_b"] for r in sub]
        ax[0].plot(xs, [r["omega_meas_full_rad_s"] for r in sub], "o-",
                   label=f"full arm, R/dx = {lv:g}")
    first = sorted([r for r in runs if r["R_eq_over_dx"] == levels[0]],
                   key=lambda r: r["aspect_a_over_b"])
    ax[0].plot([r["aspect_a_over_b"] for r in first],
               [r["omega_theory_wall_rad_s"] for r in first], "k-",
               label="analytic, rigid wall at k=1.6")
    ax[0].plot([r["aspect_a_over_b"] for r in first],
               [r["omega_theory_unconf_rad_s"] for r in first], "k--",
               label="analytic, unconfined (band top)")
    ax[0].set_xlabel("aspect a/b")
    ax[0].set_ylabel(r"mode-2 $\omega$ [rad/s]")
    ax[0].set_title(r"(a) shipped arm vs analytic Rayleigh $\omega_2$")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    for lv in levels:
        sub = sorted([r for r in runs if r["R_eq_over_dx"] == lv],
                     key=lambda r: r["aspect_a_over_b"])
        xs = [r["aspect_a_over_b"] for r in sub]
        ax[1].semilogy(xs, [max(r["R2_velocity"], 1e-12) for r in sub], "o-",
                       label=f"R/dx = {lv:g}")
    ax[1].axhline(0.95, color="r", ls=":", label="H2 gate 0.95")
    ax[1].axhline(1e-7, color="gray", ls="-.", label="round-off floor")
    ax[1].set_xlabel("aspect a/b")
    ax[1].set_ylabel(r"$R_2$ = $\omega$(balanced) / $\omega$(full)")
    ax[1].set_title("(b) capillary dynamics retained by the balanced arm")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")

    sub = sorted([r for r in runs if r["R_eq_over_dx"] == levels[0]],
                 key=lambda r: r["aspect_a_over_b"])
    width = 0.35
    idx = list(range(len(sub)))
    ax[2].bar([i - width / 2 for i in idx],
              [r["solenoidal_fraction"] for r in sub], width,
              label=r"$\|a_{sol}\|/\|a\|$ (dropped)")
    ax[2].bar([i + width / 2 for i in idx],
              [1.0 - min(r["R2_velocity"], 1.0) for r in sub], width,
              label=r"$1 - R_2$ (dropped dynamics)")
    ax[2].set_xticks(idx)
    ax[2].set_xticklabels([f"{r['aspect_a_over_b']:g}" for r in sub])
    ax[2].set_xlabel("aspect a/b")
    ax[2].set_title(f"(c) force dropped vs dynamics dropped, R/dx = {levels[0]:g}")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUTDIR / "modal_transferability.png", dpi=140)
    print("wrote", OUTDIR / "modal_transferability.png")


if __name__ == "__main__":
    main()
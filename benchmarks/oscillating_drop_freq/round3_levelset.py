"""Round 3 of ``CASE-SPH-OSC-DROP-FREQ``: level-set interface reconstruction.

Round 2 left the case at honest FAIL with two attributed defects (anisotropic
colour-gradient surface measure, curvature with no deformation response) and one
*claimed* pass (``delta_exact``, ``-0.41 %`` at ``R/dx = 8``).  This round does
three things, in the order they matter:

A. **convention audit** -- the projection round 2 used for every row counts the
   modal traction twice: it keeps the equilibrium curvature ``2 sigma/R`` while
   projecting on the *deformed* measure, and it does not subtract the base
   curvature before projecting.  Both give an ``O(a)`` contribution of the same
   size as the physics, of *opposite* sign to the ``delta kappa`` term only in
   appearance, so the sum is ``2x`` the correct stiffness.  A surface built from
   *analytically exact* mode-l coefficients is fed through the old and the new
   projection: the old one reports ``+41.6 %`` frequency error on an error-free
   geometry, the new one ``< 1 %``.  Round 2's single passing row is re-scored
   under the corrected convention -- it was a ``2x`` over-count cancelling a
   ~30 % amplitude under-recovery.

B. **level-set ladder** -- the colour field ``c = sum_j V_j W`` is evaluated
   *between* particles and followed to ``c = 1/2`` along 240 rays, the resulting
   ``r(theta)`` is fitted with 7 Legendre coefficients, and area/curvature are
   taken analytically on the reconstructed surface
   (:mod:`tensorsph.interface_level_set`).  Modes 2 and 3, ``eps`` in
   ``{0, 0.02, 0.05}``, ``R/dx`` in ``{8 ... 32}``.

C. **orientation ensemble** -- the physics is axisymmetric, so the answer may not
   depend on how the hex lattice sits relative to the drop axis.  12 rotations
   per point give the mean (the answer) and the scatter (what a single
   realisation would have misled you with).

Output: ``round3.json`` (every number in ``docs/case_sph_osc_drop_freq_round3.md``
is rendered from it).

Run: ``PYTHONPATH=src python benchmarks/oscillating_drop_freq/round3_levelset.py``
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from tensorsph.interface_level_set import (  # noqa: E402
    measure_revolution, modal_capillary_stress, reconstructed_surface,
)
from tensorsph.oscillating_drop3d import (  # noqa: E402
    axisym_drop_geometry, exact_modal_stress, rayleigh_frequency_3d,
)
from tensorsph.surface_tension_dispersion import csf_state  # noqa: E402

from ensemble import axisym_geometry_rotated  # noqa: E402
from interface_reconstruction import (  # noqa: E402
    fit_interface, reconstructed_fields,
)

OUT = HERE / "round3.json"
R_DEFAULT, SIGMA_DEFAULT, RHO_DEFAULT = 0.02, 7.2, 1000.0
GATE_PCT = 6.0


def _omega_error(dp: float, dp_exact: float) -> dict:
    if dp_exact == 0.0:
        return {"dp": dp, "dp_exact": dp_exact, "stiffness_ratio": None,
                "omega_error_pct": None}
    ratio = dp / dp_exact
    omega_ratio = math.copysign(math.sqrt(abs(ratio)), ratio)
    return {"dp": dp, "dp_exact": dp_exact, "stiffness_ratio": ratio,
            "omega_ratio": omega_ratio, "omega_error_pct": (omega_ratio - 1.0) * 100.0}


# --------------------------------------------------------------------------- #
# A. how much of round 2's verdict was a projection convention?
# --------------------------------------------------------------------------- #
def convention_audit(radius: float, sigma: float, rho: float) -> dict:
    """Score an *exact* mode-l surface with the old and the corrected projection."""
    rows = []
    for mode in (2, 3):
        for eps in (0.01, 0.02, 0.05):
            coef = torch.zeros(7, dtype=torch.float64)
            coef[0] = radius
            coef[mode] = eps * radius
            surf = reconstructed_surface(coef, 6, radius=radius)
            dp_exact = exact_modal_stress(sigma, radius, mode, eps)
            old = modal_capillary_stress(surf, sigma=sigma, radius=radius, mode=mode,
                                         subtract_base_curvature=False,
                                         reference_measure=False)
            new = modal_capillary_stress(surf, sigma=sigma, radius=radius, mode=mode)
            rows.append({"surface": "analytic (coef == seeded, no measurement)",
                         "mode": mode, "eps": eps,
                         "old": _omega_error(old["dp"], dp_exact),
                         "new": _omega_error(new["dp"], dp_exact),
                         "ratio_old_over_new": old["dp"] / new["dp"]})
    return rows


def round2_rescore(radius: float, sigma: float, rho: float, *, kmax: int = 6,
                   ladder=(8.0, 12.0, 16.0), mode: int = 2,
                   eps: float = 0.02) -> list:
    """Round 2's ``delta_exact`` route, scored with both projections.

    The route is: ``delta_s``-weighted Legendre fit of the *particle radii*
    (``fit_interface``) then analytic curvature of the fitted surface.  Round 2
    reported this route's frequency error as ``-0.41 %`` at ``R/dx = 8``.
    """
    rows = []
    for rdx in ladder:
        geom = axisym_drop_geometry(radius, radius / rdx, mode=mode, eps=eps, rho0=rho)
        state, _, _ = csf_state(geom, sigma, periodic_x=False)
        fit = fit_interface(geom, state, kmax=kmax)
        surf = reconstructed_fields(geom, fit)
        # reconstructed_fields returns the same grid the new operator expects
        surf = {"theta": surf["theta"], "cos_theta": surf["cos_theta"],
                "r": surf["r"], "dl": surf["dl"], "darea": surf["darea"],
                "kappa": surf["kappa"], "radius": radius}
        dp_exact = exact_modal_stress(sigma, radius, mode, eps)
        old = modal_capillary_stress(surf, sigma=sigma, radius=radius, mode=mode,
                                     subtract_base_curvature=False,
                                     reference_measure=False)
        new = modal_capillary_stress(surf, sigma=sigma, radius=radius, mode=mode,
                                     reference_radius=float(fit["coef"][0]))
        rows.append({"route": "round2 delta_exact", "radius_over_dx": rdx,
                     "n_particles": geom["n_particles"],
                     "amplitude_recovery": float(fit["coef"][mode]) / radius / eps,
                     "old": _omega_error(old["dp"], dp_exact),
                     "new": _omega_error(new["dp"], dp_exact)})
        print(f"  [A2] rdx={rdx:4.0f} amp_rec={rows[-1]['amplitude_recovery']:+.3f} "
              f"old={rows[-1]['old']['omega_error_pct']:+.2f}% "
              f"new={rows[-1]['new']['omega_error_pct']:+.2f}%", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# B. level-set ladder
# --------------------------------------------------------------------------- #
def level_set_ladder(radius: float, sigma: float, rho: float, *, kmax: int,
                     modes, ladder, eps_list, n_rays: int, n_radial: int) -> list:
    rows = []
    for mode in modes:
        for rdx in ladder:
            for eps in eps_list:
                geom = _geometry(radius, rdx, mode, eps, rho, psi=0.0)
                m = measure_revolution(geom, mode=mode, sigma=sigma, kmax=kmax,
                                       n_rays=n_rays, n_radial=n_radial)
                res = {
                    "mode": mode, "radius_over_dx": rdx, "eps": eps,
                    "psi_deg": 0.0, "n_particles": geom["n_particles"],
                    "amplitude": m["amplitude"],
                    "amplitude_recovery": m.get("amplitude_recovery"),
                    "rms_amplitude": m["rms_amplitude"],
                    "rms_amplitude_recovery": m.get("rms_amplitude_recovery"),
                    "level_set_offset": m["level_set_offset"],
                    "area_ratio": m["area_ratio"],
                    "kappaR_mean": m["kappaR_mean"],
                    "rms_residual": m["rms_residual"],
                    "crossed_fraction": m["crossed_fraction"],
                    "omega_analytic": rayleigh_frequency_3d(sigma, rho, radius, mode),
                }
                if eps > 0.0:
                    res.update(_omega_error(m["dp"],
                                            exact_modal_stress(sigma, radius, mode, eps)))
                else:
                    res["dp"] = m["dp"]
                    res["dp_exact"] = 0.0
                rows.append(res)
                werr = res.get("omega_error_pct")
                print(f"  [B] l={mode} rdx={rdx:4.0f} eps={eps:<5g} "
                      f"N={res['n_particles']:5d} amp={res['amplitude']:+.5f} "
                      f"rec={(res['amplitude_recovery'] or float('nan')):+.3f} "
                      f"werr={(werr if werr is not None else float('nan')):+7.2f}% "
                      f"area={res['area_ratio']:.4f}", flush=True)
    return rows


def _geometry(radius: float, rdx: float, mode: int, eps: float, rho: float,
              psi: float) -> dict:
    dx = radius / rdx
    if psi == 0.0:
        return axisym_drop_geometry(radius, dx, mode=mode, eps=eps, rho0=rho)
    return axisym_geometry_rotated(radius, dx, mode=mode, eps=eps, rho0=rho, psi=psi)


def orientation_ensemble(radius: float, sigma: float, rho: float, *, kmax: int,
                         modes, ladder, eps: float, n_psi: int,
                         n_rays: int, n_radial: int) -> list:
    rows = []
    for mode in modes:
        dp_exact = exact_modal_stress(sigma, radius, mode, eps)
        for rdx in ladder:
            amps, werrs = [], []
            for k in range(n_psi):
                psi = math.pi * k / n_psi
                geom = _geometry(radius, rdx, mode, eps, rho, psi=psi)
                m = measure_revolution(geom, mode=mode, sigma=sigma, kmax=kmax,
                                       n_rays=n_rays, n_radial=n_radial)
                amps.append(m["amplitude"])
                w = _omega_error(m["dp"], dp_exact)["omega_error_pct"]
                werrs.append(w)
            a = torch.tensor(amps, dtype=torch.float64)
            w = torch.tensor(werrs, dtype=torch.float64)
            row = {"mode": mode, "radius_over_dx": rdx, "eps": eps, "n_psi": n_psi,
                   "amplitude_mean": float(a.mean()), "amplitude_std": float(a.std()),
                   "amplitude_recovery_mean": float(a.mean()) / eps,
                   "amplitude_recovery_std": float(a.std()) / eps,
                   "omega_error_mean_pct": float(w.mean()),
                   "omega_error_std_pct": float(w.std()),
                   "n_passing": int((w.abs() < GATE_PCT).sum()),
                   "min_abs_omega_error_pct": float(w.abs().min()),
                   "max_abs_omega_error_pct": float(w.abs().max())}
            rows.append(row)
            print(f"  [C] l={mode} rdx={rdx:4.0f} n_psi={n_psi} "
                  f"rec={row['amplitude_recovery_mean']:+.3f}"
                  f"±{row['amplitude_recovery_std']:.3f} "
                  f"werr={row['omega_error_mean_pct']:+6.2f}"
                  f"±{row['omega_error_std_pct']:.2f}% "
                  f"pass {row['n_passing']}/{n_psi}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# D. transfer law and verdict
# --------------------------------------------------------------------------- #
def transfer_law(rows: list) -> dict:
    """Fit ``amplitude_recovery = 1 - a / (R/dx)`` on the clean (l=3) rows."""
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        fits = {}
        for eps in sorted({r["eps"] for r in rows if r["mode"] == mode and r["eps"] > 0}):
            pts = [(r["radius_over_dx"], r["amplitude_recovery"]) for r in rows
                   if r["mode"] == mode and r["eps"] == eps and r["amplitude_recovery"]]
            if len(pts) < 3:
                continue
            x = torch.tensor([1.0 / p[0] for p in pts], dtype=torch.float64)
            y = torch.tensor([p[1] for p in pts], dtype=torch.float64)
            xm, ym = x.mean(), y.mean()
            slope = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
            icpt = float(ym - slope * xm)
            resid = y - (icpt + slope * x)
            fits[eps] = {"intercept": icpt, "deficit_coefficient": -slope,
                         "max_abs_residual": float(resid.abs().max())}
        out[mode] = fits
    # gate crossing from the clean law: sqrt(1 - a/x) - 1 = -gate  =>  x = a/(1-(1-gate)^2)
    clean = out.get(3, {}).get(0.02) or out.get(3, {}).get(0.05)
    if clean:
        a = clean["deficit_coefficient"]
        x_gate = a / (1.0 - (1.0 - GATE_PCT / 100.0) ** 2)
        out["predicted_gate_radius_over_dx"] = x_gate
        out["law"] = f"amplitude_recovery = 1 - {a:.3f}/(R/dx), i.e. omega deficit {100*(1-math.sqrt(1-a/x_gate)):.1f}% at the crossing"
    return out


def verdict(ladder_rows: list, ens_rows: list) -> dict:
    def worst(rows, key="omega_error_pct"):
        vals = [abs(r[key]) for r in rows if r.get(key) is not None]
        return max(vals) if vals else None
    out = {"gate_pct": GATE_PCT, "single_realisation": {}, "ensemble": {}}
    for mode in (2, 3):
        for eps in (0.02, 0.05):
            sel = [r for r in ladder_rows if r["mode"] == mode and r["eps"] == eps
                   and r.get("omega_error_pct") is not None]
            if sel:
                out["single_realisation"][f"l{mode}_eps{eps}"] = {
                    "worst_abs_pct": worst(sel),
                    "resolutions_passing": [r["radius_over_dx"] for r in sel
                                            if abs(r["omega_error_pct"]) < GATE_PCT],
                    "first_resolution_with_monotone_pass":
                        min([r["radius_over_dx"] for r in sel
                             if abs(r["omega_error_pct"]) < GATE_PCT], default=None),
                }
    for r in ens_rows:
        key = f"l{r['mode']}_rdx{int(r['radius_over_dx'])}"
        out["ensemble"][key] = {"omega_error_mean_pct": r["omega_error_mean_pct"],
                                "omega_error_std_pct": r["omega_error_std_pct"],
                                "passing": abs(r["omega_error_mean_pct"]) < GATE_PCT,
                                "n_single_realisations_passing": r["n_passing"],
                                "n_psi": r["n_psi"]}
    passing = [k for k, v in out["ensemble"].items() if v["passing"]]
    out["ensemble_passing_rows"] = passing
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=R_DEFAULT)
    ap.add_argument("--sigma", type=float, default=SIGMA_DEFAULT)
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--rays", type=int, default=240)
    ap.add_argument("--radial", type=int, default=81)
    ap.add_argument("--n-psi", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--quick", action="store_true",
                    help="short ladder (no ensemble), for a smoke check")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    ladder = (8.0, 12.0, 16.0, 20.0, 24.0, 32.0) if not args.quick else (12.0, 20.0)
    modes = (2, 3)
    eps_list = (0.0, 0.02, 0.05)

    print("[A1] convention audit on an error-free analytic surface", flush=True)
    audit = convention_audit(args.radius, args.sigma, args.rho)
    for r in audit:
        print(f"  [A1] l={r['mode']} eps={r['eps']:<5g} "
              f"old={r['old']['omega_error_pct']:+.2f}% "
              f"new={r['new']['omega_error_pct']:+.2f}% "
              f"old/new={r['ratio_old_over_new']:.3f}", flush=True)

    print("[A2] round 2 delta_exact re-scored", flush=True)
    rescore = round2_rescore(args.radius, args.sigma, args.rho)

    print("[B] level-set ladder", flush=True)
    lad = level_set_ladder(args.radius, args.sigma, args.rho, kmax=args.kmax,
                           modes=modes, ladder=ladder, eps_list=eps_list,
                           n_rays=args.rays, n_radial=args.radial)

    ens = []
    if not args.quick:
        print("[C] orientation ensemble", flush=True)
        ens = orientation_ensemble(args.radius, args.sigma, args.rho, kmax=args.kmax,
                                   modes=modes, ladder=(16.0, 20.0, 24.0), eps=0.02,
                                   n_psi=args.n_psi, n_rays=args.rays,
                                   n_radial=args.radial)

    law = transfer_law(lad)
    ver = verdict(lad, ens)
    doc = {
        "case": "SPH-OSC-DROP-FREQ", "round": 3,
        "study": "level-set interface reconstruction + projection convention audit",
        "reference": "Rayleigh 1879, Proc. R. Soc. Lond. 29:71; Lamb 1932 Art. 275",
        "gate_pct": GATE_PCT,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "device": "cpu", "threads": args.threads},
        "parameters": {"radius_m": args.radius, "sigma_N_m": args.sigma,
                       "rho_kg_m3": args.rho, "kmax": args.kmax,
                       "n_rays": args.rays, "n_radial": args.radial,
                       "wall_seconds": round(time.time() - t0, 1)},
        "convention_audit": audit,
        "round2_rescore": rescore,
        "ladder": lad,
        "ensemble": ens,
        "transfer_law": law,
        "verdict": ver,
    }
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"\nwrote {args.out} in {doc['parameters']['wall_seconds']} s")
    print(f"transfer law: {law.get('law')}")
    print(f"predicted gate crossing at R/dx = {law.get('predicted_gate_radius_over_dx')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
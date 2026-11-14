"""Interface-reconstruction capillary operator for the axisymmetric drop.

Why this file exists
--------------------
:mod:`benchmarks.oscillating_drop_freq.diagnose` shows that the two surface
measures available in the repository at ``R/dx <= 20`` neither of them can
resolve the ``l = 2`` restoring force of a lattice drop:

* the colour-gradient CSF delta (``f = -sigma kappa delta_s``) gives an
  axisymmetric surface measure of ``0.62`` (of ``4 pi R^2``) with a strong
  equator/pole anisotropy whose ``P_2`` projection is of the same order as the
  seeded perturbation;
* an interface-*marker* quadrature over the same shell has a polyline arc length
  ``dl / (2 pi R) = 1.64 .. 1.78`` -- the extracted "curve" is the lattice
  staircase (a band of radial jitter of order ``dx``), so its measure is unusable
  as well.

Both are *sub-particle* representations of the interface, and both fail for the
same reason: the interface of a hex lattice cut by a sphere carries a radial
staircase of amplitude ``O(dx)``, which is ``O(eps R)`` at the amplitudes the
Rayleigh gate is posed at.

This module therefore reconstructs the interface *as a surface* before measuring
anything: the shell particles of the CSF state are fitted with a truncated
Legendre series in ``cos theta``,

    r(theta) = sum_{k=0..K} c_k P_k(cos theta),

and the capillary traction is then integrated **analytically along the
reconstructed curve**,

    dA       = 2 pi r sin(theta) sqrt(r^2 + r'^2) dtheta          (surface of revolution)
    kappa_m  = (r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^{3/2}        (meridional curvature)
    n_rho    = sin(theta) - (r'/r) cos(theta)                     (rho component of n)
    kappa    = kappa_m + n_rho / rho,      rho = r sin(theta)      (div n in cylindrical)

so that the surface measure and the curvature are those of one smooth surface
instead of those of ``N`` staircase particles.  This is the standard cure used by
interface-reconstruction surface-tension models (Cummins, Francois & Kothe 2005;
Marchandise et al. 2007 for the curvature fit), applied here in the ``m = 0``
reduction where it is cheap enough to be exact.

The fit itself is the discretisation: its amplitude error on mode ``l`` is what
limits the frequency.  ``fit_noise_floor`` measures that error directly, by
reconstructing an *undeformed* (``eps = 0``) drop and reporting the ``P_l``
coefficient the reconstruction invents out of lattice staircase alone.

Usage::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/interface_reconstruction.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorsph.oscillating_drop3d import (  # noqa: E402
    axisym_drop_geometry,
    exact_modal_stress,
    legendre_p,
    rayleigh_frequency_3d,
)
from tensorsph.surface_tension_dispersion import csf_state  # noqa: E402

OUT = Path(__file__).resolve().parent
GATE_PCT = 6.0
R_DEFAULT, SIGMA_DEFAULT, RHO_DEFAULT = 0.02, 7.2, 1000.0


def legendre_basis(kmax: int, c: torch.Tensor) -> torch.Tensor:
    """``[P_0(c), ..., P_kmax(c)]`` as an ``(n, kmax+1)`` matrix."""
    return torch.stack([legendre_p(k, c) for k in range(kmax + 1)], dim=1)


def fit_interface(geom: dict, state, *, kmax: int = 6,
                  cutoff_frac: float = 0.05) -> dict:
    """Weighted Legendre least squares for ``r(theta)`` on the CSF shell.

    The weights are ``delta_s`` (the surface measure itself), so the fit is a
    surface-weighted mean of the interface and particles in the interior tail of
    the delta function do not drag the fit inward.
    """
    ds = state.delta_s
    keep = ds > cutoff_frac * float(ds.max())
    pos = geom["pos"][keep]
    w = ds[keep]
    rad = torch.linalg.vector_norm(pos, dim=1)
    c = pos[:, 1] / rad.clamp_min(1e-30)
    basis = legendre_basis(kmax, c)
    sw = w / w.sum()
    ata = (basis * sw[:, None]).T @ basis
    atb = (basis * sw[:, None]).T @ rad
    coef = torch.linalg.solve(ata + 1e-12 * torch.eye(kmax + 1, dtype=ata.dtype), atb)
    resid = rad - basis @ coef
    return {"coef": coef, "kmax": kmax, "n_surface": int(keep.sum()),
            "rms_residual_over_R": float((resid * sw).pow(2).sum().sqrt())
            / geom["radius"]}


def fit_interface_color(geom: dict, state, *, kmax: int = 6,
                        n_sectors: int | None = None) -> dict:
    """Sub-particle interface reconstruction: ``r`` where the colour crosses 1/2.

    The colour of :func:`tensorsph.csf_consistent.csf_fields` is a kernel sum that
    decays from 1 inside the fluid to 0 in the void, so the interface is the
    level set ``c = 1/2``.  Its crossing along a radial ray is obtained by linear
    interpolation between two *adjacent lattice particles*, which makes the
    reconstructed radius sub-particle accurate instead of quantised to ``dx``
    (the quantisation that inflates the marker arc length by 1.6-1.8x in
    ``round2.py`` and biases the delta-weighted fit inward).

    Rays are the angular sectors of the meridian, ``theta = atan2(rho, z)``.
    """
    pos = geom["pos"]
    rad = torch.linalg.vector_norm(pos, dim=1)
    rho = geom["rho"]
    theta = torch.atan2(rho, pos[:, 1])                        # 0 .. pi (rho >= 0)
    col = state.color
    if n_sectors is None:
        n_shell = int((state.delta_s > 0.05 * float(state.delta_s.max())).sum())
        n_sectors = int(min(512, max(2 * kmax + 6, n_shell // 4)))
    edges = torch.linspace(0.0, math.pi, n_sectors + 1, dtype=pos.dtype)
    bin_id = torch.clamp(torch.bucketize(theta, edges) - 1, 0, n_sectors - 1)
    ths, rs = [], []
    for b in range(n_sectors):
        m = bin_id == b
        if int(m.sum()) < 3:
            continue
        r_b = rad[m]
        c_b = col[m]
        order = torch.argsort(r_b)
        r_s = r_b[order]
        c_s = c_b[order]
        # first outward crossing of 0.5; on a lattice the outermost particle
        # centre sits ~0.5 dx inside the interface and still carries c ~ 0.59,
        # so where no particle reaches 0.5 the crossing is extrapolated from the
        # outer pair instead of being dropped (otherwise the reconstruction is
        # biased inward by half a cell, i.e. 6 % of R at R/dx = 8).
        below = c_s < 0.5
        if bool(below[0]):
            continue
        if not bool(below.any()):
            if c_s.numel() < 2:
                continue
            c0, c1 = float(c_s[-2]), float(c_s[-1])
            r0, r1 = float(r_s[-2]), float(r_s[-1])
            if c0 <= 0.5 or c0 == c1:
                continue
            ths.append(0.5 * float(edges[b] + edges[b + 1]))
            rs.append(r1 + (c1 - 0.5) / (c0 - c1) * (r1 - r0))
            continue
        j = int(torch.nonzero(below, as_tuple=False)[0])
        c0, c1 = float(c_s[j - 1]), float(c_s[j])
        r0, r1 = float(r_s[j - 1]), float(r_s[j])
        frac = (c0 - 0.5) / (c0 - c1) if c0 != c1 else 0.5
        ths.append(0.5 * float(edges[b] + edges[b + 1]))
        rs.append(r0 + frac * (r1 - r0))
    if len(rs) < kmax + 2:
        raise RuntimeError("colour reconstruction found too few crossings")
    theta_fit = torch.tensor(ths, dtype=pos.dtype)
    r_fit = torch.tensor(rs, dtype=pos.dtype)
    basis = legendre_basis(kmax, torch.cos(theta_fit))
    coef = torch.linalg.lstsq(basis, r_fit.unsqueeze(1)).solution.squeeze(1)
    resid = r_fit - basis @ coef
    return {"coef": coef, "kmax": kmax, "n_surface": len(rs),
            "rms_residual_over_R": float(resid.pow(2).mean().sqrt()) / geom["radius"]}


def reconstructed_fields(geom: dict, fit: dict, *, n_theta: int = 4001) -> dict:
    """Evaluate ``r, r', r'', kappa, dA`` on a fine ``theta`` grid."""
    theta = torch.linspace(1.0e-6, math.pi - 1.0e-6, n_theta, dtype=torch.float64)
    c = torch.cos(theta)
    basis = legendre_basis(fit["kmax"], c)
    r = basis @ fit["coef"]
    dth = float(theta[1] - theta[0])
    rp = torch.gradient(r, spacing=dth)[0]
    rpp = torch.gradient(rp, spacing=dth)[0]
    rho = r * torch.sin(theta)
    dl = torch.sqrt(r ** 2 + rp ** 2)
    darea = 2.0 * math.pi * rho * dl * dth
    kappa_m = (r ** 2 + 2.0 * rp ** 2 - r * rpp) / (r ** 2 + rp ** 2) ** 1.5
    n_rho = torch.sin(theta) - (rp / r) * torch.cos(theta)
    kappa_az = n_rho / rho.clamp_min(1e-30)
    return {"theta": theta, "cos_theta": c, "r": r, "dl": dl, "darea": darea,
            "kappa_m": kappa_m, "kappa_az": kappa_az,
            "kappa": kappa_m + kappa_az, "radius_over_dx": geom["radius"] / geom["dx"]}


def modal_stress_reconstructed(geom, sigma: float, fit: dict, *, mode: int,
                               eps: float) -> dict:
    fld = reconstructed_fields(geom, fit)
    share = legendre_p(mode, fld["cos_theta"])
    area = 4.0 * math.pi * geom["radius"] ** 2
    dp = float((2.0 * mode + 1.0) / area
               * (-sigma * fld["kappa"] * fld["darea"] * share).sum())
    dp_exact = exact_modal_stress(sigma, geom["radius"], mode, eps)
    ratio = (dp / dp_exact) if dp_exact != 0.0 else float("nan")
    omega_ratio = (math.copysign(1.0, ratio) * math.sqrt(abs(ratio))
                   if math.isfinite(ratio) else float("nan"))
    return {"dp_measured": dp, "dp_exact": dp_exact, "stiffness_ratio": ratio,
            "omega_ratio": omega_ratio,
            "omega_error_pct": ((omega_ratio - 1.0) * 100.0
                                if math.isfinite(omega_ratio) else float("nan")),
            "area_ratio": float(fld["darea"].sum()) / area,
            "kappaR_mean": float((fld["kappa"] * fld["darea"]).sum()
                                 / fld["darea"].sum()) * geom["radius"],
            "mode_l_amplitude": float(fit["coef"][mode]) / geom["radius"],
            "mode_l_seeded": eps,
            "amplitude_recovery_pct": (float(fit["coef"][mode]) / geom["radius"]
                                       / eps - 1.0) * 100.0 if eps else None,
            "rms_residual_over_R": fit["rms_residual_over_R"],
            "n_surface": fit["n_surface"]}


def run_ladder(radius: float, sigma: float, rho: float, *, kmax: int,
               ladder, modes, eps_list, noise_eps: float = 1.0e-6,
               fit_method: str = "delta") -> list:
    rows = []
    for mode in modes:
        for eps in list(eps_list) + [noise_eps, 0.0]:
            for rdx in ladder:
                geom = axisym_drop_geometry(radius, radius / rdx, mode=mode, eps=eps,
                                            rho0=rho)
                state, _, _ = csf_state(geom, sigma, periodic_x=False)
                fit = (fit_interface(geom, state, kmax=kmax) if fit_method == "delta"
                       else fit_interface_color(geom, state, kmax=kmax))
                res = modal_stress_reconstructed(geom, sigma, fit, mode=mode, eps=eps)
                res.update({"mode": mode, "eps": eps, "radius_over_dx": rdx,
                            "fit_method": fit_method,
                            "n_particles": geom["n_particles"],
                            "omega_analytic": rayleigh_frequency_3d(sigma, rho, radius, mode),
                            "is_noise_floor": eps <= noise_eps})
                if eps <= noise_eps:
                    # traction that the reconstruction invents from staircase alone,
                    # in units of the eps = 0.02 signal
                    res["noise_in_eps02_units"] = abs(res["dp_measured"]) / abs(
                        exact_modal_stress(sigma, radius, mode, 0.02))
                rows.append(res)
                print(f"l={mode} eps={eps:<8g} R/dx={rdx:4.0f} N={res['n_particles']:5d} "
                      f"A={res['area_ratio']:.4f} kR={res['kappaR_mean']:.4f} "
                      f"c_l={res['mode_l_amplitude']:+.5f} "
                      f"werr={res['omega_error_pct']:+8.2f}%"
                      f" resid/R={res['rms_residual_over_R']:.4f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fit", choices=("delta", "color"), default="color")
    args = ap.parse_args()
    ladder = [8.0, 12.0, 16.0] if args.quick else [8.0, 12.0, 16.0, 20.0, 30.0, 40.0, 60.0]
    modes = [2] if args.quick else [2, 3]
    eps_list = [0.02] if args.quick else [0.02, 0.05]
    kmax = 6
    t0 = time.time()
    rows = run_ladder(R_DEFAULT, SIGMA_DEFAULT, RHO_DEFAULT, kmax=kmax, ladder=ladder,
                      modes=modes, eps_list=eps_list, fit_method=args.fit)
    scored = [r for r in rows if r["eps"] > 1e-5
              and math.isfinite(r["stiffness_ratio"]) and r["stiffness_ratio"] > 0.0]
    passing = [r for r in scored if abs(r["omega_error_pct"]) < GATE_PCT]
    finest = [r for r in scored if r["radius_over_dx"] == max(ladder)
              and r["eps"] == min(eps_list)]
    payload = {
        "case": "SPH-OSC-DROP-FREQ", "round": 2,
        "operator": "interface reconstruction (Legendre fit of r(theta)) + analytic "
                    "axisymmetric capillary traction",
        "fit_method": args.fit,
        "reference": "Rayleigh 1879, Proc. R. Soc. Lond. 29:71; Lamb 1932 Art. 275",
        "gate_pct": GATE_PCT,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "parameters": {"radius_m": R_DEFAULT, "sigma_N_m": SIGMA_DEFAULT,
                       "rho_kg_m3": RHO_DEFAULT, "legendre_kmax": kmax,
                       "theta_samples": 4001, "wall_seconds": time.time() - t0},
        "rows": rows,
        "verdict": {
            "gate_pct": GATE_PCT,
            "n_scored": len(scored),
            "n_passing": len(passing),
            "gate_pass": bool(finest) and all(abs(r["omega_error_pct"]) < GATE_PCT
                                              for r in finest),
            "finest_ladder_rows": [{"mode": r["mode"], "eps": r["eps"],
                                    "radius_over_dx": r["radius_over_dx"],
                                    "omega_error_pct": r["omega_error_pct"]}
                                   for r in finest],
            "best": min(scored, key=lambda r: abs(r["omega_error_pct"]))["omega_error_pct"],
        },
    }
    (OUT / "interface_reconstruction.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
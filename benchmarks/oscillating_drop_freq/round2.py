"""Round 2 of ``CASE-SPH-OSC-DROP-FREQ``: interface-resolved surface quadrature.

Round 1 (``run_freq.py``) measured the modal stiffness with the **colour-gradient
surface delta** of :func:`tensorsph.surface_tension_dispersion.csf_state` and got
a traction of the wrong sign, 2-6x too large, with an equivalent noise amplitude
comparable to the seeded perturbation.  ``diagnose.py`` shows that the reported
``<kappa> R = 1.2`` was *not* a curvature defect (the estimator gives 2.04-2.14
against the exact 2 once the axisymmetric coarea factor is applied); the defect
sits in the surface measure.

This script therefore replaces the surface *measure* only, keeping everything
else from the shipped operator, so the two error sources are scored apart:

``delta``        round-1 route: ``f = -sigma kappa delta_s`` with the
                 colour-gradient ``delta_s`` (the shipped CSF force).
``shell``        interface-resolved route: the ``delta_s`` shell is extracted,
                 ordered along the meridian and the force is quadratured with
                 the *geometric* surface weight ``dA = pi rho dl`` (``dl`` the
                 marker arc length; the ``pi`` carries the double cover of the
                 full-circle meridian map, the same convention as the
                 ``dVol = pi rho dA`` of the geometry).

For each route the curvature is taken from

``estimator``  ``fit_quadratic`` meridional curvature + CSF-normal azimuthal
               term (``axisym_curvature``): what a running solver actually has;
``marker``     curvature from a centred-difference circle fit of the ordered
               shell markers (independent check of ``estimator``);
``exact``      linearised exact ``2/R + (l-1)(l+2) eps P_l / R``: isolates the
               measure/quadrature error alone.

Invariants that must hold before any frequency number is believable:
``volume`` ``sum dV / (4/3 pi R^3) = 1``;
``area``   ``sum dA_shell / (4 pi R^2) = 1``;
``coarea`` ``2 sum delta_s dV / (4 pi R^2) = 1`` (the ``2`` is the meridian
double cover, which round 1 omitted).

Usage::

    PYTHONPATH=src python benchmarks/oscillating_drop_freq/round2.py [--quick]
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
    axisym_curvature,
    axisym_drop_geometry,
    exact_modal_stress,
    legendre_p,
    rayleigh_frequency_3d,
)
from tensorsph.surface_tension_dispersion import csf_state  # noqa: E402

R_DEFAULT = 0.02
SIGMA_DEFAULT = 7.2
RHO_DEFAULT = 1000.0
GATE_PCT = 6.0
OUT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# shell extraction + arc-length quadrature
# --------------------------------------------------------------------------- #

def shell_quadrature(geom: dict, state, *, cutoff_frac: float = 0.05) -> dict:
    """Order the CSF shell along the meridian and give it geometric weights.

    ``cutoff_frac`` keeps particles with ``delta_s > cutoff_frac * max`` (the
    same 5 % cut ``CSFState.interface_mask`` uses) and drops the inner shell,
    whose ``delta_s`` is a colour-gradient tail rather than an interface.
    """
    ds = state.delta_s
    keep = ds > cutoff_frac * float(ds.max())
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    pos = geom["pos"][idx]
    phi = torch.atan2(pos[:, 1], pos[:, 0])
    order = torch.argsort(phi)
    idx = idx[order]
    pos = pos[order]
    phi = phi[order]
    n = idx.shape[0]
    nxt = torch.roll(torch.arange(n), -1)
    prv = torch.roll(torch.arange(n), 1)
    dl = 0.5 * torch.linalg.vector_norm(pos[nxt] - pos[prv], dim=1)  # polyline arc
    rho = geom["rho"][idx]
    darea = math.pi * rho * dl            # axisymmetric area element (double-cover pi)
    rad = torch.linalg.vector_norm(pos, dim=1)
    return {"idx": idx, "pos": pos, "phi": phi, "dl": dl, "darea": darea,
            "cos_theta": pos[:, 1] / rad.clamp_min(1e-30), "n": n}


def marker_curvature(shell: dict) -> torch.Tensor:
    """Signed meridional plane-curve curvature from the ordered markers.

    ``kappa = (t x t') / |t|^3`` with ``t`` a centred difference of the marker
    positions, signed so a convex (outward-bulging) interface has ``kappa > 0``,
    matching the CSF convention.
    """
    pos = shell["pos"]
    n = pos.shape[0]
    nxt = torch.roll(torch.arange(n), -1)
    prv = torch.roll(torch.arange(n), 1)
    d1 = 0.5 * (pos[nxt] - pos[prv])
    d2 = pos[nxt] - 2.0 * pos + pos[prv]
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    kap = cross / torch.linalg.vector_norm(d1, dim=1).clamp_min(1e-30) ** 3
    if float(kap.mean()) < 0.0:
        kap = -kap
    return kap


def marker_normal(shell: dict) -> torch.Tensor:
    pos = shell["pos"]
    n = pos.shape[0]
    nxt = torch.roll(torch.arange(n), -1)
    prv = torch.roll(torch.arange(n), 1)
    t = pos[nxt] - pos[prv]
    t = t / torch.linalg.vector_norm(t, dim=1, keepdim=True).clamp_min(1e-30)
    nrm = torch.stack((-t[:, 1], t[:, 0]), dim=1)
    radial = pos / torch.linalg.vector_norm(pos, dim=1, keepdim=True).clamp_min(1e-30)
    flip = torch.where((nrm * radial).sum(1) < 0.0, -1.0, 1.0)
    return nrm * flip.unsqueeze(-1)


def exact_curvature(geom: dict, mode: int, eps: float) -> torch.Tensor:
    """Linearised exact ``kappa = 2/R + (l-1)(l+2) eps P_l / R``."""
    share = legendre_p(mode, geom["cos_theta"])
    return 2.0 / geom["radius"] + (mode - 1.0) * (mode + 2.0) * eps * share / geom["radius"]


# --------------------------------------------------------------------------- #
# modal stiffness on the two surface measures
# --------------------------------------------------------------------------- #

def modal_stress_delta(geom, sigma, *, mode, eps, curvature="estimator",
                       delta_normalization="legacy") -> dict:
    state, h, _ = csf_state(geom, sigma, periodic_x=False,
                            delta_normalization=delta_normalization)
    kappa = (axisym_curvature(state, geom) if curvature == "estimator"
             else exact_curvature(geom, mode, eps))
    dvol = geom["dvol"]
    area = 4.0 * math.pi * geom["radius"] ** 2
    share = legendre_p(mode, geom["cos_theta"])
    dp = float((2.0 * mode + 1.0) / area
               * (-sigma * kappa * state.delta_s * dvol * share).sum())
    w = state.delta_s * dvol
    return {"route": "delta", "curvature": curvature, "dp_measured": dp,
            "area_ratio": float(2.0 * w.sum() / area),
            "kappaR_mean": float((kappa * w).sum() / w.sum()) * geom["radius"],
            "n_surface": int((state.delta_s > 0.05 * float(state.delta_s.max())).sum()),
            "h": h}


def modal_stress_shell(geom, sigma, *, mode, eps, curvature="estimator",
                       cutoff_frac: float = 0.05) -> dict:
    state, h, _ = csf_state(geom, sigma, periodic_x=False)
    shell = shell_quadrature(geom, state, cutoff_frac=cutoff_frac)
    sign = torch.where(shell["pos"][:, 0] >= 0.0, 1.0, -1.0)
    if curvature == "estimator":
        kappa_m = state.kappa[shell["idx"]]
        n = state.normal[shell["idx"]]
        kappa = kappa_m + n[:, 0] * sign / geom["rho_safe"][shell["idx"]]
    elif curvature == "marker":
        kappa_m = marker_curvature(shell)
        n = marker_normal(shell)
        kappa = kappa_m + n[:, 0] * sign / geom["rho_safe"][shell["idx"]]
    else:
        kappa = exact_curvature(geom, mode, eps)[shell["idx"]]
    area = 4.0 * math.pi * geom["radius"] ** 2
    share = legendre_p(mode, shell["cos_theta"])
    dp = float((2.0 * mode + 1.0) / area
               * (-sigma * kappa * shell["darea"] * share).sum())
    darea_total = float(shell["darea"].sum())
    return {"route": "shell", "curvature": curvature, "dp_measured": dp,
            "area_ratio": darea_total / area,
            "kappaR_mean": float((kappa * shell["darea"]).sum() / darea_total)
            * geom["radius"],
            "n_surface": int(shell["n"]),
            "dl_over_true": float(shell["dl"].sum() / (2.0 * math.pi * geom["radius"])),
            "h": h}


def row(geom, sigma, mode, eps) -> dict:
    out = {}
    for curv in ("estimator", "exact"):
        out[f"delta_{curv}"] = modal_stress_delta(geom, sigma, mode=mode, eps=eps,
                                                  curvature=curv)
    for curv in ("estimator", "marker", "exact"):
        out[f"shell_{curv}"] = modal_stress_shell(geom, sigma, mode=mode, eps=eps,
                                                  curvature=curv)
    dp_exact = exact_modal_stress(sigma, geom["radius"], mode, eps)
    omega_exact = rayleigh_frequency_3d(sigma, geom["rho0"], geom["radius"], mode)
    for val in out.values():
        ratio = val["dp_measured"] / dp_exact
        val["dp_exact"] = dp_exact
        val["stiffness_ratio"] = ratio
        val["omega_ratio"] = math.copysign(1.0, ratio) * math.sqrt(abs(ratio))
        val["omega_error_pct"] = (val["omega_ratio"] - 1.0) * 100.0
        val["omega_measured"] = omega_exact * val["omega_ratio"]
    return {"radius_over_dx": geom["radius"] / geom["dx"], "mode": mode, "eps": eps,
            "n_particles": geom["n_particles"], "omega_analytic": omega_exact,
            "volume_ratio": float(geom["dvol"].sum()
                                 / (4.0 / 3.0 * math.pi * geom["radius"] ** 3)),
            "measurements": out}


def verdict(rows: list) -> dict:
    """Score each route: the primary gate is the *shell* route with the shipped
    curvature estimator -- the strongest statement the linear-response data
    supports.  The ``delta`` route is the round-1 behaviour, kept for contrast."""
    def best(route, key="omega_error_pct"):
        cand = [r for r in rows if r["measurements"][route]["omega_ratio"] > 0.0]
        if not cand:
            return None
        return min(cand, key=lambda r: abs(r["measurements"][route][key]))

    out = {"gate_pct": GATE_PCT, "routes": {}}
    for route in ("delta_estimator", "delta_exact", "shell_estimator",
                  "shell_marker", "shell_exact"):
        ok = [r for r in rows
              if r["measurements"][route]["omega_ratio"] > 0.0
              and abs(r["measurements"][route]["omega_error_pct"]) < GATE_PCT]
        b = best(route)
        out["routes"][route] = {
            "n_rows": len(rows),
            "n_passing": len(ok),
            "n_sign_flipped": sum(1 for r in rows
                                  if r["measurements"][route]["omega_ratio"] <= 0.0),
            "best": None if b is None else {
                "mode": b["mode"], "eps": b["eps"],
                "radius_over_dx": b["radius_over_dx"],
                "omega_analytic": b["omega_analytic"],
                "omega_measured": b["measurements"][route]["omega_measured"],
                "omega_error_pct": b["measurements"][route]["omega_error_pct"]},
            "passing_rows": [{"mode": r["mode"], "eps": r["eps"],
                              "radius_over_dx": r["radius_over_dx"],
                              "omega_error_pct":
                                  r["measurements"][route]["omega_error_pct"]}
                             for r in ok],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ladder = [8.0, 12.0, 16.0] if args.quick else [8.0, 12.0, 16.0, 20.0, 30.0, 40.0]
    eps_list = [0.02] if args.quick else [0.02, 0.05]
    modes = [2] if args.quick else [2, 3]
    rows = []
    t_start = time.time()
    for mode in modes:
        for eps in eps_list:
            for rdx in ladder:
                geom = axisym_drop_geometry(R_DEFAULT, R_DEFAULT / rdx, mode=mode,
                                            eps=eps, rho0=RHO_DEFAULT)
                r = row(geom, SIGMA_DEFAULT, mode, eps)
                rows.append(r)
                m = r["measurements"]
                print(f"l={mode} eps={eps:g} R/dx={rdx:4.0f} N={r['n_particles']:5d} "
                      f"| dl/est {m['delta_estimator']['omega_error_pct']:+8.1f}%"
                      f" | sh/est {m['shell_estimator']['omega_error_pct']:+8.1f}%"
                      f" sh/mkr {m['shell_marker']['omega_error_pct']:+8.1f}%"
                      f" sh/exa {m['shell_exact']['omega_error_pct']:+8.1f}%"
                      f" | A={m['shell_exact']['area_ratio']:.3f}"
                      f" coarea={m['delta_exact']['area_ratio']:.3f}"
                      f" dl/2piR={m['shell_exact']['dl_over_true']:.3f}", flush=True)
    payload = {
        "case": "SPH-OSC-DROP-FREQ", "round": 2, "reference": "Rayleigh 1879",
        "gate_pct": GATE_PCT,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "device": "cpu"},
        "parameters": {"radius_m": R_DEFAULT, "sigma_N_m": SIGMA_DEFAULT,
                       "rho_kg_m3": RHO_DEFAULT, "wall_seconds": time.time() - t_start},
        "rows": rows,
        "verdict": verdict(rows),
    }
    (OUT / "round2.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
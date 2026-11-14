"""Single-phase (free-surface) CSS surface tension: interface completion closure.

Anchor: ``XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION``.
Reference: exact 2-D Young-Laplace ``Delta p = sigma / R`` for a circular
interface (no discretisation error in the reference itself).

Motivation (open item of round 2, ``docs/xval_sphinxys_mixture_laplace``)
------------------------------------------------------------------------
Round 2 gated the *two-phase* mixture CSS model at -4.7 / -5.8 / -6.2 %
(R/dx = 16/32/64) but left the single-phase (vacuum-reduced) configuration at
**-29.7 ... -30.0 %**, with the note: "the vacuum reduction of the mixture
model is not Laplacian-accurate with this force assembly; it needs the
both-phase shell."  A free surface is exactly the ship-hydrodynamics case
(spray, droplets, jet breakup without resolving air), so the -30 % matters.

What is measured
----------------
The one-sided stress flux ``sum_{j present} V_j (S_i + S_j) . grad_i W_ij``
does not represent a full kernel support: no neighbour exists on the far side
of the interface, so the tangential part of the stress divergence is
under-resolved, and a Laplace reading built from it is short by ~30 %.

:func:`tensorsph.sphinxys_css.interface_completion` supplies the specular image
of each present kernel gradient across the local interface tangent plane --
the standard "restore the missing half support" closure.  Full mirror weight
(``weight = 1``) over-restores, because the mixture colour gradient is itself
one-sided: only a fraction ``1 - f_in`` of the interface delta lives outside
the free surface.  ``weight = alpha`` is therefore a *closure coefficient*.

Protocol (no leakage): ``alpha`` is fitted at the **middle rung** of the ladder
(R/dx = 32) only, then applied unchanged to R/dx = 16 and 64 (cross-resolution
validation) and to a smoothing-width sweep (cross-parameter validation).
Every number below is written to ``out_single_phase_completion/gate.json`` and
read back by ``emit_tables.py`` -- zero hand copy.

Usage::

    PYTHONPATH=src python benchmarks/surface_tension/single_phase_completion.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import tensorsph.sphinxys_css as sc
from tensorsph.droplet_impact import wendland_c2_gradient
from tensorsph.neighbor_search import cell_linked_pairs

SIGMA = 0.072            # N/m, water-air at 20 C
RADIUS = 0.1             # m
RHO_WATER = 1000.0
H_OVER_DX = sc.SPHINXSYS_H_SPACING_RATIO
GATE = 0.15              # ledger criterion |error| < 15 %
LADDER = (16, 32, 64)
SWEEP = (0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0)
FIT_RUNG = 32
OUT = Path(__file__).resolve().parent / "out_single_phase_completion"


# --------------------------------------------------------------------------- #
# scene + reading
# --------------------------------------------------------------------------- #
def make_disc(radius: float, dx: float):
    """Single-phase water disc on a square lattice (vacuum reduction)."""
    span = radius + 2.0 * dx
    axis = torch.arange(-span, span + 0.5 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    dist = torch.linalg.vector_norm(pts, dim=1)
    keep = dist <= radius + 1.0e-12
    pos = pts[keep].clone()
    r = dist[keep]
    vol = torch.full((pos.shape[0],), dx * dx, dtype=pos.dtype)
    rho0 = torch.full((pos.shape[0],), RHO_WATER, dtype=pos.dtype)
    return pos, r, rho0, vol


def measure(radius: float, dx: float, alpha: float, h_over_dx: float = H_OVER_DX,
            hourglass: bool = False) -> dict:
    """Inward force density integrated over the whole body -> Laplace reading."""
    h = h_over_dx * dx
    pos, r, rho0, vol = make_disc(radius, dx)
    mass = rho0 * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    frac = torch.ones(pos.shape[0], dtype=pos.dtype)
    acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs, None, SIGMA, h, frac=frac,
        mode="single_phase", delta="sphinxys", hourglass=hourglass,
        assembly="consistent", completion="mirror", completion_weight=alpha)
    n_out = pos / r.clamp_min(1e-300)[:, None]
    # f_density = rho a is a force per volume; dV = dx^2 on this lattice.
    f_inward = -rho0 * (acc * n_out).sum(dim=1) * (dx * dx)
    area = 2.0 * math.pi * radius
    dp = float(f_inward.sum()) / area
    scale = float((rho0 * vol * torch.linalg.vector_norm(acc, dim=1)).sum())
    mom = (rho0 * vol)[:, None].mul(acc).sum(dim=0)
    delta_integral = float((fields.delta_s * vol).sum()) / area
    theory = SIGMA / radius
    return {
        "radius_over_dx": radius / dx,
        "dx_m": dx,
        "h_over_dx": h_over_dx,
        "particles": int(pos.shape[0]),
        "completion_weight": alpha,
        "hourglass": hourglass,
        "dp_pa": dp,
        "dp_theory_pa": theory,
        "err_pct": (dp - theory) / theory * 100.0,
        "stress_reading_err_pct": (SIGMA * delta_integral / radius - theory)
        / theory * 100.0,
        "delta_integral": delta_integral,
        "momentum_residual": float(torch.linalg.vector_norm(mom)) / max(scale, 1e-300),
    }


def delta_mass_split(radius: float, dx: float) -> dict:
    """Where does the mixture colour delta live in the two-phase reference run?

    Used to test the tempting explanation "``alpha*`` is the mass fraction of the
    interface delta that sits beyond the free surface": with ``rho_air/rho_w =
    1e-3`` the contact fraction of the air phase is ``f_air approx 1e-3``, so the
    air-side delta mass is ~0.1 % of the water side and that explanation dies.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mixture_laplace as ML

    h = H_OVER_DX * dx
    pos, r, phase, rho0, vol = ML.make_two_phase(radius, dx, ML.RHO_AIR)
    mass = rho0 * vol
    pairs = ML.cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    same = phase[i] == phase[j]
    frac = sc.contact_fraction(
        rho0, torch.where(phase == 0,
                          torch.tensor(ML.RHO_AIR, dtype=pos.dtype),
                          torch.tensor(RHO_WATER, dtype=pos.dtype)))
    _acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs[same], pairs[~same], SIGMA, h, frac=frac,
        mode="two_phase", hourglass=False, assembly="consistent",
        bracket="symmetric")
    water = phase == 0
    d_water = float((fields.delta_s[water] * vol[water]).sum())
    d_air = float((fields.delta_s[~water] * vol[~water]).sum())
    return {"radius_over_dx": radius / dx,
            "delta_mass_water": d_water, "delta_mass_air": d_air,
            "air_over_water": d_air / d_water}


def fit_alpha(sweep: list[dict]) -> dict:
    """Linear interpolation of the zero crossing of ``err(alpha)``."""
    pts = sorted((s["completion_weight"], s["err_pct"]) for s in sweep)
    for (a0, e0), (a1, e1) in zip(pts, pts[1:]):
        if e0 == 0.0:
            return {"alpha_star": a0, "bracket": [a0, a1], "method": "exact node"}
        if e0 * e1 < 0.0:
            return {"alpha_star": a0 - e0 * (a1 - a0) / (e1 - e0),
                    "bracket": [a0, a1], "method": "linear interpolation"}
    raise RuntimeError("error does not change sign over the sweep")


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def plot(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for rung in LADDER:
        pts = [s for s in payload["sweep"] if s["radius_over_dx"] == rung]
        ax.plot([p["completion_weight"] for p in pts], [p["err_pct"] for p in pts],
                "o-", ms=4, label=f"R/dx = {rung}")
    ax.axhspan(-GATE * 100, GATE * 100, color="0.85", label="gate < 15 %")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(payload["fit"]["alpha_star"], color="r", ls="--", lw=1,
               label=rf"$\alpha^*$ = {payload['fit']['alpha_star']:.3f}")
    ax.set_xlabel(r"completion weight $\alpha$")
    ax.set_ylabel(r"$\Delta p$ error vs $\sigma/R$  [%]")
    ax.set_title("(a) closure scan")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    rungs = [str(r) for r in LADDER]
    base = [p["err_pct"] for p in payload["ladder_baseline"]]
    gated = [p["err_pct"] for p in payload["ladder_validated"]]
    x = range(len(rungs))
    ax.bar([i - 0.2 for i in x], base, 0.4, label=r"$\alpha=0$ (round-2 open item)")
    ax.bar([i + 0.2 for i in x], gated, 0.4, label=rf"$\alpha=\alpha^*$ (closure)")
    ax.axhspan(-GATE * 100, GATE * 100, color="0.85", zorder=0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"R/dx={r}" for r in rungs])
    ax.set_ylabel(r"$\Delta p$ error [%]")
    ax.set_title("(b) ladder: baseline vs closure")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    hs = [s["h_over_dx"] for s in payload["smoothing_sensitivity"]]
    es = [s["err_pct"] for s in payload["smoothing_sensitivity"]]
    ax.plot(hs, es, "s-", color="C3",
            label=rf"$\alpha=\alpha^*$ at R/dx={FIT_RUNG}")
    ax.axhspan(-GATE * 100, GATE * 100, color="0.85")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("h / dx")
    ax.set_ylabel(r"$\Delta p$ error [%]")
    ax.set_title("(c) smoothing-width transfer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("SPHinXsys CSS mixture model, single-phase free surface: "
                 "interface-completion closure vs exact Young-Laplace")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii-dx", default=",".join(str(r) for r in LADDER))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    ladder = tuple(int(s) for s in args.radii_dx.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    payload: dict = {
        "case": {"radius_m": RADIUS, "sigma_nm": SIGMA, "rho_water": RHO_WATER,
                 "h_over_dx": H_OVER_DX, "kernel": "wendland_c2",
                 "lattice": "square", "dimension": 2,
                 "reference": "Delta p = sigma / R (exact)"},
        "gate_pct": GATE * 100,
        "fit_rung": FIT_RUNG,
        "ladder": list(ladder),
    }

    # S0/S1: baseline + closure scan (the fit rung only is used to fit alpha*)
    payload["ladder_baseline"] = [measure(RADIUS, RADIUS / r, 0.0) for r in ladder]
    sweep = [measure(RADIUS, RADIUS / FIT_RUNG, a) for a in SWEEP]
    payload["sweep"] = sweep
    payload["fit"] = fit_alpha(sweep)
    alpha_star = payload["fit"]["alpha_star"]

    # S2: cross-resolution validation with the transferable alpha*
    payload["ladder_validated"] = [measure(RADIUS, RADIUS / r, alpha_star)
                                  for r in ladder]

    # S3: cross-parameter transfer: smoothing ratio h/dx changed away from 1.3
    payload["smoothing_sensitivity"] = [
        measure(RADIUS, RADIUS / FIT_RUNG, alpha_star, h_over_dx=hv)
        for hv in (1.1, 1.2, 1.3, 1.4, 1.5)]

    # S4: hourglass ablation at the fit rung (upstream penalty on the free surface)
    payload["hourglass_ablation"] = [
        measure(RADIUS, RADIUS / FIT_RUNG, alpha_star, hourglass=hg)
        for hg in (False, True)]

    # S5: momentum conservation of the completed flux (must stay machine zero)
    payload["momentum"] = {
        "residual_alpha0": payload["ladder_baseline"][1]["momentum_residual"],
        "residual_alpha_star": payload["ladder_validated"][1]["momentum_residual"],
    }

    # S6: two-phase regression -- the closure is single-phase only, defaults off
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mixture_laplace as ML   # sibling benchmark, same scene builder
    payload["two_phase_regression"] = []
    for r in ladder:
        rec = ML.measure(RADIUS, RADIUS / r, rho_air=ML.RHO_AIR, mode="two_phase",
                         assembly="consistent", hourglass=False,
                         bracket="symmetric")
        payload["two_phase_regression"].append(
            {"radius_over_dx": r, "dp_pa": rec["dp_flux_all_pa"],
             "err_pct": rec["dp_flux_all_err_pct"]})

    # S7: does alpha* equal the missing-side delta mass fraction? (falsification)
    payload["delta_mass_split"] = delta_mass_split(RADIUS, RADIUS / FIT_RUNG)
    payload["full_mirror_hypothesis"] = {
        "alpha": 1.0,
        "err_pct": next(s["err_pct"] for s in sweep if s["completion_weight"] == 1.0),
        "note": "plane-symmetric full mirror: over-restores the free-surface flux",
    }

    payload["runtime_s"] = time.time() - t0
    (out / "gate.json").write_text(json.dumps(payload, indent=2))
    plot(payload, out / "gate.png")

    print(f"alpha* = {alpha_star:.4f}  ({payload['fit']['method']}, "
          f"bracket {payload['fit']['bracket']})")
    for rec in payload["ladder_baseline"]:
        print(f"  R/dx={rec['radius_over_dx']:3.0f}  alpha=0      "
              f"err={rec['err_pct']:8.3f} %")
    for rec in payload["ladder_validated"]:
        print(f"  R/dx={rec['radius_over_dx']:3.0f}  alpha=alpha* "
              f"err={rec['err_pct']:8.3f} %")
    for rec in payload["smoothing_sensitivity"]:
        print(f"  h/dx={rec['h_over_dx']:.1f}  err={rec['err_pct']:8.3f} %")
    print(f"  momentum: alpha=0 {payload['momentum']['residual_alpha0']:.2e}  "
          f"alpha* {payload['momentum']['residual_alpha_star']:.2e}")
    print(f"  two-phase regression: "
          + ", ".join(f"{r['err_pct']:.2f}%" for r in payload["two_phase_regression"]))
    print(f"  runtime {payload['runtime_s']:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Mixture material + surface tension: Young-Laplace gate (XVAL round 2, m18642).

Anchor: ``XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION``.
Reference model: SPHinXsys continuous surface stress (CSS) with a *mixture*
contact-fraction colour gradient; raw sources and sha256 in
``docs/xval_sphinxys_surface_tension/raw/``.  Acceptance criterion of the
ledger: ``|error| < 15 %`` against ``Delta p = sigma / R`` (2-D circular
interface, exact).

What this script establishes, in one run
----------------------------------------
0. **operator floor** -- before blaming the colour field, the discrete stress
   divergence is fed an *analytic* continuous surface stress
   ``S = sigma f(r) (I - n n^T)`` with ``int f dn = 1``.  Its exact divergence
   is ``-sigma f(r) n / r``, so the exact Laplace reading is ``sigma/R``.
   Whatever this returns is the irreducible discrete-operator error; the
   colour-field/model error is whatever exceeds it.
1. **assembly normalisation** -- the ported pair weight ``(m_i/rho_i) V_j``
   carries the particle mass, so the force density scales with ``dx^2`` and
   decays as the grid is refined.  Weight ``V_j/rho_i`` is what
   ``a_i = (1/rho_i) div S_i`` requires.  Both are run; the ratio of the two
   readings must equal the particle mass ``m_i`` to machine precision.
2. **Laplace ladder** -- ``R/dx`` in {16, 32, 64}, two-phase (1000:1) and
   single-phase (vacuum reduction), for both assemblies.  Two readings of the
   *same* force field: over the water disc only, and over both phases.  The
   interface shell carries ``|grad c|`` on both sides (the mixture colour
   gradient is one-sided, ~70 % of the delta mass sits on the water side), so
   restricting the normal integral to the water disc discards part of the
   interface; the both-phase reading is the Laplace jump.
3. **mixture density ratio** -- the contact fraction
   ``f_k = rho0_i/(rho0_i + rho0_k)`` is the mixture-material knob.  Sweep
   ``rho_air/rho_water`` and report ``int |grad c| dn`` and the Laplace error.
4. **hourglass ablation** -- the upstream zero-surface-energy penalty breaks
   momentum conservation by construction; report the momentum residual with it
   on and off.

Every number in ``docs/xval_sphinxys_mixture_laplace/README.md`` is read back
from ``out_mixture_laplace/gate.json`` written here (zero hand-copy).

Usage::

    python benchmarks/surface_tension/mixture_laplace.py                 # 16,32
    python benchmarks/surface_tension/mixture_laplace.py --radii-dx 64   # heavy
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

SIGMA = 0.072           # N/m, water-air at 20 C
RADIUS = 0.1            # m
RHO_WATER = 1000.0
RHO_AIR = 1.0           # 1000:1, the upstream square-droplet ratio
H_OVER_DX = sc.SPHINXSYS_H_SPACING_RATIO
RINGS = 6               # air rings outside the interface (support = 2h = 2.6 dx)
GATE_PCT = 15.0
OUT = Path(__file__).resolve().parent / "out_mixture_laplace"


# --------------------------------------------------------------------------- #
# lattices
# --------------------------------------------------------------------------- #
def make_two_phase(radius: float, dx: float, rho_air: float, rings: int = RINGS):
    """Water disc + ``rings`` air rings on one square lattice (as upstream)."""
    span = radius + (rings + 1) * dx
    axis = torch.arange(-span, span + 0.5 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    dist = torch.linalg.vector_norm(pts, dim=1)
    keep = dist <= radius + rings * dx + 1.0e-12
    pos, r = pts[keep].clone(), dist[keep]
    phase = (r > radius + 1.0e-12).to(torch.int64)          # 0 = water, 1 = air
    rho0 = torch.where(phase == 0, torch.tensor(RHO_WATER, dtype=pos.dtype),
                       torch.tensor(rho_air, dtype=pos.dtype))
    vol = torch.full((pos.shape[0],), dx * dx, dtype=pos.dtype)
    return pos, r, phase, rho0, vol


def make_single_phase(radius: float, dx: float):
    """Water disc only: the vacuum reduction of the mixture colour gradient."""
    span = radius + 2.0 * dx
    axis = torch.arange(-span, span + 0.5 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    dist = torch.linalg.vector_norm(pts, dim=1)
    keep = dist <= radius + 1.0e-12
    pos, r = pts[keep].clone(), dist[keep]
    phase = torch.zeros(pos.shape[0], dtype=torch.int64)
    rho0 = torch.full((pos.shape[0],), RHO_WATER, dtype=pos.dtype)
    vol = torch.full((pos.shape[0],), dx * dx, dtype=pos.dtype)
    return pos, r, phase, rho0, vol


def readings(pos, r, phase, rho0, vol, acc, radius, dx) -> dict:
    """Normal integral of the surface force density, over water / both phases."""
    n_out = pos / r.clamp_min(1e-300)[:, None]
    f_density = -rho0 * (acc * n_out).sum(dim=1) * (dx * dx)   # inward > 0
    water = phase == 0
    area = 2.0 * math.pi * radius
    total_w = float(f_density[water].sum())
    total_a = float(f_density.sum())
    scale = float((rho0 * vol * torch.linalg.vector_norm(acc, dim=1)).sum())
    mom = (rho0 * vol)[:, None].mul(acc).sum(dim=0)
    return {
        "dp_water_pa": total_w / area,
        "dp_all_pa": total_a / area,
        "momentum_residual": float(torch.linalg.vector_norm(mom)) / max(scale, 1e-300),
    }


# --------------------------------------------------------------------------- #
# 0. operator floor: analytic continuous surface stress on a uniform lattice
# --------------------------------------------------------------------------- #
def operator_floor(radius: float, dx: float, width_over_h: float = 1.0,
                   rho: float = RHO_WATER) -> dict:
    h = H_OVER_DX * dx
    pos, r, phase, rho0, vol = make_two_phase(radius, dx, rho, rings=RINGS)
    n = pos.shape[0]
    rho0 = torch.full((n,), rho, dtype=pos.dtype)          # uniform: isolate operator
    vol = vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    width = width_over_h * h
    dist = r
    f = torch.exp(-(((dist - radius) / width) ** 2)) / (width * math.sqrt(math.pi))
    n_out = pos / dist.clamp_min(1e-300)[:, None]
    eye = torch.eye(2, dtype=pos.dtype)[None]
    stress = SIGMA * f[:, None, None] * (eye - n_out[:, :, None] * n_out[:, None, :])
    gw = wendland_c2_gradient(pos[i] - pos[j], h)
    flux = torch.einsum("pab,pb->pa", stress[i] + stress[j], gw)
    acc = torch.zeros_like(pos)
    acc.index_add_(0, i, (vol[j] / rho0[i])[:, None] * flux)
    acc.index_add_(0, j, -(vol[i] / rho0[j])[:, None] * flux)
    theory = SIGMA / radius
    rd = readings(pos, r, phase, rho0, vol, acc, radius, dx)
    return {
        "radius_over_dx": radius / dx,
        "width_over_h": width_over_h,
        "dp_pa": rd["dp_all_pa"],
        "err_pct": (rd["dp_all_pa"] - theory) / theory * 100.0,
        "momentum_residual": rd["momentum_residual"],
    }


# --------------------------------------------------------------------------- #
# 1-4. the mixture model itself
# --------------------------------------------------------------------------- #
def measure(radius: float, dx: float, rho_air: float = RHO_AIR,
            mode: str = "two_phase", assembly: str = "upstream",
            hourglass: bool = True, delta: str = "sphinxys",
            bracket: str = "mixture") -> dict:
    h = H_OVER_DX * dx
    if mode == "two_phase":
        pos, r, phase, rho0, vol = make_two_phase(radius, dx, rho_air)
    else:
        pos, r, phase, rho0, vol = make_single_phase(radius, dx)
    mass = rho0 * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    same = phase[i] == phase[j]
    pairs_inner, pairs_cross = pairs[same], pairs[~same]
    frac = sc.contact_fraction(rho0, torch.where(phase == 0,
                                                torch.tensor(rho_air, dtype=pos.dtype),
                                                RHO_WATER))
    acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs_inner, pairs_cross, SIGMA, h, frac=frac,
        mode=mode, delta=delta, hourglass=hourglass,
        hourglass_coeff=sc.SPHINXSYS_HOURGLASS_COEFF, assembly=assembly,
        bracket=bracket)

    water = phase == 0
    area = 2.0 * math.pi * radius
    delta_integral = float((fields.delta_s[water] * vol[water]).sum()) / area
    dp_stress = SIGMA * delta_integral / radius
    rd = readings(pos, r, phase, rho0, vol, acc, radius, dx)
    theory = SIGMA / radius
    return {
        "radius_over_dx": radius / dx,
        "dx_m": dx,
        "h_over_dx": H_OVER_DX,
        "particles": int(pos.shape[0]),
        "water_particles": int(water.sum()),
        "rho_air_over_rho_water": rho_air / RHO_WATER,
        "contact_fraction_water": float(frac[water][0]),
        "mode": mode,
        "assembly": assembly,
        "hourglass": hourglass,
        "delta": delta,
        "bracket": bracket,
        "particle_mass_water": float(mass[water][0]),
        "delta_integral": delta_integral,
        "dp_stress_pa": dp_stress,
        "dp_stress_err_pct": (dp_stress - theory) / theory * 100.0,
        "dp_flux_water_pa": rd["dp_water_pa"],
        "dp_flux_water_err_pct": (rd["dp_water_pa"] - theory) / theory * 100.0,
        "dp_flux_all_pa": rd["dp_all_pa"],
        "dp_flux_all_err_pct": (rd["dp_all_pa"] - theory) / theory * 100.0,
        "theory_pa": theory,
        "momentum_residual": rd["momentum_residual"],
    }


def gate(rows, criteria=None) -> dict:
    """Machine gate: |err| < 15 % on the both-phase flux reading, consistent assembly."""
    rows = [r for r in rows if r["assembly"] == "consistent"
            and r["bracket"] == "symmetric"
            and r["mode"] == "two_phase" and not r["hourglass"]]
    worst = max(rows, key=lambda r: abs(r["dp_flux_all_err_pct"]))
    stress_worst = max(rows, key=lambda r: abs(r["dp_stress_err_pct"]))
    ok = abs(worst["dp_flux_all_err_pct"]) < GATE_PCT
    return {
        "criterion_pct": GATE_PCT,
        "reading": "sum over both phases of rho (-a.n) A / (2 pi R)",
        "assembly": "consistent (V_j/rho_i)",
        "bracket": "symmetric (S_i + S_j)",
        "model": "two_phase, hourglass off (momentum-exact), delta=sphinxys",
        "ladder": [{"radius_over_dx": r["radius_over_dx"],
                    "err_pct": r["dp_flux_all_err_pct"]} for r in rows],
        "worst_abs_err_pct": abs(worst["dp_flux_all_err_pct"]),
        "worst_at_radius_over_dx": worst["radius_over_dx"],
        "stress_reading_worst_abs_err_pct": abs(stress_worst["dp_stress_err_pct"]),
        "verdict": "pass" if ok else "fail",
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def plot(summary, ratios, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    ax = axes[0]
    ax.axhline(1.0, color="0.4", lw=1.0, ls=":", label="exact $\\sigma/R$")
    for asm, style, lab in (("upstream", "s--", "upstream $m_iV_j/\\rho_i$"),
                            ("consistent", "o-", "consistent $V_j/\\rho_i$")):
        rows = [r for r in summary if r["assembly"] == asm and r["mode"] == "two_phase"
                and r["hourglass"] and r["bracket"] in ("mixture", "symmetric")]
        res = [r["radius_over_dx"] for r in rows]
        ax.plot(res, [r["dp_flux_all_pa"] / r["theory_pa"] for r in rows], style,
                label=lab)
        ax.plot(res, [r["dp_flux_water_pa"] / r["theory_pa"] for r in rows], style,
                alpha=0.45, lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$R/\\Delta x$")
    ax.set_ylabel("$\\Delta p/(\\sigma/R)$")
    ax.set_title("Laplace law, two-phase 1000:1\n(faint = water-only reading)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[1]
    rows = [r for r in summary if r["assembly"] == "consistent" and r["hourglass"]
            and r["bracket"] == "symmetric"]
    for mode, style in (("two_phase", "o-"), ("single_phase", "^-")):
        sub = [r for r in rows if r["mode"] == mode]
        ax.plot([r["radius_over_dx"] for r in sub],
                [r["dp_flux_all_err_pct"] for r in sub], style, label=mode)
    ax.axhspan(-GATE_PCT, GATE_PCT, color="0.85", label="$\\pm15\\%$ gate")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$R/\\Delta x$")
    ax.set_ylabel("error vs $\\sigma/R$ [%]")
    ax.set_title("consistent assembly: both-phase reading")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    fk = [r["contact_fraction_water"] for r in ratios]
    ax.plot(fk, [r["delta_integral"] for r in ratios], "o-", label="$\\int|\\nabla c|\\,dn$")
    ax.plot(fk, [r["dp_flux_all_pa"] / r["theory_pa"] for r in ratios], "s-",
            label="$\\Delta p/(\\sigma/R)$")
    for xx, r in zip(fk, ratios):
        ax.annotate(f"{r['rho_air_over_rho_water']:g}", (xx, r["delta_integral"]),
                    textcoords="offset points", xytext=(3, 4), fontsize=7)
    ax.axhline(1.0, color="0.4", ls=":", lw=1.0)
    ax.set_xlabel("contact fraction $f_w$ (mixture material)")
    ax.set_ylabel("dimensionless")
    ax.set_title("density-ratio sweep (labels: $\\rho_a/\\rho_w$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radii-dx", default="16,32")
    ap.add_argument("--rho-ratios", default="1,0.1,0.01,0.001",
                    help="rho_air/rho_water values (upstream case: 0.001)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mad = [float(x) for x in args.radii_dx.split(",")]

    floor = [operator_floor(RADIUS, RADIUS / rdx) for rdx in mad]
    for row in floor:
        print(f"[operator] R/dx={row['radius_over_dx']:.0f} dp/(sigma/R)="
              f"{row['dp_pa']/(SIGMA/RADIUS):.4f} err={row['err_pct']:.2f}%", flush=True)

    summary = []
    # gated / reported configurations: (assembly, bracket, hourglass)
    combos = [("upstream", "mixture", True),      # faithful SPHinXsys transcription
              ("consistent", "mixture", True),
              ("consistent", "symmetric", True),  # gated configuration
              ("consistent", "symmetric", False), # momentum-exact ablation
              ("upstream", "symmetric", True)]
    for rdx in mad:
        dx = RADIUS / rdx
        for mode in ("two_phase", "single_phase"):
            for assembly, bracket, hg in combos:
                row = measure(RADIUS, dx, RHO_AIR, mode, assembly, hg, bracket=bracket)
                summary.append(row)
                print(f"[ladder] R/dx={rdx:>4.0f} {mode:>12} {assembly:>10} "
                      f"{bracket:>9} hg={str(hg):>5} "
                      f"stress={row['dp_stress_err_pct']:7.2f}% "
                      f"flux_water={row['dp_flux_water_err_pct']:8.2f}% "
                      f"flux_all={row['dp_flux_all_err_pct']:7.2f}% "
                      f"mom={row['momentum_residual']:.1e}", flush=True)

    # normalisation identity: consistent / upstream == particle mass
    ident = []
    for mode in ("two_phase", "single_phase"):
        up = next(r for r in summary if r["mode"] == mode and r["assembly"] == "upstream"
                  and r["bracket"] == "mixture"
                  and r["hourglass"] and r["radius_over_dx"] == mad[0])
        co = next(r for r in summary if r["mode"] == mode and r["assembly"] == "consistent"
                  and r["bracket"] == "mixture"
                  and r["hourglass"] and r["radius_over_dx"] == mad[0])
        ident.append({
            "mode": mode,
            "ratio_consistent_over_upstream": co["dp_flux_all_pa"] / up["dp_flux_all_pa"],
            "particle_mass_water": co["particle_mass_water"],
            "rel_deviation": abs(co["dp_flux_all_pa"] / up["dp_flux_all_pa"]
                                 - co["particle_mass_water"])
                            / co["particle_mass_water"],
        })

    ratios = [measure(RADIUS, RADIUS / mad[0], RHO_WATER * r, "two_phase",
                      "consistent", True, bracket="symmetric")
              for r in [float(x) for x in args.rho_ratios.split(",")]]
    for row in ratios:
        print(f"[ratio] rho_a/rho_w={row['rho_air_over_rho_water']:<7g} "
              f"f_w={row['contact_fraction_water']:.4f} "
              f"int|gc|dn={row['delta_integral']:.4f} "
              f"flux_all={row['dp_flux_all_err_pct']:7.2f}%", flush=True)

    result = {
        "reference": {
            "model": "SPHinXsys continuous surface stress + mixture contact fraction",
            "raw_sources": "docs/xval_sphinxys_surface_tension/raw/ (sha256 in MANIFEST.json)",
            "theory": "Young-Laplace, 2-D circular interface: Delta p = sigma/R",
            "sigma_n_m": SIGMA,
            "radius_m": RADIUS,
            "rho_water": RHO_WATER,
            "rho_air": RHO_AIR,
            "h_over_dx": H_OVER_DX,
        },
        "operator_floor": floor,
        "ladder": summary,
        "normalisation_identity": ident,
        "density_ratio_sweep": ratios,
        "gate": gate(summary),
        "wall_clock_s": time.time() - t0,
    }
    (out / "gate.json").write_text(json.dumps(result, indent=1))
    if not args.no_plot:
        plot(summary, ratios, out / "gate.png")
    print(json.dumps(result["gate"], indent=1), flush=True)
    print(f"wall_clock_s={result['wall_clock_s']:.1f}", flush=True)
    return result


if __name__ == "__main__":
    main()
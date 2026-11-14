"""Free-surface (single-phase) closure of the SPHinXsys CSS surface-tension flux.

Anchor: XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION.

Problem
-------
In ``mode="single_phase"`` the mixture model is reduced to a free surface: the
second phase is replaced by the kernel-sum identity.  The colour gradient then
reproduces the surface delta to ~2 %, but the *stress-flux* sum runs over the
present half-support only, so the Laplace reading is short by ~30 %
(``out_single_phase_completion/gate.json``).  The previous closure multiplied
the specular image of the present neighbours by a fitted coefficient
``alpha* = 0.4263`` -- a calibration, not a derivation.

This benchmark tests a parameter-free closure instead: the kernel-support
fraction ``s_i = trace(M_i)/2`` (``M`` = first-moment completeness matrix) is
1 in the bulk and 1/2 on an exact free surface, so ``1/s`` restores exactly the
missing part of the flux.  The factor is evaluated per *pair* and applied to
both ends, which keeps momentum conservation exact.

Protocol
--------
* Ladder R/dx = 16/32/64: variants ``none`` (shipped), ``mirror`` (alpha=1),
  ``mirror`` with the previously fitted ``alpha*``, ``completeness``.
* Cross-parameter sweep at R/dx = 32: kernel {wendland, cubic} x h/dx {1.3, 1.5, 2.0}.
* Every number is written to ``out_free_surface_completeness/gate.json``.

Usage::

    PYTHONPATH=src python benchmarks/surface_tension/free_surface_completeness.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

import tensorsph.sphinxys_css as sc
from tensorsph.neighbor_search import cell_linked_pairs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import single_phase_completion as SPC  # noqa: E402  (case + reading, unchanged)

SIGMA, RADIUS, RHO, H_OVER_DX = SPC.SIGMA, SPC.RADIUS, SPC.RHO_WATER, SPC.H_OVER_DX
GATE_PCT = 15.0
LADDER = (16, 32, 64)
PREVIOUS_ALPHA = 0.4263334171275849      # single_phase_completion out/gate.json: fit.alpha_star
KERNELS = ("wendland", "cubic")
H_SWEEP = (1.3, 1.5, 2.0)
OUT = Path(__file__).resolve().parent / "out_free_surface_completeness"


def measure(radius_over_dx: float, completion: str = "none", weight: float = 1.0,
            kernel: str = "wendland", h_over_dx: float = H_OVER_DX) -> dict:
    """Laplace reading ``dp = sum rho (-a.n) dx^2 / (2 pi R)`` for one configuration."""
    dx = RADIUS / radius_over_dx
    h = h_over_dx * dx
    pos, r, rho0, vol = SPC.make_disc(RADIUS, dx)
    mass = rho0 * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs, None, SIGMA, h,
        frac=torch.ones(pos.shape[0], dtype=pos.dtype), mode="single_phase",
        kernel=kernel, delta="sphinxys", hourglass=False, assembly="consistent",
        completion=completion, completion_weight=weight)
    n_out = pos / r.clamp_min(1e-300)[:, None]
    force_in = -rho0 * (acc * n_out).sum(dim=1) * (dx * dx)
    dp = float(force_in.sum()) / (2.0 * math.pi * RADIUS)
    theory = SIGMA / RADIUS
    support = sc.support_fraction(pos, vol, pairs, h, kernel)
    pair_factor = sc.pair_renormalisation(support, pairs)
    mom = (rho0 * vol)[:, None].mul(acc).sum(dim=0)
    scale = float((rho0 * vol * torch.linalg.vector_norm(acc, dim=1)).sum())
    band = fields.delta_s > 0.05 * fields.delta_s.max()
    return {
        "radius_over_dx": radius_over_dx,
        "dx_m": dx,
        "particles": int(pos.shape[0]),
        "kernel": kernel,
        "h_over_dx": h_over_dx,
        "completion": completion,
        "completion_weight": weight,
        "dp_pa": dp,
        "dp_theory_pa": theory,
        "err_pct": (dp - theory) / theory * 100.0,
        "delta_integral": float((fields.delta_s * vol).sum()) / (2.0 * math.pi * RADIUS),
        "support_fraction_band_mean": float(support[band].mean()),
        "support_fraction_bulk_min": float(support[support > 0.9].min())
        if bool((support > 0.9).any()) else float("nan"),
        "pair_factor_band_mean": float(pair_factor[
            band[pairs[:, 0]] | band[pairs[:, 1]]].mean()) if pairs.numel() else float("nan"),
        "momentum_residual": float(torch.linalg.vector_norm(mom)) / max(scale, 1e-300),
    }


def plot(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    for name, style in (("none", "o-"), ("mirror", "s-"), ("completeness", "^-")):
        rows = [r for r in payload["ladder"] if r["completion"] == name]
        ax.plot([r["radius_over_dx"] for r in rows], [r["err_pct"] for r in rows],
                style, label={"none": "one-sided (shipped)",
                              "mirror": "specular mirror, alpha=1",
                              "completeness": "support completeness (this work)"}[name])
    rows = [r for r in payload["ladder"] if r["completion"] == "mirror"
            and abs(r["completion_weight"] - PREVIOUS_ALPHA) < 1e-12]
    ax.plot([r["radius_over_dx"] for r in rows], [r["err_pct"] for r in rows], "x--",
            label=f"mirror, fitted alpha*={PREVIOUS_ALPHA:.3f}")
    ax.axhspan(-GATE_PCT, GATE_PCT, color="0.85", zorder=0, label="15 % gate")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("R / dx"); ax.set_ylabel("Laplace reading error [%]")
    ax.set_title("free-surface Laplace reading"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    sup = payload["support_profile"]
    ax.plot(sup["depth_over_dx"], sup["support_fraction"], "o-", label="measured s_i")
    ax.axhline(0.5, color="r", ls=":", label="exact half-space = 1/2")
    ax.axhline(1.0, color="k", ls=":", label="complete support = 1")
    ax.set_xlabel("inward depth from the free surface / dx")
    ax.set_ylabel("support fraction s = trace(M)/2")
    ax.set_title("kernel-support measurement"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    rows = payload["sweep"]
    labels = [f"{r['kernel']}\nh={r['h_over_dx']}" for r in rows]
    ax.bar(range(len(rows)), [r["err_pct"] for r in rows], color="tab:blue")
    ax.axhspan(-GATE_PCT, GATE_PCT, color="0.85", zorder=0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Laplace reading error [%]")
    ax.set_title("cross-parameter sweep, R/dx = 32")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("SPHinXsys CSS free-surface closure: parameter-free support completeness")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def support_profile(radius_over_dx: float = 32.0) -> dict:
    """``s_i`` as a function of the inward distance from the free surface."""
    dx = RADIUS / radius_over_dx
    h = H_OVER_DX * dx
    pos, r, _rho0, vol = SPC.make_disc(RADIUS, dx)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    s = sc.support_fraction(pos, vol, pairs, h, "wendland")
    depth = (RADIUS - r) / dx
    edges = torch.arange(0.0, float(depth.max()) + 1.0, 1.0)
    centres, means = [], []
    for k in range(len(edges) - 1):
        m = (depth >= edges[k]) & (depth < edges[k + 1])
        if int(m.sum()) == 0:
            continue
        centres.append(0.5 * float(edges[k] + edges[k + 1]))
        means.append(float(s[m].mean()))
    return {"radius_over_dx": radius_over_dx, "depth_over_dx": centres,
            "support_fraction": means,
            "floor": float(s.min()), "bulk_max": float(s.max())}


def main() -> int:
    t0 = time.time()
    ladder = []
    for rdx in LADDER:
        for completion in ("none", "mirror", "completeness"):
            ladder.append(measure(rdx, completion=completion, weight=1.0))
        ladder.append(measure(rdx, completion="mirror", weight=PREVIOUS_ALPHA))
    sweep = [measure(32.0, completion="completeness", kernel=k, h_over_dx=hv)
             for k in KERNELS for hv in H_SWEEP]
    payload = {
        "reference": {
            "oss": "Xiangyu-Hu/SPHinXsys",
            "model": "continuous surface stress + mixture contact fraction, free-surface reduction",
            "theory": "Young-Laplace, 2-D circular interface: dp = sigma / R",
            "sigma_n_m": SIGMA, "radius_m": RADIUS, "rho_water": RHO, "h_over_dx": H_OVER_DX,
            "raw_sources": "docs/xval_sphinxys_surface_tension/raw/",
        },
        "gate_pct": GATE_PCT,
        "closure": {
            "name": "support completeness",
            "matrix": "M_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij",
            "factor": "per pair: 1 / [(s_i + s_j)/2],  s_i = trace(M_i)/2",
            "free_parameters": 0,
            "momentum": "pair-symmetric factor keeps pair contributions equal and opposite",
        },
        "ladder": ladder,
        "sweep": sweep,
        "support_profile": support_profile(),
    }
    closed = [r for r in ladder if r["completion"] == "completeness"]
    payload["gate"] = {
        "criterion_pct": GATE_PCT,
        "ladder_err_pct": {f"R_over_dx={r['radius_over_dx']:.0f}": r["err_pct"] for r in closed},
        "worst_abs_err_pct": max(abs(r["err_pct"]) for r in closed),
        "within_3pct": {f"R_over_dx={r['radius_over_dx']:.0f}": abs(r["err_pct"]) < 3.0
                        for r in closed},
        "baseline_one_sided_worst_abs_pct": max(
            abs(r["err_pct"]) for r in ladder if r["completion"] == "none"),
        "closure_gain_factor": max(abs(r["err_pct"]) for r in ladder if r["completion"] == "none")
        / max(abs(r["err_pct"]) for r in closed),
        "sweep_worst_abs_pct": max(abs(r["err_pct"]) for r in sweep),
        "verdict": "pass" if max(abs(r["err_pct"]) for r in closed) < GATE_PCT
        and max(abs(r["err_pct"]) for r in sweep) < GATE_PCT else "fail",
    }
    payload["runtime_s"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(payload, indent=1))
    plot(payload, OUT / "gate.png")
    print(json.dumps(payload["gate"], indent=1))
    for r in ladder:
        print(f"R/dx={r['radius_over_dx']:>4} {r['completion']:>12} w={r['completion_weight']:.3f}"
              f"  dp={r['dp_pa']:.5f}  err={r['err_pct']:+.2f}%  mom={r['momentum_residual']:.1e}")
    for r in sweep:
        print(f"sweep {r['kernel']:>8} h={r['h_over_dx']}  err={r['err_pct']:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
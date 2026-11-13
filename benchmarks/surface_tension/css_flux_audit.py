"""Adversarial audit of the SPHinXsys CSS surface-tension *flux* reading.

Anchor: XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION.

Background.  The earlier cross-validation (m17909, `sphinxys_xval.py`,
`docs/xval_sphinxys_surface_tension/`) reported that the ported force assembly
(inner + contact stress flux) reproduces only ~1% of the Young-Laplace jump and
attributed it to "the discrete divergence of the tangential stress cancelling
on a particle lattice", i.e. an inherent property of the CSS formulation.

This script re-derives that number from scratch and tests the attribution with
four independent controls:

O1  operator calibration on a plain lattice: the conservative pair divergence
    ``sum_j V_j (S_i + S_j) . grad W_ij`` applied to ``S = x^2 I``, whose exact
    divergence is ``2 x`` (this is a pure operator test: no surface tension),
O2  reproduce the m17909 flux reading (ported assembly, uncorrected sums),
O3  same assembly with the quantity normalisation made explicit: the ported
    acceleration carries the upstream prefactor ``(m_i/rho_i) = V_i``, so the
    force density is recovered as ``grad.S = a_i / V_i``.  Both readings are
    reported so the deficit can be attributed to physics or to bookkeeping,
O4  per-particle comparison of the ported acceleration against the exact
    curvature force ``-sigma |grad c| n / (rho r)`` over the interface band.

Outputs: ``out_css_flux_audit/audit.json`` and ``audit.png``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

from tensorsph import sphinxys_css as sc
from tensorsph.neighbor_search import cell_linked_pairs

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sphinxys_xval", _HERE / "sphinxys_xval.py")
_xval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xval)

SIGMA = _xval.SIGMA
RADIUS = _xval.RADIUS
RHO_WATER = _xval.RHO_WATER
RHO_AIR = _xval.RHO_AIR
DP_RATIO = _xval.DP_RATIO
RINGS = _xval.RINGS
OUT = _HERE / "out_css_flux_audit"
THEORY = SIGMA / RADIUS
HOURGLASS = False          # isolate the stress flux (the hourglass penalty is
HOURGLASS_COEFF = 0.0      # a separate, non-conservative model choice)


def _radial_frame(pos):
    centre = pos.mean(dim=0)
    radial = pos - centre
    r_loc = torch.linalg.vector_norm(radial, dim=1)
    return r_loc, radial / r_loc.clamp_min(1e-30)[:, None]


def laplace_from_density(pos, density, vol, mask):
    """``dp = sum_i density_i A_i / (2 pi r_c)`` with ``r_c`` the centroid radius.

    ``density`` is the *force per unit volume* whose volume integral is the
    inward radial force per unit length of interface.  ``A_i = vol_i``.
    """
    r_loc, _ = _radial_frame(pos)
    weight = density[mask] * vol[mask]
    total = float(weight.sum())
    r_c = float((weight * r_loc[mask]).sum() / total) if abs(total) > 1e-30 else float("nan")
    return total / (2.0 * math.pi * r_c), r_c, total


def pair_divergence(pos, vol, pairs, stress, h, kernel="wendland"):
    """``acc_i = V_i sum_j V_j (S_i + S_j) . grad_i W_ij`` (upstream inner form)."""
    acc = torch.zeros_like(pos)
    if pairs.numel() == 0:
        return acc
    i, j = pairs.unbind(dim=1)
    gw = sc._grad(pos[i] - pos[j], h, kernel)
    flux = torch.einsum("pab,pb->pa", stress[i] + stress[j], gw)
    coarse = vol[i] * vol[j]
    acc = acc.index_add(0, i, coarse[:, None] * flux)
    acc = acc.index_add(0, j, -coarse[:, None] * flux)
    return acc


# --------------------------------------------------------------------------- #
# O1: operator calibration (no surface tension involved)
# --------------------------------------------------------------------------- #
def operator_calibration(h_over_dx: float = 1.3, half_width: int = 15) -> dict:
    dx, vol = 1.0, 1.0
    h = h_over_dx * dx
    axis = torch.arange(-half_width, half_width, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    x = pos[:, 0]
    stress = torch.zeros(pos.shape[0], 2, 2, dtype=torch.float64)
    stress[:, 0, 0] = x ** 2
    stress[:, 1, 1] = x ** 2
    div = pair_divergence(pos, torch.full_like(x, vol), pairs, stress, h)
    rows = []
    for probe in (0.0, 2.0, 5.0):
        k = int(torch.nonzero((pos[:, 0] == probe) & (pos[:, 1] == 0))[0])
        rows.append({"x": probe, "pair_div": float(div[k, 0]), "exact": 2.0 * probe,
                     "err_pct": (float(div[k, 0]) - 2.0 * probe) / (2.0 * probe) * 100.0
                     if probe else 0.0})
    worst = max(abs(r["err_pct"]) for r in rows)
    return {"operator": "sum_j V_j (S_i+S_j).gradW_ij", "field": "S = x^2 I",
            "exact_div": "2x", "h_over_dx": h_over_dx, "probes": rows,
            "worst_err_pct": worst,
            "verdict": "correct divergence operator" if worst < 5.0 else "operator defect"}


# --------------------------------------------------------------------------- #
# O2/O3/O4: the droplet
# --------------------------------------------------------------------------- #
def droplet_controls(radius_over_dx: int) -> dict:
    dx = RADIUS / radius_over_dx
    h = DP_RATIO * dx
    pos, r, phase, rho0, vol = _xval.make_two_phase(RADIUS, dx, RHO_AIR, RINGS)
    mass = rho0 * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    same = phase[i] == phase[j]
    pairs_inner, pairs_cross = pairs[same], pairs[~same]
    frac = sc.contact_fraction(rho0, torch.where(phase == 0, RHO_AIR, RHO_WATER))
    water = phase == 0

    acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs_inner, pairs_cross, SIGMA, h, frac=frac,
        mode="two_phase", delta="sphinxys", hourglass=HOURGLASS,
        hourglass_coeff=HOURGLASS_COEFF)

    r_loc, n_out = _radial_frame(pos)
    inward_acc = -(acc * n_out).sum(dim=1)           # a . (-n), inward > 0

    # O2: exactly the m17909 reading: sum of rho0 * a * A
    dp_ported, rc_ported, _ = laplace_from_density(pos, rho0 * inward_acc, vol, water)

    eye = torch.eye(2, dtype=pos.dtype)

    # O3: same assembly, quantity bookkeeping made explicit.  The ported
    # acceleration is V_i * (grad . S) because upstream multiplies by
    # mass_i/rho_i = V_i; the volume-integrated force density is therefore
    # V_i grad.S = (rho0_i a_i V_i) / m_i, i.e. the reported reading divided by
    # the particle mass m_i = rho0_i V_i.
    dp_normalised, _, _ = laplace_from_density(
        pos, (rho0 * inward_acc) / mass, vol, water)

    # O5: operator + estimator control on the *idealised* CSS field.
    # S = sigma f(r) (I - n n^T) with the exact radial normal and an exactly
    # normalised one-sided radial delta.  Swept over the interface width and
    # over lattice shifts so that a lattice-symmetry artefact cannot be
    # mistaken for an algebraic property of the operator.
    ideal = []
    for width in (1.0, 2.0, 3.0):
        for shift in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.3), (0.25, 0.5)):
            pos_s = pos + torch.tensor(shift, dtype=pos.dtype) * dx
            pairs_s = cell_linked_pairs(pos_s, 2.0 * h)
            si, sj = pairs_s.unbind(dim=1)
            inner_s = pairs_s[phase[si] == phase[sj]]
            r_s, n_s = _radial_frame(pos_s)
            f_s = torch.where(r_s < RADIUS,
                              2.0 * torch.exp(-((r_s - RADIUS) / (width * dx)) ** 2)
                              / (width * dx * math.sqrt(math.pi)),
                              torch.zeros_like(r_s))
            s_s = SIGMA * f_s[:, None, None] * (eye - torch.einsum("na,nb->nab", n_s, n_s))
            w_s = pair_divergence(pos_s, vol, inner_s, s_s, h)
            d_s = -(w_s * n_s).sum(dim=1)
            wt = d_s[water] * vol[water]
            tot = float(wt.sum())
            rc = float((wt * r_s[water]).sum() / tot) if abs(tot) > 1e-30 else float("nan")
            dp_s = tot / (2.0 * math.pi * rc) if abs(tot) > 1e-30 else 0.0
            ideal.append({"width_over_dx": width, "shift_over_dx": list(shift),
                          "dp_pa": dp_s, "dp_ratio": dp_s / THEORY})
    worst_ideal = max(abs(row["dp_ratio"]) for row in ideal)

    # surface delta normalisation and per-particle force comparison (O4)
    delta_integral = float((fields.delta_s[water] * vol[water]).sum()) / (2.0 * math.pi * RADIUS)
    band = water & (fields.delta_s * dx > 0.05)
    a_exact = SIGMA * fields.delta_s / (rho0 * r_loc.clamp_min(1e-30))
    ratio = float(inward_acc[band].abs().mean() / a_exact[band].mean())
    predicted = float((rho0 * vol)[water].mean())

    return {
        "radius_over_dx": radius_over_dx,
        "dx_m": dx,
        "particles": int(pos.shape[0] // 1),
        "water_particles": int(water.sum()),
        "delta_integral": delta_integral,
        "theory_pa": THEORY,
        "dp_ported_pa": dp_ported,
        "dp_ported_err_pct": (dp_ported - THEORY) / THEORY * 100.0,
        "dp_normalised_pa": dp_normalised,
        "dp_normalised_err_pct": (dp_normalised - THEORY) / THEORY * 100.0,
        "flux_centroid_radius_m": rc_ported,
        "band_particles": int(band.sum()),
        "particle_mass_rho0_vol": predicted,
        "band_force_ratio": ratio,
        "band_force_ratio_over_mass": float(ratio / predicted),
        "dp_ported_over_mass_ratio": float((dp_ported / predicted) / THEORY),
        "ideal_css_field_flux": ideal,
        "ideal_css_field_worst_ratio": worst_ideal,
        "band_particles_over_mass": float(ratio / predicted),
    }


def plot(rows, cal, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    res = [r["radius_over_dx"] for r in rows]
    ax.plot(res, [r["dp_normalised_pa"] / THEORY for r in rows], "o-",
            label=r"normalised reading $\sum a_i A_i/2\pi r_c$")
    ax.plot(res, [r["dp_ported_pa"] / THEORY for r in rows], "s--",
            label=r"as-reported $\sum \rho_0 a_i A_i/2\pi r_c$ (m17909)")
    ax.plot(res, [r["particle_mass_rho0_vol"] for r in rows], "^:",
            label=r"$m_i=\rho_0\Delta x^2$ (bookkeeping factor)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axhline(0.85, color="r", lw=0.6, ls=":")
    ax.axhline(1.15, color="r", lw=0.6, ls=":")
    ax.set_xlabel("R / dx")
    ax.set_ylabel(r"$\Delta p\,/\,(\sigma/R)$")
    ax.set_title("Laplace flux reading: as reported vs normalised")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    probes = [p["x"] for p in cal["probes"]]
    ax.plot(probes, [p["exact"] for p in cal["probes"]], "k-", label="exact div(S)")
    ax.plot(probes, [p["pair_div"] for p in cal["probes"]], "o--",
            label="conservative pair form")
    ax.set_xlabel("x (lattice, dx = 1)")
    ax.set_ylabel(r"$\nabla\cdot S$")
    ax.set_title(r"O1 operator calibration: $S=x^2 I$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="16,32,64")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    cal = operator_calibration()
    print(f"O1 operator calibration: worst err {cal['worst_err_pct']:.2f}% -> {cal['verdict']}")
    for p in cal["probes"]:
        print(f"    x={p['x']:.1f}  pair_div={p['pair_div']:.4f}  exact={p['exact']:.4f}")

    rows = [droplet_controls(int(v)) for v in args.levels.split(",")]
    print(f"\n{'R/dx':>5} {'dp_ported':>10} {'err%':>9} {'dp_norm':>10} {'err%':>9} "
          f"{'m_i':>9} {'|f|band/|f_exact|':>17} {'ratio/m_i':>10}")
    for r in rows:
        print(f"{r['radius_over_dx']:5d} {r['dp_ported_pa']:10.5f} {r['dp_ported_err_pct']:9.2f} "
              f"{r['dp_normalised_pa']:10.5f} {r['dp_normalised_err_pct']:9.2f} "
              f"{r['particle_mass_rho0_vol']:9.5f} {r['band_force_ratio']:17.5f} "
              f"{r['band_force_ratio_over_mass']:10.3f}")

    payload = {
        "audit": "XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION / m17909 flux-reading claim",
        "model": "SPHinXsys CSS (Zhang & Hu) as ported in src/tensorsph/sphinxys_css.py",
        "sigma": SIGMA, "radius": RADIUS, "rho_ratio": RHO_WATER / RHO_AIR,
        "h_over_dx": DP_RATIO, "hourglass": HOURGLASS, "theory_pa": THEORY,
        "operator_calibration": cal,
        "droplet": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2))
    plot(rows, cal, OUT / "audit.png")
    print(f"\nwrote {OUT/'audit.json'} and {OUT/'audit.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
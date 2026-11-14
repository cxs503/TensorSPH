"""Lamb-mode (dynamic) cross-validation of the ported SPHinXsys CSS surface tension.

Anchor: ``XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION``.

Why this script exists
----------------------
The static Young-Laplace audit of the port left a *contradiction* in the record:

* ``benchmarks/surface_tension/sphinxys_xval.py`` (m17909) reported that the
  ported **force assembly** reproduces -97 % to -99.8 % of ``sigma/R`` and
  concluded that the CSS flux is *structurally blind* to curvature;
* ``benchmarks/surface_tension/css_flux_audit.py`` re-derived the same number,
  showed that the ported acceleration carries an extra ``V_i`` bookkeeping
  factor (``a_i = V_i div S``), and after that normalisation the deficit is
  about **-30 %** -- but its own "ideal CSS field" control (a synthetic
  tangential stress with a prescribed delta profile) returned a ratio of
  ``3e-5``, i.e. *complete* cancellation.

Both statements cannot hold at once, and neither is a physical test: a static
circular droplet is the geometry on which a lattice-symmetric cancellation is
most likely, so it cannot separate "the operator cancels by symmetry" from
"the model has no curvature force".

This script measures the surface-tension force on a **geometrically perturbed**
interface, where the lattice symmetry is broken, and expresses it in the one
unit that is independent of bookkeeping: the **frequency of the small-amplitude
shape oscillation** (Lamb / Rayleigh mode ``n``).

    r(theta) = R (1 + eps cos n theta)

    exact 2-D reference (linear theory, e.g. Lamb 1932, art. 263;
    a cylinder is used, hence no (n+2) factor):
        Delta p_n = sigma (n^2 - 1) eps / R
        omega_n^2 = n (n^2 - 1) sigma / (rho R^3)

Both models are measured with **the same projector on the same geometry**:

    dp_mode = 2 / L * sum_i (f_i . e_r) V_i cos(n theta_i),   L = 2 pi R

``f`` is the *force per unit volume* of the model (2-D: N/m^2), so ``dp_mode``
is directly comparable to ``Delta p_n`` and ``omega_ratio = sqrt(dp/dp_exact)``.

* Brackbill CSF (``tensorsph.surface_tension_dispersion``, curvature from a
  quadratic fit) is the **calibration channel**: its Lamb-mode frequency is
  already known to sit within a few percent (``TOL_LAMB_PCT`` in
  ``surface_tension_dispersion``), so it proves the projector is wired to the
  right normalisation.  It is run through ``drop_modal_stiffness`` itself.
* The SPHinXsys CSS port (``tensorsph.sphinxys_css``) is the **subject**.  Its
  acceleration carries ``(m_i/rho_i) = V_i`` (upstream transcription) or
  ``1/rho_i`` (``assembly="consistent"``); the force density is therefore
  ``a_i / V_i`` resp. ``rho_i a_i`` -- the relation validated independently by
  the O1 operator calibration in ``css_flux_audit.py`` (2.6 % error on
  ``div(S = x^2 I) = 2x``).

Outputs: ``out_sphinxys_lamb/lamb.json``, ``lamb.png``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from tensorsph import sphinxys_css as sc
from tensorsph.neighbor_search import cell_linked_pairs
from tensorsph.surface_tension_dispersion import drop_geometry, drop_modal_stiffness

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sphinxys_xval", _HERE / "sphinxys_xval.py")
_xval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xval)

SIGMA = sc.SPHINXSYS_SIGMA if hasattr(sc, "SPHINXSYS_SIGMA") else 0.072
RHO0 = 1000.0            # water at 20 C, kg/m^3
RHO_AIR = 1.0            # 1000:1, the upstream square-droplet ratio
RADIUS = 0.1             # m
H_FACTOR = sc.SPHINXSYS_H_SPACING_RATIO      # 1.3, upstream adaptation.h
HOURGLASS = sc.SPHINXSYS_HOURGLASS_COEFF     # 4.5, upstream default
MODE = 2                 # Lamb mode
EPS = 0.10               # the drop_geometry default used by the CSF Lamb case
RINGS = 6                # air rings (kernel support 2h = 2.6 dx)
TOL_LAMB_PCT = 3.0       # platform gate already used by the CSF Lamb case
OUT = _HERE / "out_sphinxys_lamb"


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def two_phase_lobe_geometry(mode: int, eps: float, radius: float, dx: float,
                            rho_air: float = RHO_AIR, rings: int = RINGS):
    """Lobed water disc ``r <= R(1+eps cos n theta)`` plus ``rings`` air rings.

    Square lattice, one spacing for both phases (as upstream: both bodies share
    ``particle_spacing_ref``).  ``vol = dx^2`` for every particle.
    """
    span = radius + (rings + 1) * dx
    axis = torch.arange(-span, span + 0.5 * dx, dx, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="xy")
    pts = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    r = torch.linalg.vector_norm(pts, dim=1)
    theta = torch.atan2(pts[:, 1], pts[:, 0])
    r_surf = radius * (1.0 + eps * torch.cos(mode * theta))
    keep = r <= r_surf.max() + rings * dx + 1.0e-12
    pos = pts[keep].clone()
    r, theta, r_surf = r[keep], theta[keep], r_surf[keep]
    phase = (r > r_surf + 1.0e-12).to(torch.int64)          # 0 = water, 1 = air
    rho0 = torch.where(phase == 0, torch.tensor(RHO0, dtype=pos.dtype),
                       torch.tensor(rho_air, dtype=pos.dtype))
    vol = torch.full((pos.shape[0],), dx * dx, dtype=pos.dtype)
    return dict(pos=pos, r=r, theta=theta, phase=phase, rho0=rho0, vol=vol,
                radius=radius, eps=eps, mode=mode, dx=dx)


def lobe_geometry(mode: int, eps: float, radius: float, dx: float) -> dict:
    """Single-phase lobed drop, identical to the geometry of the CSF Lamb case."""
    geom = drop_geometry(mode, eps, radius, dx)
    geom["vol"] = geom["mass"] / RHO0
    geom["theta"] = torch.atan2(geom["pos"][:, 1], geom["pos"][:, 0])
    return geom


# --------------------------------------------------------------------------- #
# projector (shared by both models)
# --------------------------------------------------------------------------- #
def modal_pressure(pos: torch.Tensor, theta: torch.Tensor, force_density: torch.Tensor,
                   vol: torch.Tensor, mode: int, radius: float,
                   mask: torch.Tensor | None = None) -> dict:
    """``dp = 2/L sum_i (f_i . e_r) V_i cos(n theta_i)`` -- the normalisation used by
    :func:`tensorsph.surface_tension_dispersion.drop_modal_stiffness`."""
    r = torch.linalg.vector_norm(pos, dim=1)
    e_r = pos / r.clamp_min(1e-300)[:, None]
    f_r = (force_density * e_r).sum(dim=1)
    cos_nt = torch.cos(mode * theta)
    w = vol if mask is None else torch.where(mask, vol, torch.zeros_like(vol))
    proj = float((f_r * w * cos_nt).sum())
    length = 2.0 * math.pi * radius
    return {"dp_measured": 2.0 / length * proj,
            "interface_length": length,
            "radial_force_total": float((f_r * w).sum())}


def modal_summary(dp_measured: float, mode: int, radius: float, eps: float,
                  sigma: float = SIGMA, rho0: float = RHO0) -> dict:
    """Attach the exact reference and the implied frequency."""
    kappa_exact = (mode * mode - 1.0) * eps / radius
    dp_exact = -sigma * kappa_exact
    ratio = dp_measured / dp_exact if dp_exact else float("nan")
    sign = math.copysign(1.0, ratio) if ratio == ratio else 0.0
    omega_theory = math.sqrt(mode * (mode * mode - 1.0) * sigma / (rho0 * radius ** 3))
    omega_ratio = sign * math.sqrt(abs(ratio))
    return {"dp_measured_pa": dp_measured, "dp_analytic_pa": dp_exact,
            "stiffness_ratio": ratio, "omega_theory": omega_theory,
            "omega_measured": omega_ratio * omega_theory, "omega_ratio": omega_ratio,
            "omega_error_pct": (omega_ratio - 1.0) * 100.0}


# --------------------------------------------------------------------------- #
# the subject: the ported CSS model
# --------------------------------------------------------------------------- #
def css_single_phase(geom: dict, *, sigma: float = SIGMA, hourglass: bool = True,
                     hourglass_coeff: float = HOURGLASS,
                     delta: str = "sphinxys",
                     assembly: str = "upstream") -> dict:
    pos, vol = geom["pos"], geom["vol"]
    dx = geom["dx"]
    h = H_FACTOR * dx
    rho = torch.full_like(vol, RHO0)
    mass = rho * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho, vol, pairs, None, sigma, h, mode="single_phase",
        delta=delta, hourglass=hourglass, hourglass_coeff=hourglass_coeff,
        assembly=assembly)
    # a_i = (m_i/rho_i) div S = V_i div S  (upstream)  or  (1/rho_i) div S (consistent)
    if assembly == "upstream":
        force_density = acc / vol[:, None]
    elif assembly == "consistent":
        force_density = rho[:, None] * acc
    else:  # pragma: no cover - argument validation
        raise ValueError(assembly)
    proj = modal_pressure(pos, geom["theta"], force_density, vol, geom["mode"],
                          geom["radius"])
    out = modal_summary(proj["dp_measured"], geom["mode"], geom["radius"], geom["eps"])
    out.update(force_density_definition=("a_i / V_i" if assembly == "upstream"
                                        else "rho_i a_i"),
               delta_integral=float((fields.delta_s * vol).sum()
                                    / (2.0 * math.pi * geom["radius"])),
               delta=delta, hourglass=bool(hourglass),
               hourglass_coeff=float(hourglass_coeff), assembly=assembly,
               n_particles=int(pos.shape[0]))
    return out


def css_two_phase(geom: dict, *, sigma: float = SIGMA, hourglass: bool = True,
                  hourglass_coeff: float = HOURGLASS, delta: str = "sphinxys",
                  assembly: str = "upstream", bracket: str = "mixture") -> dict:
    pos, vol, phase, rho0 = geom["pos"], geom["vol"], geom["phase"], geom["rho0"]
    h = H_FACTOR * geom["dx"]
    mass = rho0 * vol
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    same = phase[i] == phase[j]
    pairs_inner, pairs_cross = pairs[same], pairs[~same]
    frac = sc.contact_fraction(rho0, torch.where(phase == 0, RHO_AIR, RHO0))
    acc, fields = sc.surface_tension_acceleration(
        pos, mass, rho0, vol, pairs_inner, pairs_cross, sigma, h, frac=frac,
        mode="two_phase", delta=delta, hourglass=hourglass,
        hourglass_coeff=hourglass_coeff, assembly=assembly, bracket=bracket)
    water = phase == 0
    force_density = acc / vol[:, None] if assembly == "upstream" else rho0[:, None] * acc
    proj = modal_pressure(pos, geom["theta"], force_density, vol, geom["mode"],
                          geom["radius"], mask=water)
    out = modal_summary(proj["dp_measured"], geom["mode"], geom["radius"], geom["eps"])
    out.update(force_density_definition=("a_i / V_i" if assembly == "upstream"
                                        else "rho_i a_i"),
               delta_integral=float((fields.delta_s[water] * vol[water]).sum()
                                    / (2.0 * math.pi * geom["radius"])),
               contact_fraction_water=float(frac[water][0]),
               rho_air=RHO_AIR, delta=delta, hourglass=bool(hourglass),
               hourglass_coeff=float(hourglass_coeff), assembly=assembly,
               bracket=bracket, n_particles=int(pos.shape[0]),
               n_water=int(water.sum()))
    return out


# --------------------------------------------------------------------------- #
# the calibration channel: Brackbill CSF through the existing Lamb path
# --------------------------------------------------------------------------- #
def csf_lamb(mode: int, eps: float, radius: float, dx: float,
             curvature_scheme: str = "fit_quadratic",
             delta_normalization: str = "legacy") -> dict:
    geom = drop_geometry(mode, eps, radius, dx)
    out = drop_modal_stiffness(geom, SIGMA, curvature_scheme=curvature_scheme,
                               delta_normalization=delta_normalization,
                               periodic_x=False)
    out.update(model="csf_brackbill", curvature_scheme=curvature_scheme,
               delta_normalization=delta_normalization, dx=dx,
               radius_over_dx=radius / dx, eps=eps, mode=mode)
    return out


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def ladder(levels=(16, 24, 32), mode: int = MODE, eps: float = EPS,
           radius: float = RADIUS) -> dict:
    dx_of = lambda ppd: radius / ppd  # noqa: E731
    csf = [dict(case=f"csf_R{ppd}", **csf_lamb(mode, eps, radius, dx_of(ppd)))
           for ppd in levels]
    css_hg = []
    css_nohg = []
    css_cons = []
    for ppd in levels:
        dx = dx_of(ppd)
        g = lobe_geometry(mode, eps, radius, dx)
        css_hg.append(dict(case=f"css_hg_R{ppd}", radius_over_dx=ppd,
                           **css_single_phase(g, hourglass=True)))
        css_nohg.append(dict(case=f"css_nohg_R{ppd}", radius_over_dx=ppd,
                             **css_single_phase(g, hourglass=False)))
        css_cons.append(dict(case=f"css_consistent_R{ppd}", radius_over_dx=ppd,
                             **css_single_phase(g, hourglass=True,
                                                assembly="consistent")))
    return {"csf": csf, "css_hourglass": css_hg, "css_no_hourglass": css_nohg,
            "css_consistent": css_cons}


def mixture_probe(mode: int = MODE, eps: float = EPS, radius: float = RADIUS,
                  radius_over_dx: int = 24) -> dict:
    """Two-phase (1000:1) lobed drop -- the *mixture material* path of the anchor."""
    dx = radius / radius_over_dx
    g = two_phase_lobe_geometry(mode, eps, radius, dx)
    rows = {}
    for label, kw in (("hourglass", dict(hourglass=True)),
                      ("no_hourglass", dict(hourglass=False)),
                      ("consistent", dict(hourglass=True, assembly="consistent")),
                      ("symmetric_bracket", dict(hourglass=True, bracket="symmetric"))):
        rows[label] = css_two_phase(g, **kw)
    return {"geometry": {"radius_over_dx": radius_over_dx, "dx": dx, "eps": eps,
                         "mode": mode, "rho_ratio": RHO0 / RHO_AIR,
                         "rings": RINGS, "n_particles": int(g["pos"].shape[0]),
                         "n_water": int((g["phase"] == 0).sum())},
            "cases": rows}


def amplitude_scan(mode: int = MODE, radius: float = RADIUS,
                   radius_over_dx: int = 24,
                   epsilons=(0.0, 0.01, 0.02, 0.05, 0.10)) -> list:
    """Separate the physical (``eps``-proportional) force from the hourglass part.

    ``eps = 0`` is a perfect circle, where the mode-``n`` modal pressure of a
    *physical* surface-tension force is exactly zero by symmetry.  Whatever the
    port returns there is a spurious lattice term.  The hourglass contribution
    is isolated as ``dp(hourglass on) - dp(hourglass off)``.
    """
    dx = radius / radius_over_dx
    rows = []
    for eps in epsilons:
        g = lobe_geometry(mode, eps, radius, dx)
        hg = css_single_phase(g, hourglass=True)
        nohg = css_single_phase(g, hourglass=False)
        csf = csf_lamb(mode, eps, radius, dx)
        dp_hg = hg["dp_measured_pa"] - nohg["dp_measured_pa"]
        rows.append({"eps": eps,
                     "css_hourglass_dp": hg["dp_measured_pa"],
                     "css_hourglass_ratio": hg["stiffness_ratio"],
                     "css_no_hourglass_dp": nohg["dp_measured_pa"],
                     "css_no_hourglass_ratio": nohg["stiffness_ratio"],
                     "hourglass_dp": dp_hg,
                     "hourglass_over_sigma_R": dp_hg * radius / SIGMA,
                     "csf_dp": csf["dp_measured"],
                     "csf_ratio": csf["stiffness_ratio"]})
    return rows


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def plot(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # (a) frequency ratio vs resolution
    ax = axes[0]
    for key, style, label in (("csf", "k-o", "Brackbill CSF (calibration)"),
                              ("css_hourglass", "r-s", "SPHinXsys CSS, hourglass 4.5"),
                              ("css_no_hourglass", "b-^", "SPHinXsys CSS, hourglass off"),
                              ("css_consistent", "g-d", "CSS, assembly=consistent")):
        rows = payload["ladder"][key]
        ppd = [r["radius_over_dx"] for r in rows]
        ratio = [r["omega_ratio"] for r in rows]
        ax.plot(ppd, ratio, style, label=label, ms=5)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axhspan(0.85, 1.15, color="r", alpha=0.08, label="15% gate")
    ax.axhline(0.0, color="0.5", lw=0.6, ls="--")
    ax.set_xlabel("R / dx")
    ax.set_ylabel(r"$\omega_n / \omega_n^{\rm exact}$")
    ax.set_title(f"(a) Lamb mode n={MODE} frequency ratio")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) modal pressures at the finest level
    ax = axes[1]
    rows = payload["ladder"]["csf"]
    def _dp_exact(row):
        return abs(row.get("dp_analytic_pa", row.get("dp_analytic")))

    dp_exact = _dp_exact(rows[-1])
    names, vals, colors = [], [], []
    for key, color in (("csf", "0.2"), ("css_hourglass", "#d62728"),
                       ("css_no_hourglass", "#1f77b4"),
                       ("css_consistent", "#2ca02c")):
        r = payload["ladder"][key][-1]
        names.append(key.replace("_", "\n"))
        vals.append(r.get("dp_measured_pa", r.get("dp_measured")) / dp_exact)
        colors.append(color)
    ax.bar(names, vals, color=colors)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel(r"$\Delta p_{\rm mode}\,/\,|\Delta p_{\rm exact}|$")
    ax.set_title(f"(b) modal pressure, R/dx = {rows[-1]['radius_over_dx']}")
    ax.grid(alpha=0.3, axis="y")

    # (c) acceleration field of the ported model on the lobed interface
    ax = axes[2]
    g = lobe_geometry(MODE, EPS, RADIUS, RADIUS / 32)
    out = css_single_phase(g, hourglass=True)
    pos, vol = g["pos"], g["vol"]
    h = H_FACTOR * g["dx"]
    rho = np.full(pos.shape[0], RHO0)
    mass = rho * vol.numpy()
    acc, fields = sc.surface_tension_acceleration(
        pos, torch.as_tensor(mass, dtype=pos.dtype), torch.as_tensor(rho, dtype=pos.dtype),
        vol, cell_linked_pairs(pos, 2.0 * h), None, SIGMA, h, mode="single_phase",
        delta="sphinxys", hourglass=True)
    r = np.linalg.norm(pos.numpy(), axis=1)
    band = fields.delta_s.numpy() * g["dx"] > 0.05
    a_r = (acc.numpy() * pos.numpy() / r[:, None]).sum(axis=1)
    sc_ = ax.scatter(pos.numpy()[band, 0], pos.numpy()[band, 1],
                     c=a_r[band] * g["dx"] ** 2, s=14, cmap="coolwarm")
    ax.set_aspect("equal")
    ax.set_title("(c) CSS radial acceleration on the band\n"
                 f"(R/dx=32, n={MODE}, eps={EPS})", fontsize=10)
    fig.colorbar(sc_, ax=ax, shrink=0.85,
                 label=r"$a_r V\ [{\rm m^3/s^2}]$")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _table(rows, title):
    print(f"\n{title}")
    print(f"{'case':>26} {'dp ratio':>10} {'omega ratio':>12} {'err %':>9}")
    for r in rows:
        print(f"{r['case']:>26} {r['stiffness_ratio']:10.4f} {r['omega_ratio']:12.4f} "
              f"{r['omega_error_pct']:9.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="16,24,32")
    ap.add_argument("--mixture-level", type=int, default=24)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    levels = tuple(int(v) for v in args.levels.split(","))
    payload = {
        "anchor": "XVAL-SPH-MIXTURE-MATERIAL-AND-SURFACE-TENSION",
        "subject": "SPHinXsys CSS (Zhang & Hu) as ported in src/tensorsph/sphinxys_css.py",
        "calibration": "Brackbill CSF via surface_tension_dispersion.drop_modal_stiffness",
        "sigma": SIGMA, "rho0": RHO0, "radius": RADIUS, "mode": MODE, "eps": EPS,
        "h_over_dx": H_FACTOR, "hourglass_coeff": HOURGLASS,
        "reference": {"dp": "sigma (n^2-1) eps / R", "omega": "sqrt(n (n^2-1) sigma / (rho R^3))"},
        "ladder": ladder(levels),
        "mixture": mixture_probe(radius_over_dx=args.mixture_level),
        "amplitude_scan": amplitude_scan(),
    }
    for key in ("csf", "css_hourglass", "css_no_hourglass", "css_consistent"):
        _table(payload["ladder"][key], f"ladder / {key}")
    print("\nmixture (two-phase 1000:1)")
    for label, row in payload["mixture"]["cases"].items():
        print(f"{label:>18} dp ratio {row['stiffness_ratio']:9.4f} "
              f"omega ratio {row['omega_ratio']:9.4f} err {row['omega_error_pct']:7.2f}%")
    print("\namplitude scan (hourglass part isolated)")
    print(f"{'eps':>6} {'dp_css_hg':>10} {'dp_css_nohg':>12} {'dp_hg':>10} "
          f"{'dp_hg/(sigma/R)':>15} {'dp_csf':>10}")
    for row in payload["amplitude_scan"]:
        print(f"{row['eps']:6.2f} {row['css_hourglass_dp']:10.4f} "
              f"{row['css_no_hourglass_dp']:12.4f} {row['hourglass_dp']:10.4f} "
              f"{row['hourglass_over_sigma_R']:15.3f} {row['csf_dp']:10.4f}")

    (outdir / "lamb.json").write_text(json.dumps(payload, indent=2))
    plot(payload, outdir / "lamb.png")
    print(f"\nwrote {outdir/'lamb.json'} and {outdir/'lamb.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
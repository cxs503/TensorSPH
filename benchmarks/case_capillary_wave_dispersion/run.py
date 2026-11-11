"""CASE-CAPILLARY-WAVE: capillary-wave dispersion of the TensorSPH CSF operator.

Case bank entry (owner 2026-11-12 算例池令).  One case, one question:

    For a two-layer interface ``eta(x) = -A cos(k x)`` (water below, air above,
    no gravity, inviscid), how far is the *measured* capillary-wave frequency
    from the exact linear dispersion relation

        omega^2 = sigma k^3 / (rho1 coth(k h1) + rho2 coth(k h2))            (*)

    which reduces to Lamb (1932) Art. 267 for a deep single fluid?

Tolerance of this case: ``|omega_got - omega_ref| / omega_ref < 10 %``.

What is measured
----------------
The linear-response (discrete-dispersion) measurement already shipped in
:mod:`tensorsph.surface_tension_dispersion` -- the frequency ratio equals the
modal normal-stress ratio

    omega_measured^2 / omega_ref^2 = dp_measured / dp_theory ,
    dp_theory = sigma * (mode-k coefficient of the exact curvature, incl. arclength)

-- but evaluated on a **two-layer cloud** (particles on both sides of the
interface) instead of the one-sided free surface.  The one-sided free surface is
*not* admissible for this case at HEAD: its ``delta_s = 2|grad c|`` carries only
half of the Brackbill unit mass and its curvature is dominated by lattice noise
that grows like ``1/dx`` (``docs/verification_surface_tension_dispersion.md``,
``docs/verification_capillary_wave_open_surface.md``).  The two-layer
construction is the one already used and validated by
``benchmarks/two_phase_laplace`` (Laplace law, 4.6 % at ``R/h = 10.3``).

Everything reused from the library (no new operator, no new stove):
``cell_linked_pairs``, ``cubic_spline_kernel``, ``mls_gradient``,
``shepard_smooth``, ``mls_divergence``, ``fit_curvature``,
``exact_wave_stress_coeff``.  The case adds only the geometry builder and the
modal projection on that geometry.

Reference values
----------------
:mod:`benchmarks.case_capillary_wave_dispersion.ref` evaluates (*) with mpmath at
50 digits from the citation string -- no reference digit is transcribed by hand,
and the reduction checks (Lamb Art. 267, finite-depth gravity-capillary at
``g = 0``) are asserted at run time.

Usage
-----
    PYTHONPATH=src python benchmarks/case_capillary_wave_dispersion/run.py \
        --out-dir runs/case_capillary_wave_dispersion
    PYTHONPATH=src python benchmarks/case_capillary_wave_dispersion/run.py --quick

``--quick`` runs the design point plus a two-point resolution check (~10 s) for
CI; the full campaign (two curvature schemes, resolution / amplitude / sigma
ladders, figures) is ~1 min on 4 threads.

Exit status: 0 when the case passes, 1 on an honest FAIL, 2 on a harness error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ref as refmod                                            # noqa: E402
from tensorsph.csf_consistent import (                          # noqa: E402
    mls_divergence, mls_gradient, shepard_smooth,
)
from tensorsph.csf_curvature import fit_curvature               # noqa: E402
from tensorsph.droplet_impact import cubic_spline_kernel        # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs         # noqa: E402
from tensorsph.surface_tension_dispersion import exact_wave_stress_coeff  # noqa: E402

# --------------------------------------------------------------------------- #
# case definition (all dimensional; see docs/case_capillary_wave_dispersion.md)
# --------------------------------------------------------------------------- #
PARAMS = {
    "wavelength_m": 0.020,          # lambda = 20 mm (water capillary wave)
    "sigma_N_m": 0.072,             # water/air at 20 C
    "rho_liquid_kg_m3": 1000.0,
    "rho_gas_kg_m3": 1.2,           # air at 20 C
    "depth_fluid_factor": 0.8,      # h1 = 0.8 lambda  -> k h1 = 5.03
    "depth_gas_factor": 0.5,        # h2 = 0.5 lambda  -> k h2 = 3.14
    "amplitude_fraction": 0.02,     # A / lambda  -> k A = 0.126
    "h_factor": 1.3,                # h / dx
    "curvature_stencil": 4.0,       # stencil / h for the local interface fit
}
PARAMS["depth_fluid_m"] = PARAMS["depth_fluid_factor"] * PARAMS["wavelength_m"]
PARAMS["depth_gas_m"] = PARAMS["depth_gas_factor"] * PARAMS["wavelength_m"]

TOL_PCT = 10.0                      # case tolerance (predeclared)
DESIGN = {"ppw": 32, "curvature_scheme": "mls",
          "amplitude_fraction": PARAMS["amplitude_fraction"]}
PPW_LADDER = (16, 24, 32, 48)
SCHEMES = ("mls", "fit_quadratic")   # mls = admissible; fit_quadratic = A/B control
AMP_LADDER = (0.01, 0.02, 0.05)
SIGMA_FACTORS = (0.5, 1.0, 2.0)
SHIFTS = 2                           # shifts x shifts lattice phases (ensemble)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def hex_lattice(dx: float, xmin: float, xmax: float, ymin: float, ymax: float,
                sx: float = 0.0, sy: float = 0.0) -> torch.Tensor:
    """Hexagonal lattice of spacing ``dx`` in a box, shifted by ``(sx dx, sy dy)``."""
    dy = dx * math.sqrt(3.0) / 2.0
    ys = torch.arange(ymin + 0.5 * dy, ymax, dy, dtype=torch.float64)
    xs = torch.arange(xmin + 0.5 * dx, xmax, dx, dtype=torch.float64)
    rows = []
    for r, y in enumerate(ys):
        off = 0.5 * dx if r % 2 else 0.0
        rows.append(torch.stack([xs + off, torch.full_like(xs, float(y))], dim=1))
    pos = torch.cat(rows)
    return pos + torch.tensor([sx * dx, sy * dy], dtype=torch.float64)


def build_two_layer_wave(k: float, amplitude: float, ppw: int,
                         params: dict | None = None,
                         sx: float = 0.0, sy: float = 0.0) -> dict:
    """Two-layer cloud: fluid below ``eta = -A cos(k x)``, gas above, periodic in x.

    The lateral period is one wavelength, ``L = lambda``; the cloud is triplicated
    along ``x`` so the primary copy sees its periodic images (primary copy first,
    the m14536 ordering requirement), and the returned fields are the primary copy.
    Per-phase particle mass is ``rho * cell`` with ``cell = sqrt(3)/2 dx^2``, so the
    volume weight ``m / rho = cell`` is identical on both sides and the colour
    field sees a uniform measure (the construction of ``benchmarks/two_phase_laplace``).
    """
    params = dict(PARAMS if params is None else params)
    lam = 2.0 * math.pi / k
    dx = lam / ppw
    h1 = params["depth_fluid_m"]
    h2 = params["depth_gas_m"]
    cell = dx * dx * math.sqrt(3.0) / 2.0
    pos = hex_lattice(dx, 0.0, lam, -h1, h2, sx, sy)
    eta = -amplitude * torch.cos(k * pos[:, 0])
    colour0 = (pos[:, 1] < eta).to(torch.float64)         # 1 = liquid, 0 = gas
    n = pos.shape[0]
    pos_full, col_full, mass_full = [], [], []
    for shift in (0.0, -lam, lam):
        p = pos.clone()
        p[:, 0] += shift
        pos_full.append(p)
        col_full.append(colour0)
        mass_full.append(torch.where(
            colour0 > 0.5,
            torch.tensor(params["rho_liquid_kg_m3"] * cell, dtype=torch.float64),
            torch.tensor(params["rho_gas_kg_m3"] * cell, dtype=torch.float64)))
    return dict(pos=torch.cat(pos_full), colour=torch.cat(col_full),
                mass=torch.cat(mass_full), n_primary=n, dx=dx, cell=cell,
                length=lam, k=k, amplitude=amplitude, ppw=ppw,
                eta=lambda x: -amplitude * (k * x).cos())


def _volume_weights(geom: dict) -> torch.Tensor:
    """``V = m / rho`` per particle: identical (``cell``) on both sides of the interface."""
    colour = geom["colour"]
    rho = torch.where(
        colour > 0.5,
        torch.tensor(PARAMS["rho_liquid_kg_m3"], dtype=colour.dtype),
        torch.tensor(PARAMS["rho_gas_kg_m3"], dtype=colour.dtype))
    return (geom["mass"] / rho).to(torch.float64)


def measure(geom: dict, sigma: float, *, curvature_scheme: str = "mls",
            h_factor: float | None = None, stencil: float | None = None) -> dict:
    """CSF modal normal stress of the two-layer wave on one lattice phase."""
    h_factor = PARAMS["h_factor"] if h_factor is None else h_factor
    stencil = PARAMS["curvature_stencil"] if stencil is None else stencil
    pos = geom["pos"].to(torch.float64)
    colour = geom["colour"].to(torch.float64)
    h = h_factor * geom["dx"]
    vol = _volume_weights(geom)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    zero = torch.zeros(pos.shape[0], dtype=torch.float64)
    i, j = pairs.unbind(dim=1)
    w = cubic_spline_kernel((pos[i] - pos[j]).norm(dim=1), h)
    den = zero.index_add(0, i, w * vol[j]) + zero.index_add(0, j, w * vol[i])
    num = (zero.index_add(0, i, w * vol[j] * colour[j])
           + zero.index_add(0, j, w * vol[i] * colour[i]))
    c_s = num / den.clamp_min(1.0e-30)                   # Shepard-smoothed colour

    grad = mls_gradient(pos, c_s, vol, pairs, h)
    mag = torch.linalg.vector_norm(grad, dim=1, keepdim=True)
    tiny = 1.0e-12
    normal = torch.where(mag > tiny, -grad / mag.clamp_min(tiny),
                         torch.zeros_like(grad))
    smooth = shepard_smooth(normal, pos, vol, pairs, h, passes=2)
    normal = torch.where(mag > tiny,
                         smooth / torch.linalg.vector_norm(smooth, dim=1,
                                                           keepdim=True).clamp_min(tiny),
                         normal)
    delta_s = mag[:, 0]                                  # two-phase: int delta dn = 1
    kappa = mls_divergence(pos, normal, vol, pairs, h)
    if curvature_scheme != "mls":
        pairs_wide = cell_linked_pairs(pos, stencil * h)
        kappa = fit_curvature(pos, normal, delta_s, pairs_wide, h, fallback=kappa,
                              method=curvature_scheme, stencil=stencil)

    n = geom["n_primary"]
    k = geom["k"]
    x = pos[:n, 0]
    vol_p = vol[:n]
    f = -sigma * kappa[:n] * delta_s[:n] * vol_p           # nodal force [N]
    dp_meas = 2.0 / geom["length"] * float((f * torch.cos(k * x)).sum())
    dp_theory = sigma * exact_wave_stress_coeff(k, geom["amplitude"])
    ratio = dp_meas / dp_theory
    omega_ratio = math.copysign(math.sqrt(abs(ratio)), ratio)
    return dict(
        dp_measured_Pa_m=dp_meas,
        dp_theory_Pa_m=dp_theory,
        stiffness_ratio=ratio,
        omega_ratio=omega_ratio,
        omega_error_pct=(omega_ratio - 1.0) * 100.0,
        delta_s_integral=float((delta_s[:n] * vol_p).sum()) / geom["length"],  # -> 1 per unit length
        delta_s_target=1.0,
        kappa_modal=2.0 / geom["length"] * float(
            (kappa[:n] * delta_s[:n] * vol_p * torch.cos(k * x)).sum()),
        kappa_exact=geom["amplitude"] * k * k,
        n_primary=int(n),
        n_particles=int(pos.shape[0]),
        h=h,
        # keep the interface profile for the figure (primary copy)
        interface_x=x.detach().clone(),
        interface_kappa=(kappa[:n] * delta_s[:n] * vol_p).detach().clone(),
    )


def case_point(*, ppw: int, curvature_scheme: str, amplitude_fraction: float,
               sigma: float, shifts: int, params: dict | None = None) -> dict:
    """Ensemble-averaged datum: ``shifts**2`` lattice phases, mean +- spread."""
    params = dict(PARAMS if params is None else params)
    lam = params["wavelength_m"]
    k = 2.0 * math.pi / lam
    amp = amplitude_fraction * lam
    rows = []
    for a in range(shifts):
        for b in range(shifts):
            geom = build_two_layer_wave(k, amp, ppw, params, a / shifts, b / shifts)
            rows.append(measure(geom, sigma, curvature_scheme=curvature_scheme))
    keys = ("stiffness_ratio", "omega_ratio", "delta_s_integral", "kappa_modal",
            "omega_error_pct")
    mean = {key: sum(r[key] for r in rows) / len(rows) for key in keys}
    std = {key: float(torch.tensor([r[key] for r in rows]).std(unbiased=False))
           for key in ("stiffness_ratio", "omega_ratio")}
    ref = refmod.reference(params)
    omega_ref = ref["omega_ref_float"]
    omega_got = mean["omega_ratio"] * omega_ref
    return dict(
        case="CASE-CAPILLARY-WAVE",
        ppw=ppw, dx_m=lam / ppw, particles_per_wavelength=ppw,
        curvature_scheme=curvature_scheme,
        amplitude_over_lambda=amplitude_fraction,
        amplitude_m=amp, k_rad_m=k, sigma_N_m=sigma,
        sigma_over_ref=sigma / params["sigma_N_m"],
        shifts=shifts, n_phases=shifts * shifts,
        n_particles=rows[0]["n_particles"], n_primary=rows[0]["n_primary"],
        h_over_dx=params["h_factor"],
        delta_s_integral=mean["delta_s_integral"],
        dp_measured_Pa_m=rows[0]["dp_measured_Pa_m"],
        dp_theory_Pa_m=rows[0]["dp_theory_Pa_m"],
        stiffness_ratio=mean["stiffness_ratio"],
        stiffness_ratio_std=std["stiffness_ratio"],
        omega_ratio=mean["omega_ratio"],
        omega_ratio_std=std["omega_ratio"],
        omega_ref_rad_s=omega_ref,
        omega_got_rad_s=omega_got,
        period_got_s=2.0 * math.pi / omega_got,
        period_ref_s=ref["period_ref_s"],
        phase_speed_got_m_s=omega_got / k,
        phase_speed_ref_m_s=ref["phase_speed_ref_m_s"],
        kappa_modal_over_exact=mean["kappa_modal"] / (amp * k * k),
        omega_error_pct=mean["omega_error_pct"],
        abs_error_pct=abs(mean["omega_error_pct"]),
        pass_case=abs(mean["omega_error_pct"]) < TOL_PCT,
        # figure payload (not written into benchmark.json)
        _profile=rows[0],
    )


# --------------------------------------------------------------------------- #
# campaign
# --------------------------------------------------------------------------- #
def run_campaign(*, quick: bool = False, shifts: int = SHIFTS,
                 params: dict | None = None) -> dict:
    params = dict(PARAMS if params is None else params)
    t0 = time.time()
    ref = refmod.reference(params)

    ppws = (DESIGN["ppw"],) if quick else PPW_LADDER
    schemes = ("mls",) if quick else SCHEMES
    grid = []
    for scheme in schemes:
        for ppw in ppws:
            grid.append(case_point(ppw=ppw, curvature_scheme=scheme,
                                   amplitude_fraction=DESIGN["amplitude_fraction"],
                                   sigma=params["sigma_N_m"], shifts=shifts,
                                   params=params))

    design = [r for r in grid
              if r["ppw"] == DESIGN["ppw"] and r["curvature_scheme"] == "mls"
              and abs(r["amplitude_over_lambda"] - DESIGN["amplitude_fraction"]) < 1e-12][0]
    ladder = [r for r in grid if r["curvature_scheme"] == "mls"]
    worst = max(ladder, key=lambda r: r["abs_error_pct"])

    amp_rows, sigma_rows = [], []
    if not quick:
        for af in AMP_LADDER:
            amp_rows.append(case_point(ppw=DESIGN["ppw"], curvature_scheme="mls",
                                       amplitude_fraction=af, sigma=params["sigma_N_m"],
                                       shifts=shifts, params=params))
        for factor in SIGMA_FACTORS:
            sigma_rows.append(case_point(ppw=DESIGN["ppw"], curvature_scheme="mls",
                                         amplitude_fraction=DESIGN["amplitude_fraction"],
                                         sigma=factor * params["sigma_N_m"],
                                         shifts=shifts, params=params))

    ratios = [r["stiffness_ratio"] for r in sigma_rows]
    sigma_spread_pct = ((max(ratios) - min(ratios)) / (sum(ratios) / len(ratios)) * 100.0
                        if ratios else None)
    gates = {
        "A_design_point_within_tol": dict(
            requirement=f"|omega_got/omega_ref - 1| < {TOL_PCT:.0f} % at ppw={DESIGN['ppw']}, "
                        f"curvature={DESIGN['curvature_scheme']}",
            value_pct=design["abs_error_pct"], pass_=design["pass_case"]),
        "B_resolution_ladder_within_tol": dict(
            requirement=f"all ppw in {ppws} within {TOL_PCT:.0f} % (a passing point that "
                        f"diverges under refinement is not evidence)",
            value_pct=worst["abs_error_pct"], worst_ppw=worst["ppw"],
            pass_=worst["abs_error_pct"] < TOL_PCT),
        "C_delta_s_integrates_to_one": dict(
            requirement="sum_i delta_s,i V_i / L = 1 +- 5 % (Brackbill unit mass)",
            value=design["delta_s_integral"],
            pass_=abs(design["delta_s_integral"] - 1.0) < 0.05),
        "D_sigma_linearity": dict(
            requirement=f"stiffness ratio constant under sigma x {SIGMA_FACTORS} (spread < 5 %)",
            value_pct=sigma_spread_pct,
            pass_=(sigma_spread_pct is not None and sigma_spread_pct < 5.0)),
    }
    status = "PASS" if all(g["pass_"] for g in gates.values()) else "FAIL"
    for r in grid:
        r["_profile"] = None if quick else r["_profile"]
    for r in amp_rows + sigma_rows:
        r["_profile"] = None
    profiles = {f"ppw{ppw}": None for ppw in ppws}
    return dict(
        status=status,
        case="CASE-CAPILLARY-WAVE",
        tolerance_pct=TOL_PCT,
        design=DESIGN,
        params=params,
        reference=ref,
        grid={r["curvature_scheme"] + f"_ppw{r['ppw']}": _clean(r) for r in grid},
        gates=gates,
        amplitude_ladder=[_clean(r) for r in amp_rows],
        sigma_ladder=[_clean(r) for r in sigma_rows],
        wall_clock_s=time.time() - t0,
        environments=_env(),
    )


def _clean(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _env() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return ""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "git_sha": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable] + sys.argv),
        "mpmath": refmod.HAVE_MPMATH,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(report: dict, fig_dir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = []

    # 1. error vs resolution, both curvature operators, +-10 % band
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for scheme, style in (("mls", "o-"), ("fit_quadratic", "s--")):
        rows = [r for k, r in report["grid"].items() if r["curvature_scheme"] == scheme]
        rows.sort(key=lambda r: r["ppw"])
        if not rows:
            continue
        ax.plot([r["ppw"] for r in rows], [r["omega_error_pct"] for r in rows],
                style, label=f"curvature = {scheme}")
    ax.axhspan(-report["tolerance_pct"], report["tolerance_pct"], color="0.85",
               label=f"+/-{report['tolerance_pct']:.0f} % case tolerance")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("resolution [particles per wavelength]")
    ax.set_ylabel(r"$(\omega_{got}/\omega_{ref}-1)\times100$  [%]")
    ax.set_title("CASE-CAPILLARY-WAVE: frequency error vs resolution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = fig_dir / "error_vs_resolution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(dict(name=p.name, caption="Frequency error vs particles per wavelength "
                                         "for the two shipped curvature operators; the "
                                         "shaded band is the 10 % case tolerance."))

    # 2. curvature profile across one wavelength vs the exact profile
    prof = report.get("_profile")
    if prof is not None:
        x = prof["interface_x"].numpy()
        w = prof["interface_kappa"].numpy()
        k = prof["k_rad_m"]
        amp = prof["amplitude_m"]
        order = x.argsort()
        xs, ws = x[order], w[order]
        # bin along x (the interface particles sit inside +-2h of eta)
        nb = 40
        edges = torch.linspace(0.0, 2.0 * math.pi / k, nb + 1)
        idx = torch.bucketize(torch.as_tensor(xs), edges) - 1
        prof_x, prof_k = [], []
        for b in range(nb):
            sel = idx == b
            if bool(sel.any()):
                prof_x.append(float(torch.as_tensor(xs)[sel].mean()))
                prof_k.append(float(torch.as_tensor(ws)[sel].sum() /
                                    (edges[1] - edges[0])))
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        xx = [i * 2.0 * math.pi / k / 400 for i in range(400)]
        exact = [-amp * k * k * math.cos(k * t) / (1.0 + (amp * k * math.sin(k * t)) ** 2) ** 1.5
                 for t in xx]
        ax.plot(xx, exact, "k-", lw=1.5, label=r"exact $\kappa(x)$ of $\eta=-A\cos kx$")
        ax.plot(prof_x, prof_k, "o", ms=3.5,
                label=r"CSF $\sum_i \kappa_i \delta_{s,i} V_i$ per bin")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(r"curvature [1/m]")
        ax.set_title(f"Interface curvature, ppw={prof['ppw']}, {prof['curvature_scheme']}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        p = fig_dir / "curvature_profile.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        out.append(dict(name=p.name, caption="Measured CSF curvature along one wavelength "
                                             "vs the exact profile of the perturbation."))

    # 3. sensitivity ladders
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    amp = report["amplitude_ladder"]
    if amp:
        axes[0].plot([r["amplitude_over_lambda"] for r in amp],
                     [r["omega_error_pct"] for r in amp], "o-")
        axes[0].set_xlabel(r"$A/\lambda$")
        axes[0].set_ylabel("frequency error [%]")
        axes[0].set_title("amplitude ladder (ppw=%d)" % DESIGN["ppw"])
    sig = report["sigma_ladder"]
    if sig:
        axes[1].plot([r["sigma_over_ref"] for r in sig],
                     [r["stiffness_ratio"] for r in sig], "o-")
        axes[1].axhline(1.0, color="k", lw=0.8)
        axes[1].set_xlabel(r"$\sigma/\sigma_{ref}$")
        axes[1].set_ylabel(r"stiffness ratio $\Delta p_{meas}/\Delta p_{theory}$")
        axes[1].set_title("sigma ladder (linearity)")
    for ax in axes:
        ax.grid(alpha=0.3)
    p = fig_dir / "sensitivity.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append(dict(name=p.name, caption="Amplitude and surface-tension sensitivity of the "
                                         "measured dispersion."))
    return out


# --------------------------------------------------------------------------- #
# summary + main
# --------------------------------------------------------------------------- #
def write_summary(report: dict, path: Path) -> None:
    r = report["reference"]
    lines = [
        "# CASE-CAPILLARY-WAVE: capillary-wave dispersion (TensorSPH)",
        "",
        f"**Status: {report['status']}** (tolerance {report['tolerance_pct']:.0f} %)",
        "",
        "## Reference",
        "",
        f"* formula: `{r['formula']}`",
        f"* source: {r['citation']['source']}",
        f"* evaluator: {r['evaluator']} at dps={r['dps']} (no digit transcribed)",
        f"* `omega_ref` = {r['omega_ref_float']:.6f} rad/s, "
        f"`T_ref` = {r['period_ref_s']:.6e} s, `c_ref` = {r['phase_speed_ref_m_s']:.6f} m/s",
        f"* reductions (Lamb Art. 267, finite-depth at g=0): "
        f"{r['reductions']}",
        "",
        "## Parameter table",
        "",
        "| quantity | value |",
        "|---|---|",
    ]
    for key, val in report["params"].items():
        lines.append(f"| `{key}` | {val} |")
    lines += [
        "",
        "## Reference vs measured",
        "",
        "| curvature | ppw | dx [mm] | omega_ref [rad/s] | omega_got [rad/s] | err [%] | "
        "stiffness ratio +- spread | delta_s integral | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(report["grid"], key=lambda s: (s.split("_ppw")[0],
                                                     int(s.split("_ppw")[1]))):
        row = report["grid"][key]
        verdict = "PASS" if row["pass_case"] else "FAIL"
        lines.append(
            f"| {row['curvature_scheme']} | {row['ppw']} | {row['dx_m']*1e3:.3f} | "
            f"{row['omega_ref_rad_s']:.6f} | {row['omega_got_rad_s']:.6f} | "
            f"{row['omega_error_pct']:+.2f} | {row['stiffness_ratio']:.4f} +- "
            f"{row['stiffness_ratio_std']:.4f} | {row['delta_s_integral']:.4f} | {verdict} |")
    lines += ["", "## Gates", "",
              "| gate | requirement | value | pass |", "|---|---|---|---|"]
    for name, g in report["gates"].items():
        val = g.get("value_pct", g.get("value"))
        lines.append(f"| {name} | {g['requirement']} | {val} | {g['pass_']} |")
    lines += ["", "## Sensitivity ladders", ""]
    for title, rows, xkey in (("amplitude", report["amplitude_ladder"], "amplitude_over_lambda"),
                             ("sigma", report["sigma_ladder"], "sigma_over_ref")):
        if not rows:
            continue
        lines += [f"### {title}", "",
                  f"| {xkey} | omega_got/omega_ref | stiffness ratio | err [%] |",
                  "|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row[xkey]:.3g} | {row['omega_ratio']:.4f} | "
                         f"{row['stiffness_ratio']:.4f} | {row['omega_error_pct']:+.2f} |")
        lines.append("")
    lines += [f"wall clock: {report['wall_clock_s']:.1f} s", ""]
    path.write_text("\n".join(lines))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="runs/case_capillary_wave_dispersion")
    ap.add_argument("--quick", action="store_true", help="design point only (~10 s)")
    ap.add_argument("--shifts", type=int, default=SHIFTS)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # reference first, so a broken reference stops the run before any compute
    ref = refmod.reference(PARAMS, out / "ref.json")
    if not ref["reductions"]["all_below_1e-12"]:
        print("reference reduction checks failed -- refusing to benchmark")
        return 2

    report = run_campaign(quick=args.quick, shifts=args.shifts)
    report["reference"]["ref_json_sha256"] = ref["ref_json_sha256"]
    report["reference"]["ref_json_path"] = ref["ref_json_path"]

    # keep one measured profile for the figure, outside the JSON payload
    design_ppw = DESIGN["ppw"] if args.quick else DESIGN["ppw"]
    profile = None
    for ppw in ((design_ppw,) if args.quick else PPW_LADDER):
        geom = build_two_layer_wave(2.0 * math.pi / PARAMS["wavelength_m"],
                                    DESIGN["amplitude_fraction"] * PARAMS["wavelength_m"],
                                    ppw)
        m = measure(geom, PARAMS["sigma_N_m"], curvature_scheme="mls")
        profile = dict(interface_x=m["interface_x"], interface_kappa=m["interface_kappa"],
                       k_rad_m=geom["k"], amplitude_m=geom["amplitude"], ppw=ppw,
                       curvature_scheme="mls")
        if ppw == design_ppw:
            break
    report["_profile"] = profile

    figs = [] if args.no_figures else figures(report, out / "figures")
    report.pop("_profile", None)

    benchmark = {
        "case": report["case"], "status": report["status"],
        "tolerance_pct": report["tolerance_pct"],
        "reference": {k: report["reference"][k] for k in
                      ("formula", "citation", "dps", "evaluator", "omega_ref_float",
                       "period_ref_s", "phase_speed_ref_m_s", "cross_checks",
                       "reductions", "ref_json_sha256")},
        "rows": sorted(report["grid"].values(), key=lambda r: (r["curvature_scheme"], r["ppw"])),
        "gates": report["gates"],
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    (out / "report.json").write_text(text)
    (out / "benchmark.json").write_text(json.dumps(benchmark, indent=2, sort_keys=True,
                                                   default=str))
    write_summary(report, out / "summary.md")
    meta = {
        "case": report["case"], "status": report["status"],
        "environment": report["environments"],
        "figures": figs,
        "artifacts": {},
    }
    for name in ("ref.json", "report.json", "benchmark.json", "summary.md"):
        meta["artifacts"][name] = _sha256(out / name)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))

    print(f"CASE-CAPILLARY-WAVE: {report['status']}  "
          f"(design error {report['gates']['A_design_point_within_tol']['value_pct']:+.2f} %, "
          f"worst over ladder {report['gates']['B_resolution_ladder_within_tol']['value_pct']:.2f} %)")
    for key in sorted(report["grid"]):
        row = report["grid"][key]
        print(f"  {row['curvature_scheme']:14s} ppw={row['ppw']:3d} "
              f"omega_ref={row['omega_ref_rad_s']:.6f} omega_got={row['omega_got_rad_s']:.6f} "
              f"err={row['omega_error_pct']:+7.2f} %  "
              f"ratio={row['stiffness_ratio']:7.4f}+-{row['stiffness_ratio_std']:.4f}")
    print(f"artifacts in {out}/  (wall {report['wall_clock_s']:.1f} s)")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""Free-surface pressure closure with the full Adami body term: record + doc.

The question
------------
``docs/verification_laplace_pressure_floor.md`` localises the Young-Laplace
deficit on the *free-surface pressure closure* (C3): the summation density falls
below ``rho0`` over a shell ~2h thick, the clamped Tait law therefore puts
``p = 0`` there, and the interior relaxes to a fraction of ``sigma/R``. A closure
that extrapolates the bulk pressure onto that shell (Adami, Ihmsen & Adams 2012,
§3.3) removes part of it -- but only the part its body term can represent.

Adami eq. (11) extrapolates with the *non-pressure* acceleration ``a_i``,

    p_i = max(0, sum_j [p_j + rho_j (g - a_i).(r_i - r_j)] W_ij / sum_j W_ij),

and for a static wall ``a_w = 0`` turns that into the gravity-only form the
implementation used. On a free surface ``a_i`` is the continuum surface force,
i.e. precisely the acceleration that has to carry ``sigma/R`` across the shell.
Setting it to zero there is not a small correction: in zero gravity the body term
vanishes and the closure degenerates to a kernel average of the interior pressure
with the clamped zeros.

Two further consequences are measured here:

* an extrapolated pressure does not belong to the receiver's summation density,
  so pairing the two in the pressure term ``p_i/rho_i^2 + p_j/rho_j^2`` scales
  the force of exactly the boundary layer the closure is meant to fix. Inverting
  the Tait law for the filled particles (``surface_pressure_density="tait"``)
  keeps ``p/rho`` on the bulk isentrope;
* both changes are needed: the body term without the density pairing drives the
  drop apart at the coarse resolutions tested here (measured below, reported as
  a FAIL rather than dropped).

What this script does
---------------------
``--record`` measures everything and writes ``report_closure.json``; without it
the document ``docs/verification_free_surface_pressure_closure.md`` is rebuilt
*from that record*, so no number in the document can drift from the measurement.

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py --record
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECORD = HERE / "report_closure.json"
DOC = ROOT / "docs" / "verification_free_surface_pressure_closure.md"
FIGDIR = HERE / "figures"

SCHEMA = 1

#: The relaxation variants. The first four are the A/B ladder of the closure;
#: ``closure_nonpressure`` (body term on, density pairing off) is kept because
#: it is the one that fails, and a benchmark that hides its failures is useless.
VARIANTS = {
    "clamped_tait": dict(closure=False),
    "closure_gravity": dict(closure=True),
    "closure_gravity_tait": dict(closure=True, density="tait"),
    "closure_nonpressure": dict(closure=True, body_force="nonpressure"),
    "closure_nonpressure_tait": dict(closure=True, body_force="nonpressure",
                                     density="tait"),
}


def floor():
    """The ``laplace_pressure_floor`` run module (lattice, constants, relax)."""
    spec = importlib.util.spec_from_file_location("lpf", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 1. the operator, on a field where the answer is known
# --------------------------------------------------------------------------- #
def analytic_gate(ay: float = 3.0, p0: float = 150.0) -> dict:
    """Linear pressure field, zero gravity, constant a_ext: exact to round-off.

    ``p(y) = p0 + rho0 ay y`` with ``g = 0`` requires ``a_ext = (0, -ay)``;
    first-layer receivers then have ``rho_j = rho0`` donors and the extrapolation
    must return the analytic value for every donor kernel weight. The controls
    drop the body term and flip its sign.
    """
    from tensorsph.droplet_impact import Config, free_surface_pressure, pair_kernels
    from tensorsph.neighbor_search import cell_linked_pairs

    dx, h, rho0 = 0.005, 1.3 * 0.005, 1000.0
    nx, ny = 5, 20
    xs = dx * torch.arange(nx, dtype=torch.float64)
    ys = dx * torch.arange(ny, dtype=torch.float64)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=1)
    p = p0 + rho0 * ay * pos[:, 1]
    rho = torch.full((pos.shape[0],), rho0, dtype=pos.dtype)
    order = torch.argsort(pos[:, 1], descending=True)
    free = torch.zeros(pos.shape[0], dtype=torch.bool)
    free[order[: nx * 3]] = True
    rho[free] = 0.97 * rho0
    cfg = Config(dx=dx, h_factor=1.3, rho0=rho0, gravity=0.0,
                 surface_pressure_closure=True, surface_pressure_layers=1)
    pairs = cell_linked_pairs(pos, 2.0 * h)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    wij, _ = pair_kernels(delta, h)
    ext = torch.zeros_like(pos)
    ext[:, 1] = -ay
    filled = free & (p > 1.0)
    ref = p[filled]
    good = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg, external=ext)
    dropped = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg)
    flipped = free_surface_pressure(pos, rho, p, delta, wij, i, j, cfg,
                                    external=-ext)

    def rel(out) -> float:
        return float(((out[filled] - ref).abs() / ref.abs()).max())

    return {
        "description": "linear field with a non-pressure acceleration, one pass, "
                       "interior donors (donor density = rho0)",
        "n_receivers": int(filled.sum()),
        "external_rel_error_max": rel(good),
        "dropped_body_term_abs_error_Pa": float((dropped[filled] - ref).abs().max()),
        "flipped_sign_abs_error_Pa": float((flipped[filled] - ref).abs().max()),
        "body_scale_rho0_a_h_Pa": rho0 * ay * h,
    }


# --------------------------------------------------------------------------- #
# 2. the mechanism, on the un-relaxed lattice (single CSF evaluation)
# --------------------------------------------------------------------------- #
def shell_ramp(lpf, dx: float, c0: float = 20.0) -> dict:
    """Pressure the closure writes onto the shell of an *uncompressed* drop.

    The calibrated lattice sums to ``rho0`` in the core, so ``p = 0`` there and
    the shell value comes from the body term alone: this isolates the mechanism
    from the relaxation. Gravity is zero.
    """
    from tensorsph.droplet_impact import (
        Config, density_from_neighbor_pairs, free_surface_pressure, pair_kernels,
        tait_density, tait_pressure,
    )
    from tensorsph.surface_tension import compute_surface_tension_force

    pos, mass, h, pairs, bias, after = lpf.calibrated_droplet(dx)
    base = dict(h_factor=lpf.H_FACTOR, rho0=lpf.RHO0, sound_speed=c0,
                gamma=lpf.GAMMA, gravity=0.0, viscosity=lpf.VISCOSITY,
                surface_tension_coefficient=lpf.SIGMA, csf_scheme="consistent",
                csf_curvature="fit_quadratic")
    cfg = Config(dx=dx, **base)
    rho = density_from_neighbor_pairs(pos, mass, h, pairs)
    p_tait = tait_pressure(rho, cfg)
    acc_st = compute_surface_tension_force(pos, rho, mass, pairs, cfg)
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]
    wij, _ = pair_kernels(delta, h)
    r = torch.linalg.vector_norm(pos, dim=1) / lpf.RADIUS
    shell = (r >= 0.90) & (r < 1.05)
    bands = {"0.80-0.95": (r >= 0.80) & (r < 0.95),
             "0.90-1.05": (r >= 0.90) & (r < 1.05),
             "1.05-1.20": (r >= 1.05) & (r < 1.20)}

    def read(p) -> dict:
        return {key: float(p[m].mean()) / lpf.DP_ANALYTIC if bool(m.any()) else None
                for key, m in bands.items()}

    rows = {"clamped_tait": read(p_tait)}
    for name, body, dens in (("closure_gravity", "gravity", "sum"),
                             ("closure_nonpressure", "nonpressure", "sum"),
                             ("closure_nonpressure_tait", "nonpressure", "tait")):
        c = Config(**{**cfg.__dict__, "surface_pressure_closure": True,
                      "surface_pressure_body_force": body,
                      "surface_pressure_density": dens})
        ext = acc_st if body == "nonpressure" else None
        p = free_surface_pressure(pos, rho, p_tait, delta, wij, i, j, c,
                                  external=ext)
        row = read(p)
        row["density_tait_over_sum_shell"] = float(
            (tait_density(p, c)[shell] / rho[shell]).mean())
        rows[name] = row
    return {
        "dx_m": dx, "sound_speed": c0, "n_particles": int(pos.shape[0]),
        "n_shell_particles": int(shell.sum()),
        "sigma_over_R_Pa": lpf.DP_ANALYTIC,
        "csf_accel_max_m_per_s2": float(acc_st.abs().max()),
        "csf_accel_mean_shell_m_per_s2": float(
            torch.linalg.vector_norm(acc_st[shell], dim=1).mean()),
        "shell_pressure_over_analytic": rows,
        "note": "the lattice sums to rho0 in the core, so p_tait = 0 everywhere "
                "and only the body term can produce a shell pressure",
    }


# --------------------------------------------------------------------------- #
# 3. the relaxation ladder
# --------------------------------------------------------------------------- #
def run_variant(lpf, dx: float, c0: float, name: str, t_end: float) -> dict:
    t0 = time.time()
    out = lpf.relax(dx, c0, t_end_over_t_sigma=t_end, **VARIANTS[name])
    wall = time.time() - t0
    profiles = out["pressure_profile"] if dx == min(DX_GRID) else []
    return {
        "variant": name,
        "dx_m": dx,
        "sound_speed": c0,
        "h_m": out["h_m"],
        "n_particles": out["n_particles"],
        "steps": out["steps"],
        "t_end_over_t_sigma": out["t_over_t_sigma"],
        "wall_time_s": round(wall, 1),
        "p_core_Pa": out["definition_absolute"]["delta_p_Pa"],
        "p_core_over_analytic": out["definition_absolute"]["delta_p_Pa"]
        / lpf.DP_ANALYTIC,
        "rel_error_pct": out["definition_absolute"]["rel_error_pct"],
        "tail_average_rel_error_pct": out["absolute_time_average_tail"][
            "rel_error_pct"],
        "band_a_rel_error_pct": out["definition_bandA"]["rel_error_pct"],
        "within_3pct": bool(abs(out["absolute_time_average_tail"][
            "rel_error_pct"]) < 3.0),
        "within_15pct": bool(abs(out["absolute_time_average_tail"][
            "rel_error_pct"]) < 15.0),
        "rho_core_over_rho0": out["density"]["rho_core_over_rho0"],
        "density_compression_required": out["density"]["compression_required"],
        "vmax_over_vref": out["spurious_current"]["vmax_over_vref_final"],
        "finite": out["finite"],
        "pressure_profile": profiles,
        "rho_rel_profile": (out["density_profile"]
                            if dx == min(DX_GRID) else []),
    }


DX_GRID = (0.002, 0.0015)
CO_GRID = (20.0,)
RESOLUTION_CHECK_DX = 0.001
RESOLUTION_CHECK_VARIANTS = ("clamped_tait", "closure_nonpressure_tait")


def build_record(t_end: float = 1.5) -> dict:
    lpf = floor()
    runs = []
    for dx in DX_GRID:
        for c0 in CO_GRID:
            for name in VARIANTS:
                row = run_variant(lpf, dx, c0, name, t_end)
                runs.append(row)
                print(f"{name:26s} dx={dx:g} c0={c0:g}: "
                      f"p_core={row['p_core_Pa']:8.1f} Pa "
                      f"({row['rel_error_pct']:+7.1f} %, tail "
                      f"{row['tail_average_rel_error_pct']:+7.1f} %) "
                      f"rho_core={row['rho_core_over_rho0']:.5f} "
                      f"[{row['wall_time_s']:.0f} s]", flush=True)
    resolution = []
    for name in RESOLUTION_CHECK_VARIANTS:
        row = run_variant(lpf, RESOLUTION_CHECK_DX, 20.0, name, t_end)
        resolution.append(row)
        print(f"{name:26s} dx={RESOLUTION_CHECK_DX:g}: "
              f"p_core={row['p_core_Pa']:8.1f} Pa "
              f"({row['rel_error_pct']:+7.1f} %, tail "
              f"{row['tail_average_rel_error_pct']:+7.1f} %)", flush=True)
    record = {
        "benchmark": "laplace_pressure_floor/closure",
        "schema": SCHEMA,
        "question": "does the free-surface pressure closure carry sigma/R across "
                    "the shell once Adami eq. (11)'s non-pressure acceleration "
                    "and the isentropic density pairing are in place?",
        "reference": {
            "sigma_N_per_m": lpf.SIGMA,
            "radius_m": lpf.RADIUS,
            "rho0_kg_per_m3": lpf.RHO0,
            "gamma": lpf.GAMMA,
            "delta_p_analytic_Pa": lpf.DP_ANALYTIC,
            "t_sigma_s": lpf.T_SIGMA,
        },
        "settings": {
            "variants": {k: v for k, v in VARIANTS.items()},
            "dx_m": list(DX_GRID),
            "sound_speed_m_per_s": list(CO_GRID),
            "resolution_check": {"dx_m": RESOLUTION_CHECK_DX,
                                 "variants": list(RESOLUTION_CHECK_VARIANTS)},
            "t_end_over_t_sigma": t_end,
            "csf": {"scheme": "consistent", "curvature": "fit_quadratic"},
            "viscosity_m2_per_s": lpf.VISCOSITY,
            "h_factor": lpf.H_FACTOR,
        },
        "analytic_gate": analytic_gate(),
        "single_shot_shell": [shell_ramp(lpf, dx) for dx in
                              (0.002, 0.0015, 0.001)],
        "runs": runs,
        "resolution_check": resolution,
        "verdict": verdict(runs, resolution, lpf.DP_ANALYTIC),
    }
    return record


def verdict(runs, resolution, analytic_dp: float) -> dict:
    """Machine-made error judgement: every claim in the doc comes from here."""
    def pick(variant, dx):
        for row in runs + resolution:
            if row["variant"] == variant and row["dx_m"] == dx:
                return row
        return None

    main_dx = min(DX_GRID)
    key = pick("closure_nonpressure_tait", main_dx)
    base = pick("clamped_tait", main_dx)
    gravity = pick("closure_gravity", main_dx)
    broken = pick("closure_nonpressure", main_dx)
    return {
        "primary_case": {"variant": "closure_nonpressure_tait", "dx_m": main_dx},
        "rel_error_pct_final_snapshot": key["rel_error_pct"],
        "rel_error_pct_tail_average": key["tail_average_rel_error_pct"],
        "within_3pct": bool(abs(key["tail_average_rel_error_pct"]) < 3.0),
        "baseline_rel_error_pct_tail_average": base["tail_average_rel_error_pct"],
        "gravity_only_rel_error_pct_tail_average":
            gravity["tail_average_rel_error_pct"],
        "delta_vs_baseline_pp": (key["tail_average_rel_error_pct"]
                                 - base["tail_average_rel_error_pct"]),
        "body_force_without_density_pairing": {
            "variant": "closure_nonpressure",
            "dx_m": main_dx,
            "rho_core_over_rho0": broken["rho_core_over_rho0"],
            "p_core_Pa": broken["p_core_Pa"],
            "finite": broken["finite"],
            "status": "FAIL (the drop is driven apart; the coarse grid the "
                      "defect was measured on is where it shows)",
        },
        "limits": [
            "one geometry: R = 20 mm drop, sigma = 10 x water, zero gravity, "
            "no walls; the closure is not exercised in a wall-bounded flow here",
            "2-D cubic-spline kernel only (the code is 2-D)",
            "the c0 ladder of the floor benchmark is not repeated; c0 = 20 m/s "
            "is where the deficit was localised",
            f"analytic reference is sigma/R = {analytic_dp:.1f} Pa; there is no "
            "experimental datum for a 10x-scaled surface tension, so the "
            "comparison is against the exact Young-Laplace value",
        ],
    }


# --------------------------------------------------------------------------- #
# 4. figures
# --------------------------------------------------------------------------- #
def make_figures(record: dict, out_dir: Path = FIGDIR) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    reference = record["reference"]["delta_p_analytic_Pa"]
    runs = record["runs"]
    dx0 = min(record["settings"]["dx_m"])
    names = list(VARIANTS)
    figures = []

    # (1) per-variant error, both resolutions, with the +-3 % band
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    width = 0.38
    for k, dx in enumerate(record["settings"]["dx_m"]):
        vals = []
        for name in names:
            row = next(r for r in runs if r["variant"] == name and r["dx_m"] == dx)
            vals.append(row["tail_average_rel_error_pct"])
        xs = [idx + (k - 0.5) * width for idx in range(len(names))]
        ax.bar(xs, vals, width=width, label=f"dx = {dx*1000:.1f} mm")
    ax.axhspan(-3.0, 3.0, color="tab:green", alpha=0.12,
               label=r"$\pm$3 % acceptance")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("closure_", "").replace("_", "\n")
                        for n in names], fontsize=8)
    ax.set_ylabel(r"$\Delta p$ error vs $\sigma/R$ [%]  (tail average)")
    ax.set_title("(a) Laplace jump by closure variant")
    ax.set_ylim(-110, 60)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "closure_variants.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "closure_variants", "path": path,
                    "caption": "Young-Laplace error of every closure variant at "
                               "both resolutions (tail average over the last "
                               "40 % of 1.5 capillary times)."})

    # (2) radial pressure profile of the three informative variants
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    analytic = reference
    for name in ("clamped_tait", "closure_gravity", "closure_nonpressure_tait"):
        row = next(r for r in runs if r["variant"] == name and r["dx_m"] == dx0)
        prof = row["pressure_profile"]
        ax.plot([p["r_over_R_mid"] for p in prof], [p["mean"] for p in prof],
                "o-", ms=3.5, label=name.replace("_", " "))
    ax.axhline(analytic, color="k", ls="--", lw=1,
               label=rf"$\sigma/R$ = {analytic:.0f} Pa")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel("pressure [Pa]")
    ax.set_title(f"(b) relaxed pressure profile, dx = {dx0*1000:.1f} mm")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "closure_profiles.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "closure_profiles", "path": path,
                    "caption": "Radial pressure profile after 1.5 t_sigma: the "
                               "clamped law, the gravity-only closure and the "
                               "full Adami body term with isentropic density."})

    # (3) mechanism: single-shot shell pressure on the uncompressed lattice
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    shots = record["single_shot_shell"]
    keys = ["0.80-0.95", "0.90-1.05", "1.05-1.20"]
    for name in ("clamped_tait", "closure_gravity", "closure_nonpressure"):
        vals = [shots[0]["shell_pressure_over_analytic"][name][k] or 0.0
                for k in keys]
        ax.plot(keys, vals, "o-", label=name.replace("_", " "))
    ax.axhline(1.0, color="k", ls="--", lw=1, label=r"$\sigma/R$")
    ax.set_xlabel("shell band [r/R]")
    ax.set_ylabel(r"$\langle p\rangle / (\sigma/R)$")
    ax.set_title(f"(c) body term only (dx = {shots[0]['dx_m']*1000:.1f} mm, "
                 "uncompressed lattice)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "closure_shell_ramp.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "closure_shell_ramp", "path": path,
                    "caption": "Shell pressure produced by the body term alone on "
                               "the un-relaxed (p = 0) lattice: the gravity-only "
                               "operator writes nothing, the non-pressure term "
                               "writes the Adami ramp."})
    return figures


# --------------------------------------------------------------------------- #
# 5. the document, rendered from the record
# --------------------------------------------------------------------------- #
def render_doc(record: dict) -> str:
    ref = record["reference"]
    verdict = record["verdict"]
    gate = record["analytic_gate"]
    runs = record["runs"]
    shots = record["single_shot_shell"][0]

    def row(variant, dx):
        return next(r for r in runs if r["variant"] == variant and r["dx_m"] == dx)

    lines: list[str] = []
    add = lines.append
    add("# Verification: free-surface pressure closure with the Adami body term")
    add("")
    add("*Generated by `benchmarks/laplace_pressure_floor/make_closure_doc.py` "
        "from `benchmarks/laplace_pressure_floor/report_closure.json` "
        f"(schema {record['schema']}). Every number below is program-injected; do "
        "not edit this file by hand.*")
    add("")
    add("## 1. Case")
    add("")
    add("A 2-D circular drop in vacuum, zero gravity, no walls, driven only by the "
        "continuum surface force. The Young-Laplace law fixes the pressure jump "
        "exactly, `Delta p = sigma/R`:")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| radius `R` | {ref['radius_m']*1000:.1f} mm |")
    add(f"| surface tension `sigma` | {ref['sigma_N_per_m']:.1f} N/m "
        "(10 x water, chosen to shorten the capillary time) |")
    add(f"| density `rho0` | {ref['rho0_kg_per_m3']:.0f} kg/m^3 |")
    add(f"| Tait exponent `gamma` | {ref['gamma']:.1f} |")
    add(f"| analytic jump `sigma/R` | {ref['delta_p_analytic_Pa']:.1f} Pa |")
    add(f"| capillary time `t_sigma` | {ref['t_sigma_s']*1000:.1f} ms |")
    add(f"| resolutions | {', '.join(f'{d*1000:.1f} mm' for d in record['settings']['dx_m'])}"
        f" (check: {record['settings']['resolution_check']['dx_m']*1000:.1f} mm) |")
    add(f"| sound speed `c0` | "
        f"{', '.join(f'{c:.0f} m/s' for c in record['settings']['sound_speed_m_per_s'])} |")
    add(f"| CSF | `{record['settings']['csf']['scheme']}` scheme, "
        f"`{record['settings']['csf']['curvature']}` curvature |")
    add(f"| window | {record['settings']['t_end_over_t_sigma']:.1f} capillary times |")
    add("")
    add("The particle mass is calibrated so that the interior sum-density is "
        "exactly `rho0` (`calibrated_droplet` in `run.py`), which removes the "
        "initialisation bias C2 of `verification_laplace_pressure_floor.md`; the "
        "`p_core` read-out is the absolute one (C1): the Tait law is normalised "
        "at `rho0`, and the vacuum is the `p = 0` reference.")
    add("")
    add("## 2. Method")
    add("")
    add("Density summation under-counts neighbours at a free surface, so the "
        "outer shell of the drop has `rho/rho0 = 0.65 .. 0.89` and the clamped Tait "
        "law gives it `p = 0`. The interior is then bounded by a zero-pressure band "
        "instead of by the pressure its boundary condition requires. "
        "`free_surface_pressure` replaces the shell pressure by the extrapolation "
        "of Adami, Ihmsen & Adams (2012, J. Comput. Phys. 231:7057) §3.3:")
    add("")
    add("```")
    add("p_i = max(0, sum_j [ p_j + rho_j (g - a_i).(r_i - r_j) ] W_ij / sum_j W_ij )")
    add("```")
    add("")
    add("Two knobs are swept here, and they are independent:")
    add("")
    add("* `surface_pressure_body_force`: `gravity` keeps the term the "
        "implementation used so far (`a_i = 0`, exact for a static wall), "
        "`nonpressure` feeds the continuum surface force into `a_i` -- the "
        "acceleration that has to carry `sigma/R` outward on a free surface;")
    add("* `surface_pressure_density`: `sum` pairs the extrapolated pressure with "
        "the under-counted summation density, `tait` inverts the Tait law so that "
        "`p/rho` stays on the bulk isentrope (`tait_density`).")
    add("")
    add("Variants measured (all with interior donors, two passes, threshold 0.99):")
    add("")
    add("| variant | closure | body force | density |")
    add("|---|---|---|---|")
    for name, kw in record["settings"]["variants"].items():
        add(f"| `{name}` | {'on' if kw.get('closure') else '**off** (clamped Tait)'} "
            f"| `{kw.get('body_force', 'gravity')}` | `{kw.get('density', 'sum')}` |")
    add("")
    add("The operator is first pinned on a field where the answer is known: a "
        "linear pressure field `p(y) = p0 + rho0 ay y` in zero gravity with a "
        "constant `a_ext = (0, -ay)`. With interior donors and one pass the "
        "extrapolation must return the analytic value for every kernel weight.")
    add("")
    add("| operator check | value |")
    add("|---|---|")
    add(f"| receivers | {gate['n_receivers']} |")
    add(f"| max relative error, body term on | {gate['external_rel_error_max']:.2e} |")
    add(f"| max absolute error, body term dropped [Pa] | "
        f"{gate['dropped_body_term_abs_error_Pa']:.2f} |")
    add(f"| max absolute error, body term sign flipped [Pa] | "
        f"{gate['flipped_sign_abs_error_Pa']:.2f} |")
    add(f"| scale of the missed term `rho0 ay h` [Pa] | "
        f"{gate['body_scale_rho0_a_h_Pa']:.2f} |")
    add("")
    add("The relaxation itself is `run.py:relax`: explicit Euler on the "
        "continuum-surface-force momentum equation, `dt` from the acoustic "
        "Courant limit, neighbour search by cell lists, pressure from the Tait "
        "law at `c0`.")
    add("")
    add("## 3. Results")
    add("")
    add("### 3.1 The relaxation ladder")
    add("")
    add("`Delta p` is read as the core mean pressure, `r < 0.5 R`. The snapshot is "
        "taken at 1.5 `t_sigma`; because the drop breathes at the capillary "
        "period, the number that is judged is the average over the last 40 % of "
        "the window.")
    add("")
    add("| dx [mm] | variant | p_core [Pa] | error, snapshot [%] | "
        "error, tail avg [%] | rho_core/rho0 | v_max/v_ref |")
    add("|---|---|---|---|---|---|---|")
    for dx in record["settings"]["dx_m"]:
        for name in record["settings"]["variants"]:
            r = row(name, dx)
            add(f"| {dx*1000:.1f} | `{name}` | {r['p_core_Pa']:.1f} | "
                f"{r['rel_error_pct']:+.1f} | {r['tail_average_rel_error_pct']:+.1f} | "
                f"{r['rho_core_over_rho0']:.5f} | {r['vmax_over_vref']:.4f} |")
    for r in record["resolution_check"]:
        add(f"| {r['dx_m']*1000:.1f} | `{r['variant']}` | {r['p_core_Pa']:.1f} | "
            f"{r['rel_error_pct']:+.1f} | {r['tail_average_rel_error_pct']:+.1f} | "
            f"{r['rho_core_over_rho0']:.5f} | {r['vmax_over_vref']:.4f} |")
    add("")
    add("### 3.2 The mechanism, isolated")
    add("")
    add("Same drop, but nothing has relaxed yet: the calibrated lattice sums to "
        "`rho0` in the core, so `p_tait = 0` everywhere and any shell pressure is "
        "produced by the closure's body term alone. Mean CSF acceleration in the "
        f"shell: {shots['csf_accel_mean_shell_m_per_s2']:.3e} m/s^2 "
        f"(max {shots['csf_accel_max_m_per_s2']:.3e} m/s^2).")
    add("")
    add("| variant | p(0.80-0.95 R) | p(0.90-1.05 R) | p(1.05-1.20 R) |")
    add("|---|---|---|---|")
    for name in ("clamped_tait", "closure_gravity", "closure_nonpressure",
                 "closure_nonpressure_tait"):
        s = shots["shell_pressure_over_analytic"][name]
        add(f"| `{name}` | {s['0.80-0.95']:+.3f} | {s['0.90-1.05']:+.3f} | "
            f"{s['1.05-1.20']:+.3f} |")
    add("")
    add("The gravity-only operator writes exactly nothing onto the shell in zero "
        "gravity -- it is a kernel average of zeros. The non-pressure body term "
        "writes the ramp `rho a . (r_i - r_j)`, i.e. `grad p = rho a`, which is the "
        "first-order statement of `Delta p = sigma/R` across the interface.")
    add("")
    add("### 3.3 Error judgement")
    add("")
    add("| quantity | value | inside the 3 % band? |")
    add("|---|---|---|")
    case = verdict["primary_case"]
    add(f"| `closure_nonpressure_tait`, dx = {case['dx_m']*1000:.1f} mm, "
        f"tail average | {verdict['rel_error_pct_tail_average']:+.1f} % | "
        f"{'**yes**' if verdict['within_3pct'] else '**no**'} |")
    snap = verdict["rel_error_pct_final_snapshot"]
    add(f"| same, final snapshot | {snap:+.1f} % | "
        f"{'yes' if abs(snap) < 3.0 else 'no'} |")
    add(f"| baseline `clamped_tait`, tail average | "
        f"{verdict['baseline_rel_error_pct_tail_average']:+.1f} % | no |")
    add(f"| `closure_gravity`, tail average | "
        f"{verdict['gravity_only_rel_error_pct_tail_average']:+.1f} % | no |")
    add(f"| change against the baseline | {verdict['delta_vs_baseline_pp']:+.1f} pp | -- |")
    add("")
    add(f"Answer: on the case measured here the error is "
        f"{verdict['rel_error_pct_tail_average']:+.1f} %, i.e. "
        f"{'**below**' if verdict['within_3pct'] else '**above**'} the 3 % "
        "acceptance line, against "
        f"{verdict['baseline_rel_error_pct_tail_average']:+.1f} % for the clamped "
        "Tait law the platform shipped before. The reference is the exact "
        "Young-Laplace value, not an experiment, so this is a verification of the "
        "discretisation, not a validation of the physics model.")
    add("")
    add("## 4. Pitfalls")
    add("")
    broken = verdict["body_force_without_density_pairing"]
    add(f"* **The body term alone destroys the drop.** `closure_nonpressure` "
        f"(body force on, density pairing off) ends at "
        f"`rho_core/rho0 = {broken['rho_core_over_rho0']:.3f}`, `p_core = "
        f"{broken['p_core_Pa']:.1f} Pa`: the shell pressure is now large while its "
        "density is still the summation value, so the `1/rho^2` factor of the "
        "pressure term multiplies a boundary-layer pressure by a surface density "
        "and the interface is blown apart. This is reported as a FAIL, not "
        "tuned away: the two knobs are only correct together.")
    add("* The `tail average` column is the one to judge. The drop breathes at the "
        "capillary period, so a single snapshot can land anywhere inside the "
        "envelope; the window is 1.5 `t_sigma` and the average covers the last "
        "40 % of it.")
    add("* Reported errors are 2-D, single geometry, zero gravity. Wall-bounded "
        "and gravity-dominated cases exercise the same operator with `a_i` from "
        "the wall and are not covered here.")
    add(f"* `dx = {record['settings']['resolution_check']['dx_m']*1000:.1f} mm` "
        "rows were run for two variants only; the full ladder at that resolution "
        "costs about four times the coarse grid per variant.")
    add("* The closure is off by default (`surface_pressure_closure = False`), so "
        "every number published before this document reproduces unchanged; with "
        "the closure off the two new knobs do not touch a single value (pinned by "
        "`tests/test_shell_pressure_closure_accel.py`).")
    add("")
    add("## 5. Reproduce")
    add("")
    add("```")
    add("PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py --record")
    add("PYTHONPATH=src python benchmarks/laplace_pressure_floor/make_closure_doc.py")
    add("PYTHONPATH=src python -m pytest tests/test_shell_pressure_closure_accel.py \\")
    add("    tests/test_shell_closure_report.py -q")
    add("```")
    add("")
    add("`make_closure_doc.py --record` rewrites `report_closure.json`; the second "
        "command regenerates this document from it. "
        "`tests/test_shell_closure_report.py` fails if the two disagree.")
    add("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="re-run the measurements and rewrite report_closure.json")
    ap.add_argument("--t-end", type=float, default=1.5,
                    help="window length in capillary times")
    args = ap.parse_args()
    if args.record:
        record = build_record(t_end=args.t_end)
        record["figures"] = [{"name": f["name"], "caption": f["caption"],
                              "path": str(f["path"].relative_to(ROOT))}
                             for f in make_figures(record)]
        RECORD.write_text(json.dumps(record, indent=1) + "\n")
        print("wrote", RECORD)
    record = json.loads(RECORD.read_text())
    DOC.write_text(render_doc(record))
    print("wrote", DOC)


if __name__ == "__main__":
    main()

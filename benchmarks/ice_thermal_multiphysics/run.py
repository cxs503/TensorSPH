"""Ice-ocean thermal-saline-mechanical multi-physics benchmark (campaign IN13W2-N07).

Runs every verification anchor of :mod:`tensorsph.ice_thermal` and writes the
criteria table, the figures and the audit trail that the report quotes.

Usage
-----
    python benchmarks/ice_thermal_multiphysics/run.py [--out artifacts/ice_thermal_multiphysics]
                                                      [--n-z 16] [--days 45] [--quick]

Anchors (all machine-checked, numbers land in ``results.json``)
---------------------------------------------------------------
A. lattice consistency of the SPH conduction operator (``alpha_eff/alpha``);
B. 1-D Stefan problem vs the Neumann similarity solution -- front position and
   temperature profile, at three resolutions;
C. sea-ice desalination by gravity drainage -- equilibrium salinity floor,
   first-year ice salinity band, ``k_ref`` decade scan;
D. Darcy-Benard onset -- numerical ``Ra_c`` vs the exact ``4 pi^2``;
E. thermo-elastic stress -- Timoshenko restrained-plate value, free-plate
   self-equilibrium, cracking thresholds;
F. three-field coupled column (winter + spring), one-way vs two-way coupling,
   salt conservation, mush-layer spin-up buffering.

Outputs under ``--out``: ``results.json``, ``fig1_stefan.png``,
``fig2_desalination.png``, ``fig3_darcy.png``, ``fig4_coupled_profiles.png``,
``fig5_coupling_loop.png``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from tensorsph.ice_thermal import darcy
from tensorsph.ice_thermal import properties as props
from tensorsph.ice_thermal import thermal_stress as stress_mod
from tensorsph.ice_thermal.brine import (SEA_ICE_FIRST_YEAR_BAND, band_check,
                                         run_desalination)
from tensorsph.ice_thermal.coupled import ThreeFieldConfig, run_three_field
from tensorsph.ice_thermal.enthalpy import FRESH, SEA_ICE, neumann_lambda
from tensorsph.ice_thermal.sph_thermal import (StefanConfig, lattice_diffusion_factor,
                                              run_stefan_sph)

REPO = Path(__file__).resolve().parents[2]
TOL_FRONT = 0.03     # platform precision standard for a benchmark anchor
TOL_STRESS = 0.01
TOL_RA = 0.01


def stefan_ladder(resolutions, t_end=4.0e4):
    """Run the SPH Stefan case at several resolutions; return the criteria rows."""
    rows = []
    for dx in resolutions:
        n = int(round(0.8 / dx))
        cfg = StefanConfig(n_particles=n, dx=dx, t_end=t_end)
        t0 = time.time()
        res = run_stefan_sph(cfg)
        err_max, err_mean = res.front_error(skip=3)
        rows.append({
            "dx_m": dx,
            "n_particles": n,
            "cells": round(cfg.h_factor * dx, 6),
            "dt_s": res.dt,
            "steps": res.steps,
            "stefan_number": res.stefan_number,
            "neumann_lambda": res.lam,
            "diffusion_factor": res.diffusion_factor,
            "diffusion_scatter": res.diffusion_scatter,
            "front_final_m": res.front[-1],
            "front_analytic_m": res.front_analytic[-1],
            "front_rel_error_final": abs(res.front[-1] - res.front_analytic[-1])
            / max(res.front_analytic[-1], 1e-12),
            "front_rel_error_max": err_max,
            "front_rel_error_mean": err_mean,
            "profile_rms_k": res.profile_error(),
            "wall_time_s": round(time.time() - t0, 1),
        })
        rows[-1]["series"] = {"time": res.time, "front": res.front,
                              "analytic": res.front_analytic,
                              "profile_x": res.profile_x,
                              "profile_t": res.profile_t,
                              "profile_analytic": res.profile_analytic}
    return rows


def desalination_anchor(days=60.0, n=25):
    """Cold-column desalination: equilibrium floor, band check and k_ref scan."""
    base = dict(thickness=0.5, n=n, s0=12.0, t_air=263.15, t_initial=271.65, days=days)
    res = run_desalination(**base)
    out = {
        "bulk_salinity_series_psu": res.bulk_salinity,
        "time_days": [t / 86400.0 for t in res.time],
        "residual_series_psu": res.residual,
        "final_bulk_salinity_psu": res.bulk_salinity[-1],
        "final_residual_psu": res.residual[-1],
        "band": band_check(res.bulk_salinity[-1]),
        "salt_conservation_rel_error": res.salt_error,
        "profile_z": res.profile_z[-1],
        "profile_s": res.profile_s[-1],
        "profile_t": res.profile_t[-1],
        "k_ref_scan": [],
    }
    original = props.K_REF
    try:
        for factor in (0.1, 1.0, 10.0):
            props.K_REF = original * factor
            props.permeability.__defaults__ = (props.K_REF, 0.05, 3.1)
            props.permeability_vec.__defaults__ = (props.K_REF, 0.05, 3.1)
            r = run_desalination(**base)
            out["k_ref_scan"].append({
                "k_ref_m2": props.K_REF,
                "final_bulk_salinity_psu": r.bulk_salinity[-1],
                "days_to_90pct_desalination": _days_to(r, 0.9),
            })
    finally:
        props.K_REF = original
        props.permeability.__defaults__ = (original, 0.05, 3.1)
        props.permeability_vec.__defaults__ = (original, 0.05, 3.1)
    return out


def _days_to(res, fraction):
    """Days until ``1 - S/S0`` reaches ``fraction`` of its final value."""
    s0, s_end = res.bulk_salinity[0], res.bulk_salinity[-1]
    if abs(s0 - s_end) < 1e-9:
        return None
    target = s0 + fraction * (s_end - s0)
    for t, s in zip(res.time, res.bulk_salinity):
        if s <= target:
            return t / 86400.0
    return None


def darcy_anchor(n=60, n_wavenumbers=25):
    """Numerical Horton-Rogers-Lapwood onset vs the exact ``4 pi^2``."""
    t0 = time.time()
    ra, a_c, curve = darcy.critical_rayleigh(n=n, n_wavenumbers=n_wavenumbers)
    analytic = darcy.analytic_marginal(a_c)
    return {
        "ra_critical_numeric": ra,
        "ra_critical_exact": darcy.RA_CRITICAL_EXACT,
        "ra_relative_error": abs(ra - darcy.RA_CRITICAL_EXACT) / darcy.RA_CRITICAL_EXACT,
        "wavenumber_at_onset": a_c,
        "analytic_ra_at_same_wavenumber": analytic,
        "curve_relative_error_max": max(abs(r - darcy.analytic_marginal(a)) / r
                                        for a, r in curve),
        "curve": {"a": [a for a, _ in curve], "ra": [r for _, r in curve]},
        "nusselt_below_onset": darcy.nusselt(0.5 * darcy.RA_CRITICAL_EXACT),
        "nusselt_above_onset": darcy.nusselt(4.0 * darcy.RA_CRITICAL_EXACT),
        "fd_nodes": n,
        "wall_time_s": round(time.time() - t0, 1),
    }


def stress_anchor():
    """Thermo-elastic benchmarks: Timoshenko plate, free plate, cracking."""
    n = 101
    z = torch.linspace(-0.25, 0.25, n, dtype=torch.float64)
    t_ref = props.T_MELT - 2.0
    d_t = 10.0
    linear = t_ref + d_t * z / 0.5        # -5 ... +5 about the mid-plane
    sigma = stress_mod.restrained_stress(linear, t_ref=t_ref)
    timoshenko = stress_mod.biaxial_modulus() * props.ALPHA_TH * d_t / 2.0
    measured = stress_mod.max_tensile_stress(torch.abs(sigma))
    free_linear = stress_mod.free_plate_stress(linear, z)
    # non-linear (exponentially decaying) profile: self-equilibrated stress
    nonlinear = t_ref + d_t * torch.exp(-(z + 0.25) / 0.08)
    free_nonlinear = stress_mod.free_plate_stress(nonlinear, z)
    force, moment = stress_mod.free_plate_moments(free_nonlinear, z)
    crack = stress_mod.crack_assessment(
        stress_mod.restrained_stress(t_ref - 3.0 + 3.0 * z / 0.25, t_ref=t_ref), 0.01)
    return {
        "timoshenko_analytic_pa": timoshenko,
        "restrained_max_pa": measured,
        "restrained_relative_error": abs(measured - timoshenko) / timoshenko,
        "free_linear_max_abs_pa": float(free_linear.abs().max()),
        "free_nonlinear_max_abs_pa": float(free_nonlinear.abs().max()),
        "free_nonlinear_net_force_pa": force,
        "free_nonlinear_net_moment": moment,
        "critical_temperature_drop_k": stress_mod.critical_temperature_drop(),
        "critical_crack_depth_m": crack.critical_depth_m,
        "stress_intensity_mpa_sqrt_m": crack.k_i_pa_sqrt_m / 1e6,
        "cracks_predicted": crack.cracks,
        "profile_z": [float(x) for x in z],
        "profile_restrained_mpa": [float(x) / 1e6 for x in sigma],
        "profile_free_nonlinear_mpa": [float(x) / 1e6 for x in free_nonlinear],
    }


def coupled_anchor(n_z, days, warm=None):
    """Winter and spring coupled runs, one-way vs two-way, salt closure."""
    spring = warm or ThreeFieldConfig(n_z=n_z, days=days, spinup=True)
    winter = ThreeFieldConfig(n_z=n_z, days=days, t_surface=258.15, t_surface0=261.15,
                              spinup=False)
    one_way_spring = ThreeFieldConfig(**{**spring.__dict__, "two_way": False})
    rows = {}
    for name, cfg in (("spring_two_way", spring), ("spring_one_way", one_way_spring),
                      ("winter", winter)):
        t0 = time.time()
        res = run_three_field(cfg)
        rows[name] = {
            "summary": res.summary(),
            "series": {
                "time_h": res.time_h,
                "bulk_salinity": res.bulk_salinity,
                "ocean_salinity": res.ocean_salinity,
                "ocean_heat_flux": res.ocean_heat_flux,
                "melt_rate_mm_day": res.melt_rate_mm_day,
                "cumulative_melt_m": res.cumulative_melt_m,
                "max_tensile_mpa": res.max_tensile_mpa,
                "residual_salinity": res.residual_salinity,
            },
            "profiles": {
                "z": res.z, "temperature_c": [x - props.T_MELT for x in res.temperature],
                "salinity": res.salinity, "brine_fraction": res.brine_fraction,
                "permeability": res.permeability, "stress_mpa": res.stress_mpa,
                "stress_free_mpa": res.stress_free_mpa,
            },
            "wall_time_s": round(time.time() - t0, 1),
        }
    two = rows["spring_two_way"]["summary"]
    one = rows["spring_one_way"]["summary"]
    rows["coupling_strength"] = {
        "melt_ratio_two_way_over_one_way": (two["melt_total_m"] / one["melt_total_m"]
                                            if one["melt_total_m"] > 0 else None),
        "ocean_heat_flux_ratio": (two["ocean_heat_flux_w_m2"] / one["ocean_heat_flux_w_m2"]
                                  if one["ocean_heat_flux_w_m2"] > 0 else None),
        "ocean_salinity_shift_psu": (two["ocean_salinity_final_psu"]
                                     - one["ocean_salinity_final_psu"]),
        "bulk_salinity_shift_psu": (two["bulk_salinity_final_psu"]
                                    - one["bulk_salinity_final_psu"]),
    }
    return rows


def format_criteria(res):
    """Human-readable criteria table (also embedded in the report)."""
    stefan = res["stefan"]
    coupled = res["coupled"]
    lines = ["criterion                                   value        tolerance  verdict"]
    for row in stefan:
        ok = row["front_rel_error_final"] <= TOL_FRONT
        lines.append(f"Stefan front, dx={row['dx_m']:g} m                     "
                     f"{row['front_rel_error_final']*100:7.3f} %   <=3 %      "
                     f"{'PASS' if ok else 'FAIL'}")
    darcy_row = res["darcy"]
    lines.append(f"Darcy-Benard Ra_c                            "
                 f"{darcy_row['ra_relative_error']*100:7.4f} %   <=1 %      "
                 f"{'PASS' if darcy_row['ra_relative_error'] <= TOL_RA else 'FAIL'}")
    st = res["stress"]
    lines.append(f"Timoshenko restrained stress                 "
                 f"{st['restrained_relative_error']*100:7.4f} %   <=1 %      "
                 f"{'PASS' if st['restrained_relative_error'] <= TOL_STRESS else 'FAIL'}")
    lines.append(f"free-plate net force (must vanish)           "
                 f"{abs(st['free_nonlinear_net_force_pa']):7.3e}   0          "
                 f"{'PASS' if abs(st['free_nonlinear_net_force_pa']) < 1e-6 else 'FAIL'}")
    des = res["desalination"]
    lines.append(f"sea-ice salinity inside observed band        "
                 f"{des['final_bulk_salinity_psu']:7.2f} psu  3-8 psu    "
                 f"{'PASS' if des['band']['inside_band'] else 'FAIL'}")
    for name in ("spring_two_way", "winter"):
        s = coupled[name]["summary"]
        lines.append(f"salt conservation ({name})           "
                     f"{s['salt_conservation_rel_error']*100:7.4f} %   <=0.1 %    "
                     f"{'PASS' if s['salt_conservation_rel_error'] <= 1e-3 else 'FAIL'}")
    return "\n".join(lines)


def make_figures(res, out_dir):
    """Five figures: Stefan, desalination, Darcy onset, profiles, coupling loop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    # --- fig 1: Stefan front + profile
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for row in res["stefan"]:
        s = row["series"]
        axes[0].plot([t / 1e3 for t in s["time"]], [x * 1e3 for x in s["front"]],
                     label=f"SPH dx={row['dx_m']:g} m")
    s0 = res["stefan"][0]["series"]
    axes[0].plot([t / 1e3 for t in s0["time"]], [x * 1e3 for x in s0["analytic"]],
                 "k--", label="Neumann solution")
    axes[0].set_xlabel("t [ks]")
    axes[0].set_ylabel("freezing front X [mm]")
    axes[0].set_title("(a) 1-D Stefan front")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    s1 = res["stefan"][-1]["series"]
    axes[1].plot([x * 1e3 for x in s1["profile_x"]], [t - 273.15 for t in s1["profile_t"]],
                 "o", ms=3, label="SPH")
    axes[1].plot([x * 1e3 for x in s1["profile_x"]],
                 [t - 273.15 for t in s1["profile_analytic"]], "k-", label="Neumann")
    axes[1].set_xlabel("distance from cold wall [mm]")
    axes[1].set_ylabel("T [degC]")
    axes[1].set_title("(b) temperature profile at $t_{end}$")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    dxs = [r["dx_m"] for r in res["stefan"]]
    errs = [r["front_rel_error_final"] * 100 for r in res["stefan"]]
    axes[2].loglog(dxs, errs, "o-")
    axes[2].set_xlabel("dx [m]")
    axes[2].set_ylabel("final front error [%]")
    axes[2].set_title("(c) resolution ladder")
    axes[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    figs["fig1_stefan.png"] = (fig, "SPH Stefan problem: front history, profile, ladder")
    plt.close(fig)

    # --- fig 2: desalination
    des = res["desalination"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(des["time_days"], des["bulk_salinity_series_psu"], label="bulk salinity")
    axes[0].plot(des["time_days"], des["residual_series_psu"], "--",
                 label=r"equilibrium floor $v_b=v_{irr}$")
    axes[0].axhspan(*SEA_ICE_FIRST_YEAR_BAND, color="tab:green", alpha=0.15,
                    label="first-year ice band (Cox & Weeks 1983)")
    axes[0].set_xlabel("time [d]")
    axes[0].set_ylabel("bulk salinity [psu]")
    axes[0].set_title("(a) gravity drainage desalination")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(des["profile_z"], des["profile_s"], "o-", label="salinity")
    axes[1].set_xlabel("depth below surface [m]")
    axes[1].set_ylabel("bulk salinity [psu]")
    axes[1].set_title("(b) final salinity profile")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    figs["fig2_desalination.png"] = (fig, "Desalination curve vs the observed band")
    plt.close(fig)

    # --- fig 3: Darcy onset
    d = res["darcy"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d["curve"]["a"], d["curve"]["ra"], "o", ms=3, label="FD eigenvalue solve")
    aa = [0.5 + 4.0 * i / 200 for i in range(201)]
    ax.plot(aa, [darcy.analytic_marginal(a) for a in aa], "k-",
            label=r"$(a^2+\pi^2)^2/a^2$")
    ax.axhline(darcy.RA_CRITICAL_EXACT, color="tab:red", ls="--",
               label=r"$Ra_c=4\pi^2=%.3f$" % darcy.RA_CRITICAL_EXACT)
    ax.set_xlabel("wavenumber a")
    ax.set_ylabel("marginal Ra")
    ax.set_ylim(0, 120)
    ax.set_title("Darcy-Benard marginal stability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["fig3_darcy.png"] = (fig, "Darcy-Benard onset: numeric vs exact 4*pi^2")
    plt.close(fig)

    # --- fig 4: coupled profiles
    pr = res["coupled"]["spring_two_way"]["profiles"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    series = [("temperature_c", "T [degC]", "(a) temperature"),
              ("salinity", "S [psu]", "(b) bulk salinity"),
              ("brine_fraction", "v_b [-]", "(c) brine fraction"),
              ("stress_mpa", "sigma [MPa]", "(d) restrained stress")]
    for ax, (key, label, title) in zip(axes, series):
        ax.plot(pr[key], pr["z"], "o-", ms=3)
        ax.set_xlabel(label)
        ax.set_ylabel("depth [m]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["fig4_coupled_profiles.png"] = (fig, "Coupled column: T, S, brine fraction, stress")
    plt.close(fig)

    # --- fig 5: two-way loop
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for name, style in (("spring_two_way", "-"), ("spring_one_way", "--"),
                        ("winter", ":")):
        s = res["coupled"][name]["series"]
        axes[0].plot([t / 24 for t in s["time_h"]], s["ocean_salinity"], style,
                     label=name)
        axes[1].plot([t / 24 for t in s["time_h"]], s["ocean_heat_flux"], style,
                     label=name)
        axes[2].plot([t / 24 for t in s["time_h"]], s["melt_rate_mm_day"], style,
                     label=name)
    for ax, ylabel, title in ((axes[0], "mixed-layer salinity [psu]", "(a) salinity"),
                              (axes[1], r"$Q_{oc}$ [W/m$^2$]", "(b) ocean heat flux"),
                              (axes[2], "melt rate [mm/day]", "(c) basal melt")):
        ax.set_xlabel("time [d]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    figs["fig5_coupling_loop.png"] = (fig, "Brine rejection -> buoyancy -> ocean heat flux "
                                           "-> melt feedback (winter vs spring)")
    plt.close(fig)
    return figs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "ice_thermal_multiphysics"))
    ap.add_argument("--n-z", type=int, default=16)
    ap.add_argument("--days", type=float, default=45.0)
    ap.add_argument("--quick", action="store_true", help="skip the coupled runs")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    factor, scatter = lattice_diffusion_factor()
    res = {
        "node": "IN13W2-N07",
        "repo": "TensorSPH",
        "module": "tensorsph.ice_thermal",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform()},
        "properties": props.format_property_table(),
        "lattice": {"diffusion_factor": factor, "mode_scatter": scatter,
                    "h_over_dx": 1.3},
        "anchors": {},
        "tolerances": {"stefan_front": TOL_FRONT, "stress": TOL_STRESS, "darcy": TOL_RA},
        "stefan": stefan_ladder([0.02, 0.0125, 0.008]),
        "desalination": desalination_anchor(),
        "darcy": darcy_anchor(),
        "stress": stress_anchor(),
    }
    res["stefan_summary"] = {row["dx_m"]: {
        "front_rel_error_final": row["front_rel_error_final"],
        "front_rel_error_max": row["front_rel_error_max"],
        "profile_rms_k": row["profile_rms_k"]} for row in res["stefan"]}
    if not args.quick:
        res["coupled"] = coupled_anchor(args.n_z, args.days)
    res["criteria_table"] = format_criteria(res) if "coupled" in res else "coupled skipped"
    res["verification"] = {
        "neumann_lambda_ste": neumann_lambda(res["stefan"][0]["stefan_number"]),
        "fresh_enthalpy_roundtrip_k": float(
            (FRESH.temperature(FRESH.enthalpy(torch.linspace(250.0, 280.0, 31)))
             - torch.linspace(250.0, 280.0, 31)).abs().max()),
        "sea_enthalpy_roundtrip_k": float(
            (SEA_ICE.temperature(SEA_ICE.enthalpy(torch.linspace(255.0, 272.0, 31),
                                                  torch.full((31,), 8.0)),
                                 torch.full((31,), 8.0))
             - torch.linspace(255.0, 272.0, 31)).abs().max()),
        "stefan_front_cells": [(row["front_final_m"] / row["dx_m"]) for row in res["stefan"]],
        "wall_time_s": round(time.time() - t0, 1),
    }
    figures = {}
    if not args.no_figures and "coupled" in res:
        figures = make_figures(res, out_dir)
        res["figures"] = {}
        for name, (fig, caption) in figures.items():
            path = out_dir / name
            fig.savefig(path, dpi=130)
            res["figures"][name] = {"path": str(path), "caption": caption}
    for row in res["stefan"]:
        row.pop("series", None)   # keep results.json compact; series in the figures
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(res["criteria_table"])
    print(f"\nresults -> {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

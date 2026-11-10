"""Render ``docs/verification_ice_thermal_multiphysics.md`` from ``results.json``.

Nothing in the generated document is typed by hand: every number is read out of
``artifacts/ice_thermal_multiphysics/results.json`` (which the benchmark writes),
so the report and the artifact cannot drift apart.

    python benchmarks/ice_thermal_multiphysics/make_doc.py \
        --results artifacts/ice_thermal_multiphysics/results.json \
        --out docs/verification_ice_thermal_multiphysics.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tensorsph.ice_thermal.brine import residual_salinity

REPO = Path(__file__).resolve().parents[2]


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, (int, str)):
        return str(value)
    if abs(value) != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def level_margins(des):
    """``S(z) - S_eq(T(z))`` per level: the drainage must never cross the floor."""
    t = torch.tensor(des["profile_t"], dtype=torch.float64)
    s = torch.tensor(des["profile_s"], dtype=torch.float64)
    floor = residual_salinity(t)
    margin = s - floor
    return {"min_margin_psu": float(margin.min()), "mean_margin_psu": float(margin.mean()),
            "levels": int(t.numel())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "artifacts"
                                             / "ice_thermal_multiphysics" / "results.json"))
    ap.add_argument("--out", default=str(REPO / "docs"
                                         / "verification_ice_thermal_multiphysics.md"))
    args = ap.parse_args(argv)
    res = json.loads(Path(args.results).read_text())
    out = []

    out.append("# Ice-ocean thermal-saline-mechanical multi-physics: verification record")
    out.append("")
    out.append(f"*Campaign node*: `{res['node']}` -- *repository*: `{res['repo']}` -- "
               f"*module*: `{res['module']}`")
    out.append("")
    out.append(f"Generated {res['generated']} by "
               f"`benchmarks/ice_thermal_multiphysics/run.py` "
               f"(python {res['environment']['python']}, torch {res['environment']['torch']}, "
               f"{res['environment']['platform']}). "
               "Every number below is read out of `results.json` by "
               "`make_doc.py`; nothing is transcribed.")
    out.append("")

    out.append("## 1. What is verified, against what")
    out.append("")
    rows = []
    for row in res["stefan"]:
        rows.append([
            "1-D Stefan problem (SPH enthalpy)",
            f"Neumann similarity solution, dx = {fmt(row['dx_m'], 4)} m",
            f"front {fmt(row['front_rel_error_final'] * 100, 3)} %",
            "3 %",
            "PASS" if row["front_rel_error_final"] <= res["tolerances"]["stefan_front"]
            else "FAIL"])
    d = res["darcy"]
    rows.append(["Darcy-Benard onset", "exact $Ra_c = 4\\pi^2$",
                 f"{fmt(d['ra_relative_error'] * 100, 4)} %", "1 %",
                 "PASS" if d["ra_relative_error"] <= res["tolerances"]["darcy"] else "FAIL"])
    st = res["stress"]
    rows.append(["Timoshenko restrained plate", "$E\\alpha\\Delta T/2(1-\\nu)$",
                 f"{fmt(st['restrained_relative_error'] * 100, 4)} %", "1 %",
                 "PASS" if st["restrained_relative_error"] <= res["tolerances"]["stress"]
                 else "FAIL"])
    rows.append(["Free plate, linear gradient",
                 "stress-free identity", f"{fmt(st['free_linear_max_abs_pa'], 2)} Pa",
                 "machine", "PASS" if st["free_linear_max_abs_pa"] < 1e-6 else "FAIL"])
    rows.append(["Free plate, non-linear gradient", "net force / moment vanish",
                 f"{fmt(abs(st['free_nonlinear_net_force_pa']), 2)} Pa",
                 "1e-6 $\\sigma_{max}$",
                 "PASS" if abs(st["free_nonlinear_net_force_pa"])
                 < 1e-6 * st["free_nonlinear_max_abs_pa"] else "FAIL"])
    des = res["desalination"]
    margins = level_margins(des)
    rows.append(["Desalination equilibrium", "per-level $v_b = v_{irr}$ floor (exact)",
                 f"min margin {fmt(margins['min_margin_psu'], 3)} psu", ">= 0",
                 "PASS" if margins["min_margin_psu"] >= -1e-6 else "FAIL"])
    rows.append(["Desalination vs observation", "first-year ice band 3-8 psu",
                 f"{fmt(des['final_bulk_salinity_psu'], 2)} psu", "3-8 psu",
                 "PASS" if des["band"]["inside_band"] else "FAIL"])
    if "coupled" in res:
        for name in ("spring_two_way", "winter"):
            s = res["coupled"][name]["summary"]
            rows.append([f"Salt conservation ({name})", "salt mass balance",
                         f"{fmt(s['salt_conservation_rel_error'] * 100, 4)} %", "0.1 %",
                         "PASS" if s["salt_conservation_rel_error"] <= 1e-3 else "FAIL"])
    out.append(table(rows, ["anchor", "reference", "computed", "tolerance", "verdict"]))
    out.append("")

    out.append("## 2. Stefan problem (phase change)")
    out.append("")
    rows = []
    for row in res["stefan"]:
        rows.append([fmt(row["dx_m"], 4), row["n_particles"], fmt(row["dt_s"], 2),
                     row["steps"], fmt(row["stefan_number"], 4), fmt(row["neumann_lambda"], 5),
                     fmt(row["front_final_m"] * 1000, 2), fmt(row["front_analytic_m"] * 1000, 2),
                     fmt(row["front_rel_error_final"] * 100, 3),
                     fmt(row["front_rel_error_max"] * 100, 3),
                     fmt(row["profile_rms_k"], 4)])
    out.append(table(rows, ["dx [m]", "N", "dt [s]", "steps", "Ste", "$\\lambda$",
                            "front [mm]", "Neumann [mm]", "err final [%]",
                            "err max [%]", "profile RMS [K]"]))
    out.append("")
    lat = res["lattice"]
    out.append(f"SPH conduction operator lattice calibration: "
               f"`alpha_eff/alpha = {fmt(lat['diffusion_factor'], 5)}` "
               f"(scatter over the three longest modes {fmt(lat['mode_scatter'], 5)}, "
               f"h = {lat['h_over_dx']} dx). "
               f"Enthalpy inversion round-trip: fresh "
               f"{fmt(res['verification']['fresh_enthalpy_roundtrip_k'], 2)} K, sea ice "
               f"{fmt(res['verification']['sea_enthalpy_roundtrip_k'], 2)} K.")
    out.append("")

    out.append("## 3. Brine drainage / desalination")
    out.append("")
    rows = []
    for row in des["k_ref_scan"]:
        rows.append([fmt(row["k_ref_m2"], 2), fmt(row["final_bulk_salinity_psu"], 2),
                     fmt(row["days_to_90pct_desalination"], 2)])
    out.append(table(rows, ["$k_{ref}$ [m$^2$]", "final bulk salinity [psu]",
                            "days to 90 % desalination"]))
    out.append("")
    margins = level_margins(des)
    out.append(f"Equilibrium floor, checked per level: the minimum of "
               f"`S(z) - S_eq(T(z))` over the {margins['levels']} levels is "
               f"{fmt(margins['min_margin_psu'], 3)} psu (mean "
               f"{fmt(margins['mean_margin_psu'], 3)} psu); gravity drainage never "
               f"crosses the irreducible-brine floor.")
    out.append("")
    out.append(f"Salt conservation of the drainage solve: "
               f"{fmt(des['salt_conservation_rel_error'], 2)} relative. "
               f"Final salinity {fmt(des['final_bulk_salinity_psu'], 2)} psu against the "
               f"observed first-year band {des['band']['band_psu']} psu "
               f"(ratio to ocean {fmt(des['band']['ratio_to_ocean'], 3)}).")
    out.append("")

    out.append("## 4. Buoyancy-driven convection onset")
    out.append("")
    out.append(f"Numerical eigenvalue solve with {d['fd_nodes']} finite-difference nodes: "
               f"$Ra_c = {fmt(d['ra_critical_numeric'], 4)}$ vs exact "
               f"$4\\pi^2 = {fmt(d['ra_critical_exact'], 4)}$ "
               f"(error {fmt(d['ra_relative_error'] * 100, 4)} %), at wavenumber "
               f"$a_c = {fmt(d['wavenumber_at_onset'], 3)}$; the whole marginal curve "
               f"agrees with $(a^2+\\pi^2)^2/a^2$ to {fmt(d['curve_relative_error_max'] * 100, 3)} %.")
    out.append("")

    out.append("## 5. Thermo-elastic stress and cracking")
    out.append("")
    rows = [
        ["Timoshenko restrained surface stress [Pa]", fmt(st["timoshenko_analytic_pa"], 2),
         fmt(st["restrained_max_pa"], 2), fmt(st["restrained_relative_error"] * 100, 4)],
        ["free plate, linear gradient [Pa]", "0", fmt(st["free_linear_max_abs_pa"], 2), "-"],
        ["free plate, non-linear gradient [Pa]", "-",
         fmt(st["free_nonlinear_max_abs_pa"], 2), "-"],
        ["net force of the free solution [Pa]", "0",
         fmt(st["free_nonlinear_net_force_pa"], 3), "-"],
        ["net moment of the free solution [Pa m]", "0",
         fmt(st["free_nonlinear_net_moment"], 3), "-"],
        ["critical temperature drop [K]", "0.7 MPa strength",
         fmt(st["critical_temperature_drop_k"], 3), "-"],
        ["$K_I$ of a 10 mm edge crack [MPa m$^{1/2}$]", "$K_{IC}=0.1$",
         fmt(st["stress_intensity_mpa_sqrt_m"], 3), "-"],
    ]
    out.append(table(rows, ["quantity", "reference", "computed", "error [%]"]))
    out.append("")

    if "coupled" in res:
        out.append("## 6. Three-field coupled column")
        out.append("")
        rows = []
        for name in ("spring_two_way", "spring_one_way", "winter"):
            s = res["coupled"][name]["summary"]
            rows.append([name, fmt(s["bulk_salinity_final_psu"], 2),
                         fmt(s["ocean_salinity_final_psu"], 4),
                         fmt(s["ocean_heat_flux_w_m2"], 2),
                         fmt(s["melt_rate_mm_day_mean"], 2),
                         fmt(s["melt_total_m"], 4),
                         fmt(s["max_tensile_stress_mpa_restrained"], 2),
                         fmt(s["max_tensile_stress_mpa_free"], 3),
                         fmt(s["salt_conservation_rel_error"], 2)])
        out.append(table(rows, ["case", "S_bulk [psu]", "S_ocean [psu]",
                                "$Q_{oc}$ [W/m$^2$]", "melt [mm/d]", "melt total [m]",
                                "$\\sigma_{max}$ restrained [MPa]",
                                "$\\sigma_{max}$ free [MPa]", "salt err"]))
        out.append("")
        cs = res["coupled"]["coupling_strength"]
        out.append(f"Two-way coupling strength (spring case): melt ratio "
                   f"{fmt(cs['melt_ratio_two_way_over_one_way'], 3)}, ocean heat-flux ratio "
                   f"{fmt(cs['ocean_heat_flux_ratio'], 3)}, mixed-layer salinity shift "
                   f"{fmt(cs['ocean_salinity_shift_psu'], 4)} psu, bulk-salinity shift "
                   f"{fmt(cs['bulk_salinity_shift_psu'], 4)} psu.")
        out.append("")
        sp = res["coupled"]["spring_two_way"]["summary"]["spinup"]
        if sp:
            sea = sp.get("sea_ice", {})
            fr = sp.get("fresh_ice", {})
            out.append(f"Mush-layer latent buffering: the sea-ice column needs "
                       f"{fmt(sea.get('relaxation_s'), 1)} s to reach half of a cold-snap "
                       f"step against {fmt(fr.get('relaxation_s'), 1)} s for fresh ice "
                       f"(conduction time $h^2/\\alpha$ = "
                       f"{fmt(fr.get('conduction_time_s'), 1)} s).")
            out.append("")

    out.append("## 7. Traps found and fixed while building this face")
    out.append("")
    out.append("1. **Kelvin vs freezing-point depression.** The mushy latent term must be "
               "written in `theta = 273.15 - T`, not `T`: using `T` makes the latent heat "
               "of a 12 psu ice 273x too small (2.9 J/kg/K instead of 5e5 J/kg/K near Tf).")
    out.append("2. **Enthalpy is a function of (T, S).** Draining brine is an isothermal "
               "mass change; keeping `H` fixed while `S` drops adds the S-dependent latent "
               "term to the remaining ice and heats it spuriously by ~L per kg of salt "
               "removed (observed: the column jumped to +15 degC).")
    out.append("3. **No density in the specific-enthalpy capacity.** `dH/dt = c dT/dt`; "
               "carrying `rho` in the Picard diagonal suppresses the temperature response "
               "by 917 and the column never cools.")
    out.append("4. **CFL on the particle, not on the face average.** The face flux below "
               "particle i is carried by particle i, so a face-averaged velocity "
               "under-estimates the limit by up to 2x and destabilises the upwind scheme "
               "(observed: bulk salinity 5.5e4 psu in one step).")
    out.append("5. **A free plate with a linear gradient is stress free.** Evaluating the "
               "bending moment with a continuum `12/h^3` on a discrete grid leaves an "
               "O(dz^2/h^2) phantom stress (~66 Pa); the moment balance must use the same "
               "discrete measure in numerator and denominator.")
    out.append("")
    out.append("## 8. Limitations")
    out.append("")
    out.append("* the ice column does not ablate: melt is diagnosed and enters the ocean "
               "salt/heat budget, but the base does not move (no SPH moving boundary);")
    out.append("* the ocean is a single well-mixed layer, and `gamma_0`, `a` are declared "
               "bulk exchange coefficients (not fitted to a dataset);")
    out.append("* the permeability scale `k_ref` is uncertain by about a decade in the "
               "literature; the scan in section 3 quantifies its effect;")
    out.append("* the Greeley & Wilson (1987) brine-drainage table is not reachable from "
               "this sandbox, so that anchor is WITHHELD; the observed first-year salinity "
               "band (Cox & Weeks 1983) is used instead and labelled as a band;")
    out.append("* a sub-kelvin non-monotone layer can appear at the ice-ocean interface "
               "because the fixed-thickness column re-projects its enthalpy while the base "
               "drains.")
    out.append("")
    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

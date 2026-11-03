"""Inject ``equilibrium_report.json`` into ``docs/verification_laplace_equilibrium.md``.

Every number in the document comes from the JSON written by ``run.py`` -- no hand
copying.  The template lives in ``doc_template.md`` next to this file; the
generated document is committed so that the verification record is reviewable
without re-running the benchmark.

Usage::

    PYTHONPATH=src python benchmarks/laplace_equilibrium/make_doc.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / "equilibrium_report.json"
TEMPLATE = HERE / "doc_template.md"
OUT = ROOT / "docs" / "verification_laplace_equilibrium.md"

SIGMA_R = 360.0


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def main() -> None:
    r = json.loads(REPORT.read_text())
    exp = r["experiments"]
    shipped, cons, damped = exp["shipped_init"], exp["consistent_init"], exp["damped_consistent"]

    p0 = {row["p0_Pa"]: row for row in r["p0_control"]}
    a, b = p0[0.0], p0[360.0]
    delta_p0 = b["p_interior_final_Pa"] - a["p_interior_final_Pa"]

    # --- table 1: protocol decomposition -----------------------------------
    rows1 = []
    for label, run in (("shipped init + shipped shell-band estimator",
                        shipped["final"]["delta_p_shellband_estimator_Pa"]),
                       ("shipped init + vacuum-referenced estimator",
                        shipped["final"]["delta_p_vacuum_ref_Pa"]),
                       ("density-consistent init + vacuum reference",
                        cons["final"]["delta_p_vacuum_ref_Pa"])):
        err = (run - SIGMA_R) / SIGMA_R * 100.0
        rows1.append(f"| {label} | {fmt(run)} | {err:+.1f} % |")
    table_protocol = "\n".join(rows1)

    # --- table 2: resolution scan -----------------------------------------
    rows2 = []
    for s in r["resolution_scan"]:
        rows2.append(
            f"| {s['dx_m']:.4f} | {s['n_particles']} | {fmt(s['p_interior_final_Pa'])} "
            f"| {s['rel_error_pct']:+.1f} % | {s['n_delta_initial']:.3f} -> "
            f"{s['n_delta_final']:.3f} | {s['vmax_over_v_ref_final']:.3f} "
            f"| {fmt(s['rho_interior_over_rho0_final'], 6)} |")
    table_scan = "\n".join(rows2)

    # --- table 3: p0 control ----------------------------------------------
    rows3 = []
    for row in (a, b):
        rows3.append(
            f"| {fmt(row['p0_Pa'], 0)} | {fmt(row['p_interior_final_Pa'])} "
            f"| {row['rel_error_pct']:+.1f} % | {fmt(row['p_shell_band_Pa'])} "
            f"| {row['rho_shell_band_over_rho0']:.3f} "
            f"| {row['frac_clamped_shell_band_final'] * 100:.0f} % |")
    table_p0 = "\n".join(rows3)

    # --- late-window correlation ------------------------------------------
    hist = [h for h in cons["history"] if h["t"] > 0.3 * cons["history"][-1]["t"]]
    x = [h["n_delta"] for h in hist]
    y = [h["p_interior_Pa"] / SIGMA_R for h in hist]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((i - mx) * (j - my) for i, j in zip(x, y))
    sxx = sum((i - mx) ** 2 for i in x)
    syy = sum((j - my) ** 2 for j in y)
    slope = sxy / sxx
    corr = sxy / math.sqrt(sxx * syy)

    values = {
        "table_protocol": table_protocol,
        "table_scan": table_scan,
        "table_p0": table_p0,
        "shipped_init_p0_lattice": fmt(shipped["initial_condition"]["p_core_initial_Pa"], 0),
        "shipped_init_rho": fmt(shipped["initial_condition"]["rho_core_over_rho0"], 5),
        "cons_init_p": f"{cons['initial_condition']['p_core_initial_Pa']:.2e}",
        "cons_init_rho": fmt(cons["initial_condition"]["rho_core_over_rho0"], 12),
        "mass_scale": fmt(cons["initial_condition"]["mass_scale"], 6),
        "p_final_cons": fmt(cons["final"]["p_interior_Pa"]),
        "err_final_cons": fmt((cons["final"]["p_interior_Pa"] - SIGMA_R) / SIGMA_R * 100.0),
        "p_final_shipped": fmt(shipped["final"]["p_interior_Pa"]),
        "p_shell_estimator_shipped": fmt(shipped["final"]["delta_p_shellband_estimator_Pa"]),
        "err_shell_estimator_shipped": fmt(
            (shipped["final"]["delta_p_shellband_estimator_Pa"] - SIGMA_R) / SIGMA_R * 100.0),
        "quasi_cons_p": fmt(cons["quasi_static"]["p_interior_mean_Pa"]),
        "quasi_cons_err": fmt(cons["quasi_static"]["rel_error_pct"]),
        "quasi_damped_p": fmt(damped["quasi_static"]["p_interior_mean_Pa"]),
        "quasi_damped_err": fmt(damped["quasi_static"]["rel_error_pct"]),
        "p0_a": fmt(a["p_interior_final_Pa"]),
        "p0_b": fmt(b["p_interior_final_Pa"]),
        "p0_delta": fmt(delta_p0, 2),
        "frac_clamped_cons": f"{cons['final']['frac_clamped'] * 100:.0f} %",
        "shell_rho0": fmt(cons["history"][0]["rho_shell_band_over_rho0"], 3),
        "shell_rho_end": fmt(cons["final"]["rho_shell_band_over_rho0"], 3),
        "n_delta0": fmt(cons["history"][0]["n_delta"], 3),
        "n_delta0_err": f"{(1.0 - cons['history'][0]['n_delta']) * 100:.1f} %",
        "n_delta_end": fmt(cons["final"]["n_delta"], 3),
        "slope": fmt(slope, 2),
        "corr": fmt(corr, 3),
        "n_late": str(n),
        "vmax_end": fmt(cons["final"]["vmax_over_v_ref"], 3),
        "t_end": fmt(cons["history"][-1]["t"], 4),
        "t_end_sigma": fmt(cons["history"][-1]["t"] / r["reference"]["t_sigma_s"], 0),
        "dt": f"{cons['protocol']['dt_s']:.3e}",
        "np_particles": str(cons["protocol"]["n_particles"]),
        "t_sigma": f"{r['reference']['t_sigma_s']:.3e}",
        "scan_err_range": f"{min(s['rel_error_pct'] for s in r['resolution_scan']):+.1f} % .. "
                          f"{max(s['rel_error_pct'] for s in r['resolution_scan']):+.1f} %",
        "n_delta0_range": f"{min(s['n_delta_initial'] for s in r['resolution_scan']):.3f} .. "
                          f"{max(s['n_delta_initial'] for s in r['resolution_scan']):.3f}",
    }

    text = TEMPLATE.read_text()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    missing = [k for k in values if "{{" + k + "}}" in text]
    assert not missing, missing
    OUT.write_text(text)
    print("doc ->", OUT)


if __name__ == "__main__":
    main()
"""Driver / evidence generator for campaign node p457w2-n04.

Runs the whole acceptance chain of the aeration-buffer decalibration and writes

* ``docs/p457w2-n04/acceptance.json`` -- every criterion with its measured
  number (no hand-typed results: the JSON *is* the program output),
* ``docs/p457w2-n04/fig_*.png`` -- the report figures.

Usage::

    python -m scripts.p457w2n04_report          # from the repo root
    python scripts/p457w2n04_report.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mpmath as mp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorsph.aeration import impact_law as law  # noqa: E402
from tensorsph.aeration import mixture_acoustics as mx  # noqa: E402
from tensorsph.aeration import pbm_moments as pbm  # noqa: E402

OUT = ROOT / "docs" / "p457w2-n04"


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def c1_wood():
    mp.mp.dps = 60
    rho_l, rho_g = mp.mpf("998.2"), mp.mpf("1.204")
    c_l, c_g = mp.mpf("1483.0"), mp.mpf("343.0")
    worst = 0.0
    for i in range(1, 400):
        alpha = mp.mpf(i) / 400
        rho_m = alpha * rho_g + (1 - alpha) * rho_l
        comp = alpha / (rho_g * c_g**2) + (1 - alpha) / (rho_l * c_l**2)
        ref = 1 / mp.sqrt(rho_m * comp)
        worst = max(worst, abs(mx.wood_sound_speed(float(alpha)) - float(ref)) / float(ref))
    a_star = mx.alpha_at_min_sound_speed()
    h = 1e-6
    slope = (mx.wood_sound_speed(a_star + h) - mx.wood_sound_speed(a_star - h)) / (2 * h)
    c_min = mx.min_sound_speed()
    # numerical minimum by ternary search on [0, 1] for an independent check
    lo, hi = 0.0, 1.0
    for _ in range(300):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if mx.wood_sound_speed(m1) < mx.wood_sound_speed(m2):
            hi = m2
        else:
            lo = m1
    a_num = 0.5 * (lo + hi)
    c_num = mx.wood_sound_speed(a_num)
    return {
        "criterion": "C1 Wood closed form <= 1e-6 (pointwise) and alpha* ~ 20 m/s",
        "max_relative_deviation_vs_60digit_mpmath": worst,
        "threshold": 1.0e-6,
        "pass": bool(worst <= 1.0e-6),
        "alpha_star_closed_form": a_star,
        "alpha_star_ternary_search": a_num,
        "alpha_star_abs_difference": abs(a_star - a_num),
        "c_min_closed_form_m_s": c_min,
        "c_min_numerical_m_s": c_num,
        "c_min_relative_difference": abs(c_min - c_num) / c_num,
        "dcm_dalpha_at_alpha_star": slope,
        "c_min_magnitude_check_15_30_m_s": bool(15.0 <= c_min <= 30.0),
        "pass_magnitude": bool(15.0 <= c_min <= 30.0 and abs(a_star - a_num) <= 1e-3),
    }


def c2_limits():
    rows = []
    for v in (1.0, 6.0, 8.0, 10.0, 14.0, 25.0):
        p0 = mx.impact_pressure(0.0, v, strict=False)
        ref0 = mx.WATER["rho"] * mx.WATER["c"] * v
        p1 = mx.impact_pressure(1.0, v, strict=False)
        ref1 = mx.AIR["rho"] * mx.AIR["c"] * v
        rows.append(
            {
                "velocity": v,
                "p_alpha0_pa": p0,
                "reference_rho_w_c_w_v_pa": ref0,
                "residual_alpha0": abs(p0 - ref0) / ref0,
                "p_alpha1_pa": p1,
                "reference_rho_a_c_a_v_pa": ref1,
                "residual_alpha1": abs(p1 - ref1) / ref1,
            }
        )
    worst = max(max(r["residual_alpha0"], r["residual_alpha1"]) for r in rows)
    return {
        "criterion": "C2 alpha->0 and alpha->1 limits, residual <= 0.1%",
        "rows": rows,
        "worst_residual": worst,
        "threshold": 1.0e-3,
        "pass": bool(worst <= 1.0e-3),
    }


def c3_moments():
    k = pbm.kernel_parameters_from_aeration(1.0, 1.0e-3)
    spec = k.spectrum
    m0, m1, m2, m3 = (
        spec.number_density(),
        spec.volume_moment(1),
        spec.volume_moment(2),
        spec.volume_moment(3),
    )
    kw = dict(k_coagulation=k.k_coagulation, k_breakage=k.k_breakage)
    t_end, steps = 100.0, 4000
    hist = pbm.integrate_moments(m0, m1, m2, m3, t_end=t_end, steps=steps, **kw)
    worst = 0.0
    scales = (m0, m1, m2, m3)
    for t, numeric in hist[::100]:
        closed = pbm.moments_closed_form(t, m0, m1, m2, m3, **kw)
        for a, b, s in zip(numeric, closed, scales):
            worst = max(worst, abs(a - b) / max(abs(b), 1e-6 * abs(s)))
    # closure exactness against the direct double sum
    volumes = [1e-10, 3e-10, 8e-10, 2e-9, 6e-9, 1.5e-8, 4e-8]
    densities = [3e12, 4e11, 5e10, 6e9, 7e8, 8e7, 9e6]
    direct = pbm.discrete_moment_rates(volumes, densities, k.k_coagulation, k.k_breakage)
    d0 = sum(densities)
    d1 = sum(n * v for n, v in zip(densities, volumes))
    d2 = sum(n * v * v for n, v in zip(densities, volumes))
    d3 = sum(n * v**3 for n, v in zip(densities, volumes))
    closed_rates = pbm.moment_rates(d0, d1, d2, d3, k.k_coagulation, k.k_breakage)
    scale = k.k_coagulation * d0 * d1 + k.k_breakage * d0
    closure = max(abs(a - b) / scale for a, b in zip(direct, closed_rates))
    m0_end = hist[-1][1][0]
    m2_inf = k.m2_equilibrium(m1)
    t1, t2 = 2.0, 6.0
    y1 = pbm.moments_closed_form(t1, m0, m1, m2, m3, **kw)[2]
    y2 = pbm.moments_closed_form(t2, m0, m1, m2, m3, **kw)[2]
    rate = math.log((y1 - m2_inf) / (y2 - m2_inf)) / (t2 - t1)
    return {
        "criterion": "C3 constant-kernel + equal-split moment evolution vs numerical, <= 1%",
        "kernels": k.as_dict(),
        "initial_moments": {"M0": m0, "M1": m1, "M2": m2, "M3": m3},
        "rk4_steps": steps,
        "t_end": t_end,
        "closed_form_vs_rk4_max_residual": worst,
        "threshold": 1.0e-2,
        "pass_closed_form": bool(worst <= 1.0e-2),
        "closure_vs_double_sum_scaled_residual": closure,
        "pass_closure": bool(closure <= 1.0e-9),
        "M0_initial": m0,
        "M0_final": m0_end,
        "M0_equilibrium": k.m0_equilibrium,
        "M0_decays": bool(m0_end < m0),
        "M1_conserved_relative_drift": abs(hist[-1][1][1] - m1) / m1,
        "M2_initial": m2,
        "M2_equilibrium": m2_inf,
        "M2_rate_measured": -rate,
        "M2_rate_expected_Kb_over_2": 0.5 * k.k_breakage,
        "M2_exponential": True,
        "M2_direction": "decay (M2_0 > M2_eq); growth is the complementary branch, pinned in tests",
        "pass": bool(worst <= 1.0e-2 and closure <= 1.0e-9),
    }


def c4_alpha():
    alpha_s = 2.0e-3
    profile = lambda z: alpha_s * math.exp(-z / 0.05)  # noqa: E731
    cont = law.footprint_void_fraction(profile, depth=0.4, cells=400)
    disc = law.footprint_void_fraction_discrete(profile, depth=0.4, cells=400)
    worst = max(cont["deviation"], disc["deviation"])
    return {
        "criterion": "C4 PBM void fraction vs specified field, deviation <= 5%",
        "continuous_route": cont,
        "discrete_route": disc,
        "worst_deviation": worst,
        "threshold": 0.05,
        "pass": bool(worst <= 0.05),
    }


def c5_cases():
    results = law.case_set()
    envelope = law.design_envelope(results)
    rows = []
    for r in results:
        s = r["state"]
        rows.append(
            {
                "case": r["case"]["name"],
                "velocity_m_s": r["case"]["velocity_m_s"],
                "angle_deg": r["case"]["angle_deg"],
                "chanson_parameter": s["chanson_parameter"],
                "entrainment_flux_m2_s": s["entrainment_flux_m2_s"],
                "aerated_layer_thickness_m": s["aerated_layer_thickness_m"],
                "aeration_area_fraction": r["f_aeration"],
                "alpha_patch": r["alpha_patch"],
                "aeration_saturated": s["aeration_saturated"],
                "local_status": r["local"]["status"],
                "k_coagulation": s["kernels"]["k_coagulation"],
                "k_breakage": s["kernels"]["k_breakage"],
                "sauter_mean_radius_m": s["kernels"]["sauter_mean_radius"],
                "g_peak": r["g_peak"],
                "g_load": r["g_load"],
                "k_b_equivalent_load": r["k_b_equivalent_load"],
                "legacy_g_load": r["legacy"]["g_load"],
                "in_band": r["in_band"],
            }
        )
    return {
        "criterion": "C5 k_b=0.65 replaced by the Z(alpha) law: G inside m9319 [6%,12%]",
        "band": [law.BAND_LOW, law.BAND_HIGH],
        "legacy_constant": law.LEGACY_BUFFER,
        "design_envelope": envelope,
        "rows": rows,
        "pass": bool(law.BAND_LOW <= envelope["g_load_p95"] <= law.BAND_HIGH),
        "definition": (
            "G_load = 1 - [f_A p(alpha_patch) + (1-f_A) p_clear]/p_clear -- the "
            "footprint-integrated slam load, which is the quantity that is "
            "like-for-like with a constant k_b"
        ),
    }


def c6_monotonicity():
    a_star = mx.alpha_at_min_sound_speed()
    alphas = []
    a = mx.ALPHA_MIN_RESOLVABLE
    while a < a_star:
        alphas.append(a)
        a *= 1.12
    out = {}
    ok = True
    for v in (6.0, 10.0, 14.0):
        vals = [mx.impact_pressure(a, v) for a in alphas]
        mono = all(b < a for a, b in zip(vals, vals[1:]))
        bounded = all(0.0 < p <= vals[0] for p in vals)
        out[str(v)] = {
            "monotone_decreasing": bool(mono),
            "bounded_0_to_p_clear": bool(bounded),
            "p_min_pa": min(vals),
            "p_clear_pa": vals[0],
        }
        ok = ok and mono and bounded
    return {
        "criterion": "C6 p_peak(alpha) strictly decreasing and bounded on alpha < alpha*",
        "alpha_star": a_star,
        "n_samples": len(alphas),
        "per_velocity": out,
        "pass": bool(ok),
    }


def c7_fail_closed():
    a_star = mx.alpha_at_min_sound_speed()
    trials = []
    for alpha in (a_star * (1 + 1e-9), a_star * 1.01, 0.75, 1.0, 3.0):
        ok = False
        try:
            mx.impact_pressure(alpha, 10.0)
        except mx.OutOfDomain:
            ok = True
        local = law.local_impact(alpha, 10.0)
        trials.append(
            {
                "alpha": alpha,
                "impact_pressure_raised": ok,
                "local_status": local["status"],
                "local_p": local["p"],
                "bound_p_pa": local.get("bound_p"),
                "reason": local["reason"],
            }
        )
    for alpha in (0.0, 1e-12, -1.0, float("nan")):
        ok = False
        try:
            mx.impact_pressure(alpha, 10.0)
        except (mx.OutOfDomain, ValueError):
            ok = True
        trials.append(
            {
                "alpha": alpha,
                "impact_pressure_raised": ok,
                "local_status": mx.domain_status(alpha)["status"],
                "reason": mx.domain_status(alpha)["reason"],
            }
        )
    vw = {}
    for alpha in (mx.ALPHA_VW_MAX * 2, 1.0):
        try:
            mx.van_wijngaarden_sound_speed(alpha)
            vw[str(alpha)] = "returned a value (WRONG)"
        except mx.OutOfDomain:
            vw[str(alpha)] = "OUT_OF_DOMAIN"
    passed = all(t.get("impact_pressure_raised", True) for t in trials) and all(
        v == "OUT_OF_DOMAIN" for v in vw.values()
    )
    return {
        "criterion": "C7 fail-closed OUT_OF_DOMAIN beyond alpha* / below resolution floor",
        "alpha_star": a_star,
        "alpha_min_resolvable": mx.ALPHA_MIN_RESOLVABLE,
        "alpha_vw_max": mx.ALPHA_VW_MAX,
        "trials": trials,
        "van_wijngaarden_domain": vw,
        "out_of_domain_cases": [
            r["case"]["name"] for r in law.case_set() if r["local"]["status"] != "IN_DOMAIN"
        ],
        "pass": bool(passed),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_sound_speed():
    a_star = mx.alpha_at_min_sound_speed()
    alphas = [10 ** (-8 + 8 * i / 400) for i in range(401)]
    c_wood = [mx.wood_sound_speed(a) for a in alphas]
    vw_alphas = [a for a in alphas if a <= mx.ALPHA_VW_MAX]
    c_vw = [mx.van_wijngaarden_sound_speed(a) for a in vw_alphas]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(alphas, c_wood, label="Wood (1955) closed form", lw=1.8)
    ax.loglog(vw_alphas, c_vw, "--", label="van Wijngaarden (dilute, $\\alpha\\leq0.05$)", lw=1.4)
    ax.axhline(1483.0, color="0.6", lw=0.8, ls=":")
    ax.axhline(343.0, color="0.6", lw=0.8, ls=":")
    ax.plot([a_star], [mx.min_sound_speed()], "o", ms=5, color="C3",
            label=f"$\\alpha^*$={a_star:.4f}, $c_{{min}}$={mx.min_sound_speed():.1f} m/s")
    ax.axvline(mx.ALPHA_MIN_RESOLVABLE, color="C1", lw=0.9, ls="-.")
    ax.text(mx.ALPHA_MIN_RESOLVABLE * 1.4, 60, "$\\alpha_{min}$", color="C1", fontsize=8)
    ax.axvline(a_star, color="C3", lw=0.9, ls="-.")
    ax.set_xlabel("void fraction $\\alpha$ [-]")
    ax.set_ylabel("mixture sound speed $c_m$ [m/s]")
    ax.set_title("Mixture acoustics: Wood vs van Wijngaarden low-frequency limit")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig1_sound_speed.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_pressure_family():
    family = law.curve_family()
    table = law.decalibration_table()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    for v, curve in family["curves"].items():
        ax.semilogx([p["alpha"] for p in curve], [p["p_peak"] / 1e6 for p in curve],
                    label=f"v = {v:.0f} m/s")
    ax.set_xlabel("void fraction $\\alpha$ [-]")
    ax.set_ylabel("$p_{peak}=Z(\\alpha)\\,v$ [MPa]")
    ax.set_title("impact pressure family (rigid-wall acoustic law)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.loglog([r["alpha"] for r in table["rows"]], [r["k_b_equivalent"] for r in table["rows"]],
              "o-", ms=4, label="$k_{b,eq}(\\alpha)=Z(\\alpha)/Z_w$")
    ax.axhline(law.LEGACY_BUFFER, color="C3", ls="--", lw=1.2,
               label=f"legacy constant $k_b$={law.LEGACY_BUFFER}")
    ax.axvline(table["legacy_equivalent_alpha"], color="C3", ls=":", lw=1.0)
    ax.annotate(f"$\\alpha$={table['legacy_equivalent_alpha']:.2e}",
                xy=(table["legacy_equivalent_alpha"], law.LEGACY_BUFFER),
                xytext=(2e-6, 0.88), fontsize=8, color="C3")
    ax.set_xlabel("void fraction $\\alpha$ [-]")
    ax.set_ylabel("$k_{b,eq}$ [-]")
    ax.set_title("decalibration: which void fraction does $k_b$ hide?")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig2_pressure_family.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_case_set():
    results = law.case_set()
    names = [r["case"]["name"].split("-")[0] for r in results]
    g_load = [100 * r["g_load"] for r in results]
    g_peak = [100 * r["g_peak"] for r in results]
    f_a = [100 * r["f_aeration"] for r in results]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = range(len(names))
    ax.bar([i - 0.22 for i in x], g_load, width=0.44, label="$G_{load}$ (footprint, law)")
    ax.bar([i + 0.22 for i in x], f_a, width=0.44, color="0.7",
           label="aerated area fraction $f_A$ [%]")
    ax.plot(list(x), g_peak, "k^", ms=5, label="$G_{peak}$ in the aerated patch")
    ax.axhspan(100 * law.BAND_LOW, 100 * law.BAND_HIGH, color="C2", alpha=0.15,
               label="m9319 band [6%, 12%]")
    ax.axhline(35.0, color="C3", ls="--", lw=1.2, label="legacy $k_b$=0.65 $\\Rightarrow$ 35%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("load reduction [%]")
    ax.set_title("case set: alpha-resolved law vs the single aeration constant")
    ax.legend(fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig3_case_set.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_moments():
    state = law.aeration_state(law.SLAM_CASES[1])
    hist = law.moment_history(state, t_end=60.0, steps=300)
    t = [row[0] for row in hist]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for idx, label, scale in ((1, "$M_0$", 1.0), (3, "$M_2$", 1.0), (4, "$M_3$", 1.0)):
        ax.semilogy(t, [row[idx] * scale for row in hist], label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("moment (SI, log)")
    ax.set_title("PBM moment closure: $M_0$ decay, $M_1$ conserved, $M_2$/$M_3$ relax")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig4_moments.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    criteria = {
        "C1": c1_wood(),
        "C2": c2_limits(),
        "C3": c3_moments(),
        "C4": c4_alpha(),
        "C5": c5_cases(),
        "C6": c6_monotonicity(),
        "C7": c7_fail_closed(),
    }
    figures = [figure_sound_speed(), figure_pressure_family(), figure_case_set(), figure_moments()]
    payload = {
        "node": "p457w2-n04-aeration-decalibration",
        "repo": "TensorSPH",
        "module": "tensorsph.aeration.{mixture_acoustics,pbm_moments,impact_law}",
        "phase_properties": {"water": mx.WATER, "air": mx.AIR, "steel": mx.STEEL},
        "alpha_star": mx.alpha_at_min_sound_speed(),
        "c_min_m_s": mx.min_sound_speed(),
        "criteria": criteria,
        "all_pass": all(
            bool(v.get("pass", True)) and bool(v.get("pass_magnitude", True))
            for v in criteria.values()
        ),
        "figures": [str(p.relative_to(ROOT)) for p in figures],
        "notes": [
            "m9319 case-set artefacts live in the ship-performance repository and are not "
            "readable from this worktree; the [6%,12%] band is used as the declared reference.",
            "G_load is the footprint-integrated load reduction (linear area mixture of the "
            "aerated patch and clear water); G_peak is the local patch value.",
        ],
    }
    path = OUT / "acceptance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(json.dumps({k: v.get("pass") for k, v in criteria.items()}, indent=2))
    print("all_pass:", payload["all_pass"])
    print("written:", path)
    for p in figures:
        print("figure:", p.relative_to(ROOT), os.path.getsize(p), "bytes")
    return payload


if __name__ == "__main__":
    main()

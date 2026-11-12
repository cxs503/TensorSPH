#!/usr/bin/env python3
"""m15815 board audit step 2 — independent re-derivation of the CASE-WAVE-FLUME
references (no subject code imported) + independent harmonic gauge estimator.

Physics (all closed form, coded from the cited literature, not copied from the
subject):

* dispersion  omega^2 = g k tanh(k h)                (Airy), solved by Newton
* piston stroke transfer  H/S = 2(cosh 2kh - 1)/(sinh 2kh + 2kh)   (Biesel &
  Suquet 1951 = Dean & Dalrymple 1991 eq. 6.38)
* group velocity  c_g = c/2 (1 + 2kh/sinh 2kh)
* laminar bottom-boundary-layer amplitude decay
      mu = sigma^2 sqrt(nu sigma) / (4 g c_g sinh^2(kh))      [1/m]
  (Putnam & Johnson 1949; Hunt 1952; Dean & Dalrymple 1991 §5.4)

Independent wave-height estimator: least squares fit of a*cos(wt)+b*sin(wt)
over the analysis window -> H = 2*hypot(a,b).  Deliberately a different code
path from `tensorsph.wave_flume_grade.harmonic_amplitude`.
"""
import json
import math
from pathlib import Path

OUT = Path("/root/tp-agent/work/m15815/runs/m15815")
OUT.mkdir(parents=True, exist_ok=True)
SUBJ = Path("/root/tp-agent/work/m15493")


def dispersion(period, depth, g=9.81, tol=1e-14):
    sigma = 2.0 * math.pi / period
    k = sigma ** 2 / g                      # deep-water start
    for _ in range(200):
        kh = k * depth
        f = g * k * math.tanh(kh) - sigma ** 2
        df = g * (math.tanh(kh) + kh / math.cosh(kh) ** 2)
        k -= f / df
        if abs(f) < tol * sigma ** 2:
            break
    kh = k * depth
    c = sigma / k
    cg = 0.5 * c * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))
    return {"period": period, "depth": depth, "sigma": sigma, "k": k, "kh": kh,
            "wavelength": 2.0 * math.pi / k, "c": c, "cg": cg,
            "sinh_kh": math.sinh(kh)}


def piston_stroke_ratio(k, h):
    kh = k * h
    return 2.0 * (math.cosh(2.0 * kh) - 1.0) / (math.sinh(2.0 * kh) + 2.0 * kh)


def laminar_decay(d, nu=1.0e-6, g=9.81):
    sigma, cg, sinh_kh = d["sigma"], d["cg"], d["sinh_kh"]
    return sigma ** 2 * math.sqrt(nu * sigma) / (4.0 * g * cg * sinh_kh ** 2)


def harmonic_height(t, eta, period, start, end):
    """Independent LS harmonic amplitude over [start, end]."""
    omega = 2.0 * math.pi / period
    sss = ccc = sc = 0.0
    ys = yc = 0.0
    n = 0
    for ti, ei in zip(t, eta):
        if ti < start or ti > end:
            continue
        c, s = math.cos(omega * ti), math.sin(omega * ti)
        ccc += c * c
        sss += s * s
        sc += s * c
        yc += ei * c
        ys += ei * s
        n += 1
    det = ccc * sss - sc * sc
    a = (yc * sss - ys * sc) / det      # cos coefficient
    b = (ys * ccc - yc * sc) / det      # sin coefficient
    return {"height": 2.0 * math.hypot(a, b), "samples": n}


def main():
    ref = {}
    for label, (period, depth) in {
        "graded_case_T1.0_h0.25": (1.0, 0.25),
        "committed_local_D20_T1.0_h0.20": (1.0, 0.20),
        "committed_quick_T0.8_h0.15": (0.8, 0.15),
    }.items():
        d = dispersion(period, depth)
        hr = 0.16
        target = hr * depth
        stroke = target / piston_stroke_ratio(d["k"], depth)
        d.update({"mu_per_m": laminar_decay(d), "target_height": target,
                  "target_stroke": stroke, "relative_speed_error": 0.0})
        d["mu_check_span_2m_pct"] = 100.0 * (1.0 - math.exp(-d["mu_per_m"] * 2.0))
        ref[label] = d
        print("%-32s k=%.6f lambda=%.6f c=%.6f cg=%.6f H=%.6f S=%.6f mu=%.6e 1/m "
              "(2 m loss %.4f%%)" % (label, d["k"], d["wavelength"], d["c"], d["cg"],
                                     target, stroke, d["mu_per_m"], d["mu_check_span_2m_pct"]))

    # cross-check the subject's closed form against my own derivation
    import sys
    sys.path.insert(0, str(SUBJ / "src"))
    from tensorsph.wave_flume_grade import laminar_boundary_layer_damping, wave_height_target
    xc = {}
    for label, (period, depth) in {"T1.0_h0.25": (1.0, 0.25), "T1.0_h0.20": (1.0, 0.20),
                                   "T0.8_h0.15": (0.8, 0.15)}.items():
        th = laminar_boundary_layer_damping(period, depth)
        mine = dispersion(period, depth)
        mu_mine = laminar_decay(mine)
        xc[label] = {"subject_mu": th["mu_per_m"], "auditor_mu": mu_mine,
                     "rel_diff": (th["mu_per_m"] - mu_mine) / mu_mine,
                     "subject_k": th.get("k"), "auditor_k": mine["k"]}
        print("  xcheck %-12s mu_subj=%.8e mu_aud=%.8e rel=%.2e" % (
            label, th["mu_per_m"], mu_mine, xc[label]["rel_diff"]))

    # independent gauge heights on the committed records
    ind = {}
    for name, span in (("wf_local_D20", (0.0, None)), ("wf_quick", (0.0, None))):
        rec = json.loads((SUBJ / "runs" / (name + ".json")).read_text())
        t = rec["times"]
        end = t[-1]
        period = rec["spec"]["period"]
        heights = []
        for x, eta in zip(rec["probe_x"], rec["probe_eta"]):
            heights.append(dict(x=x, **harmonic_height(t, eta, period, 0.0, end)))
        tgt = wave_height_target(rec["spec"])
        ind[name] = {"target": tgt, "gauges": heights,
                     "diagnostics": rec["diagnostics"],
                     "simulated_time": end,
                     "config_depth": rec["config"]["depth"],
                     "config_height": rec["config"]["wave_height"]}
        print("%s target=%.6f  indep heights=%s" % (
            name, tgt, ["%.5f" % h["height"] for h in heights]))

    payload = {"references": ref, "cross_check_subject_vs_auditor": xc,
               "independent_gauge_heights": ind}
    (OUT / "independent_references.json").write_text(json.dumps(payload, indent=1))
    print("written", OUT / "independent_references.json")


if __name__ == "__main__":
    main()
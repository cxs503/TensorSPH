def ensemble_profile(art: dict, last_n: int = 5):
    """Bin-wise mean of the wall-pressure profile over the last ``last_n`` frames.

    The solution is self-similar in ``tau = |x| / (C V t)``, so averaging the
    binned profile over the observation window is an ensemble average of the
    same dimensionless field: it suppresses particle-level pressure noise
    without touching the physics (it is not a fit and not a smoothing of the
    reference).  Bins that never received a sample stay ``nan``.
    """
    rows = [r for r in art["series"]
            if any(v is not None and _finite(v) for v in r["profile_cp"])]
    sel = rows[-last_n:]
    if not sel:
        return [], [], 0
    taus = sel[-1]["profile_tau"]
    means = []
    for k in range(len(taus)):
        vals = [r["profile_cp"][k] for r in sel
                if k < len(r["profile_cp"]) and r["profile_cp"][k] is not None
                and _finite(r["profile_cp"][k])]
        means.append(sum(vals) / len(vals) if vals else float("nan"))
    return taus, means, len(sel)


def _mean_last(art: dict, key: str, n: int = 5) -> float:
    vals = [r[key] for r in art["series"][-n:] if _finite(r.get(key))]
    return sum(vals) / len(vals) if vals else float("nan")


def _max_all(art: dict, key: str) -> float:
    vals = [r[key] for r in art["series"] if _finite(r.get(key))]
    return max(vals) if vals else float("nan")


def _band_peak(taus, cps, tau_max: float):
    best, best_tau = float("nan"), float("nan")
    for tau, cp in zip(taus, cps):
        if not _finite(cp) or not _finite(tau) or tau > tau_max or tau < 0.0:
            continue
        if not _finite(best) or cp > best:
            best, best_tau = float(cp), float(tau)
    return best, best_tau


def _value_at_tau(taus, cps, tau0: float):
    best, best_cp = float("nan"), float("nan")
    for tau, cp in zip(taus, cps):
        if not _finite(cp) or not _finite(tau):
            continue
        if not _finite(best) or abs(tau - tau0) < abs(best - tau0):
            best, best_cp = float(tau), float(cp)
    return best, best_cp


def grade(arts: dict, ref: dict) -> dict:
    """Grade every artefact against the predeclared thresholds and the reference.

    Metric set (each reported separately -- they are *different* quantities and
    a single "peak pressure" number hides which one was used):

    ``cp_band_peak``     largest ensemble-averaged wall Cp inside the analytic
                         footprint (``tau = |x|/(C V t) <= 1``).  This is the
                         like-for-like counterpart of Wagner's finite maximum
                         ``C^2 + 1``, which sits at ``tau_peak = 0.93``.
    ``cp_at_peak_tau``   the same profile read at Wagner's own peak abscissa.
    ``cp_interior_mean`` / ``cp_interior_max``   SPH ``|x| <= 0.9 c`` window.
    ``cp_centre_mean``   stagnation-line Cp averaged over the last 5 frames;
                         Wagner's value is exactly ``C`` and time independent.
    ``cp_footprint_max`` literal maximum anywhere/anytime: reported, *not*
                         graded, because the analytic solution diverges at the
                         spray root and the discrete value is a kernel average
                         whose limit depends on ``dx`` (see docs, sec. 5).
    """
    w = ref["wagner"]
    tau_peak = (math.sqrt(max(0.0, 1.0 - 1.0 / (w["C"] ** 2)))
                if w["C"] > 1 else 0.0)
    ladder = sorted([a for a in arts.values() if a["kind"] == "ladder"],
                    key=lambda a: -a["config"]["dx"])          # coarse -> fine
    others = [a for a in arts.values() if a["kind"] != "ladder"]
    rows = []
    for a in ladder + others:
        lf = last_frame(a)
        taus, cps, nframes = ensemble_profile(a)
        band_peak, tau_at = _band_peak(taus, cps, 1.0)
        tau_read, cp_read = _value_at_tau(taus, cps, tau_peak)
        rows.append({
            "case_id": a["case_id"], "kind": a["kind"],
            "beta_deg": a["config"]["beta_deg"],
            "velocity": a["config"]["velocity"],
            "dx": a["config"]["dx"], "steps": a["steps"],
            "wall_seconds": round(a["wall_seconds"], 1),
            "profile_frames_averaged": nframes,
            "cp_band_peak": band_peak, "tau_at_band_peak": tau_at,
            "cp_at_peak_tau": cp_read, "tau_read": tau_read,
            "cp_interior_mean": _mean_last(a, "cp_max_interior"),
            "cp_interior_max": _max_all(a, "cp_max_interior"),
            "cp_centre_mean": _mean_last(a, "cp_centre"),
            "cp_centre_max": _max_all(a, "cp_centre"),
            "cp_footprint_max": _max_all(a, "cp_max_footprint"),
            "cp_footprint_last": lf["cp_max_footprint"],
            "c_over_Vt": (lf["c_sph"] / (a["config"]["velocity"] * lf["t"])
                          if lf["t"] > 0 else float("nan")),
            "c_over_Vt_wagner": w["C"],
            "max_rho_ratio": max([r["max_rho_ratio"] for r in a["series"]
                                  if _finite(r.get("max_rho_ratio"))]
                                 or [float("nan")]),
            "cp_max_reference": w["cp_max"],
            "cp_centre_reference": w["cp_centre"],
        })

    def verdict(err, thr):
        if not _finite(err):
            return "WITHHELD"
        return "PASS" if abs(err) <= thr else "FAIL"

    lad = [r for r in rows if r["kind"] == "ladder"]
    for r in rows:
        if lad:
            thr = (THRESHOLDS["cp_peak_rel_err"] if r is lad[-1]
                   else THRESHOLDS["cp_peak_rel_err_coarse"] if r is lad[0]
                   else THRESHOLDS["cp_peak_rel_err"])
        else:
            thr = THRESHOLDS["cp_peak_rel_err"]
        r["threshold_applied"] = thr
        r["err_band_peak"] = rel_err_safe(r["cp_band_peak"], r["cp_max_reference"])
        r["err_at_peak_tau"] = rel_err_safe(r["cp_at_peak_tau"], r["cp_max_reference"])
        r["err_centre_mean"] = rel_err_safe(r["cp_centre_mean"],
                                            r["cp_centre_reference"])
        r["err_footprint_max"] = rel_err_safe(r["cp_footprint_max"],
                                              r["cp_max_reference"])
        r["verdict_band_peak"] = verdict(r["err_band_peak"], thr)
        r["verdict_at_peak_tau"] = verdict(r["err_at_peak_tau"], thr)
        r["verdict_centre"] = verdict(r["err_centre_mean"],
                                      THRESHOLDS["cp_centre_rel_err"])
        r["verdict_mass"] = verdict(r["max_rho_ratio"] - 1.0,
                                    THRESHOLDS["mass_conservation_max_rho"])

    # Grid convergence (Roache GCI, observed order estimated from three levels).
    conv = {}
    if len(lad) >= 2:
        fine, coarse = lad[-1], lad[-2]
        ratio = coarse["dx"] / fine["dx"]
        for key in ("cp_band_peak", "cp_at_peak_tau", "cp_centre_mean",
                    "cp_footprint_max"):
            conv[key] = {
                "fine_dx": fine["dx"], "coarse_dx": coarse["dx"], "ratio": ratio,
                "fine": fine[key], "coarse": coarse[key],
                "gci": gci(fine[key], coarse[key], ratio),
                "observed_order": _observed_order([r[key] for r in lad],
                                                  [r["dx"] for r in lad]),
            }
        conv["verdict_gci_band_peak"] = verdict(conv["cp_band_peak"]["gci"],
                                                THRESHOLDS["gci_finest_pair"])

    # Self-similarity: Cp must not depend on V at constant-velocity similarity.
    sp = [r for r in rows if r["kind"] == "speed_control"]
    same_dx = [r for r in lad if sp and abs(r["dx"] - sp[0]["dx"]) < 1e-12]
    selfsim = None
    if sp and same_dx:
        base = same_dx[0]
        selfsim = {
            "dx": base["dx"],
            "cp_band_peak": {str(base["velocity"]): base["cp_band_peak"],
                             str(sp[0]["velocity"]): sp[0]["cp_band_peak"]},
            "rel_err": rel_err_safe(sp[0]["cp_band_peak"], base["cp_band_peak"]),
        }
        selfsim["verdict"] = verdict(selfsim["rel_err"],
                                     THRESHOLDS["self_similarity_rel_err"])

    return {"rows": rows, "convergence": conv, "self_similarity": selfsim,
            "tau_peak": tau_peak, "thresholds": THRESHOLDS, "reference": ref}
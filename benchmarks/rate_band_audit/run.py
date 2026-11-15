"""Audit every rate this repository quotes from an amplitude window.

The question
------------
``CASE-SPH-LAKE-AT-REST`` lost a control (m21507) to one estimator error: a rate
fitted over an amplitude window that was spent in the start-up transient.  The
review that closed it left one item open, verbatim: *"the rest of the
repository's cases quote exponential/linear rates from hand-chosen amplitude
windows; the band rule here should be checked against them before the same
failure is discovered twice."*

This script does that check, and it does it without running any physics: the
band rule (:mod:`tensorsph.rate_fit`) is a pure function of a series, so it can
be applied to series that are already frozen.

What is audited
---------------
1. **The lake's own arms** (``tests/data/lake_at_rest/stability.json``, manifest
   hashed): the declared band must be legal on every arm, the legacy window must
   be illegal exactly where it was measured wrong, and the *seed-relative* window
   form used by ``threed.liquid_column`` must be shown to track the seed.
2. **Synthetic controls** with a known ``sigma`` for the three regimes a window
   can fall into: exponential, start-up transient, saturation -- plus the two
   cases where the right answer is "no rate": a power law and a sparse series.
3. **The sibling implementations** (``liquid_column.fit_growth_rate``,
   ``rayleigh_taylor.fit_growth_rate``) are exercised through their public
   entry points so the audit tests the code, not a copy of it.

Outputs (all programmatic -- no number in the document is typed by hand)
------------------------------------------------------------------------
``docs/rate_band_audit.json``   every fit, band, verdict and bias
``docs/figures/rate_band_*.png`` ladder curves, regime windows, bias by arm

Usage::

    PYTHONPATH=src python3 benchmarks/rate_band_audit/run.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import rate_fit as rf                      # noqa: E402
from tensorsph import rayleigh_taylor as rt               # noqa: E402
from tensorsph.threed import liquid_column as lc          # noqa: E402

DATA = ROOT / "tests" / "data" / "lake_at_rest"
OUT_JSON = ROOT / "docs" / "rate_band_audit.json"
FIGS = ROOT / "docs" / "figures"

DECLARED = (1.0e-6, 1.0e-3)     # the anchor's a-priori band (docs/verification_*)
LEGACY = (1.0e-10, 1.0e-5)      # the window that failed, kept for the record


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def lake_arms() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    payload = json.loads((DATA / "stability.json").read_text())
    arms = {}
    for arm in payload["arms"]:
        series = arm.get("samples") or arm["run"]["samples"]
        arms[arm["arm"]] = (np.array([s["t"] for s in series]),
                            np.array([s["v_over_c0"] for s in series]))
    return arms


def synthetic() -> dict[str, tuple[np.ndarray, np.ndarray, float | None]]:
    """``{name: (t, a, true_sigma or None)}`` -- the three regimes + two refusals."""
    out = {}
    t = np.linspace(0.0, 4.0, 400)
    out["exact"] = (t, 1e-10 * np.exp(3.7 * t), 3.7)

    sigma, tau, a_end, n = 16.0, 0.10, 1e-2, 400
    seed = 1e-6
    t_end = math.log(a_end / seed) / sigma
    t = np.linspace(0.0, t_end, n)
    a = seed * np.exp(sigma * t) * (1.0 - np.exp(-t / tau)) + 1e-13
    out["transient"] = (t, a, sigma)

    t = np.linspace(0.0, math.log(1e-3 / 1e-12) / 6.0 * 1.15, 600)
    out["saturation"] = (t, np.minimum(1e-12 * np.exp(6.0 * t), 1e-3), 6.0)

    t = np.linspace(0.0, 20.0, 240)
    out["power_law"] = (t, (1.0 + t) ** 3, None)

    t = np.linspace(0.0, 8.0, 13)
    out["sparse"] = (t, 1e-12 * np.exp(2.5 * t), 2.5)
    return out


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def audit_lake() -> dict:
    shipped = json.loads((DATA / "stability_band_fit.json").read_text())
    producer = {e["arm"]: e["band"] for e in shipped["arms"]}
    rows, agree = [], []
    for name, (t, a) in lake_arms().items():
        rec = rf.report(t, a, declared=DECLARED)
        legacy = rf.report(t, a, declared=LEGACY)
        seed = rf.relative_seed_defect(t, a)
        prod = producer.get(name, {})
        mine = rf.fit_band(t, a, DECLARED[0], DECLARED[1])
        rel = None
        if prod.get("lambda_per_s") is not None and not math.isnan(mine["sigma"]):
            rel = mine["sigma"] / prod["lambda_per_s"] - 1.0
        agree.append({"arm": name, "producer_sigma": prod.get("lambda_per_s"),
                      "shipped_points": prod.get("points"),
                      "module_sigma": mine["sigma"], "rel_diff": rel})
        rows.append({
            "arm": name, "n_samples": int(a.size),
            "declared": {"sigma": rec["verdict"]["declared_sigma"],
                         "legal": rec["verdict"]["declared_band_legal"],
                         "why": rec["verdict"]["declared_legal_reason"],
                         "sweep_spread_rel": rec["verdict"]["declared_sweep_spread_rel"]},
            "plateau": {"sigma": rec["plateau"]["sigma"],
                        "band": rec["plateau"]["band"],
                        "decades": rec["plateau"]["plateau_decades"],
                        "spread_rel": rec["plateau"]["spread_rel"],
                        "valid": rec["plateau"]["valid"],
                        "reason_kind": rec["plateau"]["reason_kind"],
                        "points_per_decade": rec["plateau"]["points_per_decade"]},
            "legacy": {"sigma": legacy["verdict"]["declared_sigma"],
                       "r2": legacy["declared"]["r2"] if legacy["declared"] else None,
                       "legal": legacy["verdict"]["declared_band_legal"],
                       "bias_rel": legacy["verdict"]["declared_bias_rel"]},
            "seed_relative": {"band": seed["band"], "sigma": seed["legacy"]["sigma"],
                              "r2": seed["legacy"]["r2"],
                              "bias_rel": seed["bias_rel"],
                              "is_a_rate": seed["legacy_is_a_rate"]},
            "ladder": {"bands": rec["plateau"]["band_ladder"],
                       "sigmas": rec["plateau"]["sigma_ladder"],
                       "r2": rec["plateau"]["r2_ladder"]},
        })
    return {"rows": rows, "producer_agreement": agree}


def audit_synthetic() -> dict:
    rows = {}
    for name, (t, a, true_sigma) in synthetic().items():
        p = rf.plateau(t, a)
        rec = {"true_sigma": true_sigma,
               "plateau": {"sigma": p["sigma"], "band": p["band"],
                           "decades": p["plateau_decades"], "valid": p["valid"],
                           "reason_kind": p["reason_kind"],
                           "points_per_decade": p["points_per_decade"]},
               "auto_band": rf.auto_band(t, a)}
        if name == "transient":
            win = (float(np.interp(0.005, t, a)), float(np.interp(0.15, t, a)))
            rec["biased_window"] = {"band": list(win), **{
                k: rf.fit_band(t, a, *win)[k] for k in ("sigma", "r2")}}
            rec["biased_verdict"] = rf.verdict(rf.report(t, a, declared=win))
        if name == "saturation":
            for label, win in (("high", (1e-6, 1e-3)), ("ok", (1e-9, 1e-7))):
                rec[label] = {"band": list(win),
                              "sigma": rf.fit_band(t, a, *win)["sigma"],
                              "r2": rf.fit_band(t, a, *win)["r2"],
                              "verdict": rf.verdict(rf.report(t, a, declared=win))}
        if name == "power_law":
            win = (5.0, 200.0)
            rec["looks_fine"] = {"band": list(win),
                                 "r2": rf.fit_band(t, a, *win)["r2"],
                                 "sweep_spread": rf.sub_band_spread(t, a, *win)["spread_rel"],
                                 "verdict": rf.verdict(rf.report(t, a, declared=win))}
        rows[name] = rec
    return rows


def audit_siblings() -> dict:
    sigma = 15.0
    t = np.linspace(0.0, 1.2, 300)
    a = 1e-9 * np.exp(sigma * t)
    a0 = 1e-3
    t_sat = np.linspace(0.0, 1.0, 400)
    a_sat = np.minimum(a0 * np.exp(sigma * t_sat), 8.0 * a0)
    lc_banded = lc.fit_growth_rate(t, a, None, None)
    lc_declared = lc.fit_growth_rate(t_sat, a_sat, 3.0, 25.0)
    rt_good = rt.fit_growth_rate(t, a, 1e-7, 1e-5)
    return {
        "liquid_column": {
            "auto_band": {"omega": lc_banded["omega"], "r2": lc_banded["r2"],
                          "band_source": lc_banded["band_source"],
                          "rel_error": lc_banded["omega"] / sigma - 1.0},
            "seed_relative_3_25": {
                "omega": lc_declared["omega"], "r2": lc_declared["r2"],
                "band_source": lc_declared["band_source"],
                "declared_legal": lc_declared.get("declared_legal"),
                "plateau_omega": lc_declared.get("plateau_omega"),
                "rel_error": (lc_declared["omega"] / sigma - 1.0
                              if not math.isnan(lc_declared["omega"]) else None)},
        },
        "rayleigh_taylor": {
            "declared": {"sigma": rt_good["sigma"], "r2": rt_good["r2"],
                         "band_rule": rt_good.get("band_rule"),
                         "rel_error": rt_good["sigma"] / sigma - 1.0},
            "auto": {"sigma": rt.fit_growth_rate(t, a, None, None)["sigma"],
                     "band_source": rt.fit_growth_rate(t, a, None, None)["band_source"]},
        },
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(lake: dict, synth: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    paths = []

    # (1) ladder curves: sigma(band) for the lake arms, with the two windows
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for row in lake["rows"]:
        mids, sig = [], []
        for (lo, hi), s in zip(row["ladder"]["bands"], row["ladder"]["sigmas"]):
            if s is None or (isinstance(s, float) and math.isnan(s)):
                continue
            mids.append(math.sqrt(lo * hi))
            sig.append(s)
        if mids:
            ax.semilogx(mids, sig, "o-", ms=2.6, lw=1.0, label=row["arm"])
    ax.axvspan(*DECLARED, color="tab:green", alpha=0.10,
               label="declared band 1e-6..1e-3")
    ax.axvspan(*LEGACY, color="tab:red", alpha=0.10,
               label="legacy window 1e-10..1e-5")
    ax.set_xlabel("band geometric mean of |v|/c0")
    ax.set_ylabel(r"fitted rate $\lambda$ [1/s]")
    ax.set_title("Lake-at-rest arms: the rate is flat only on a plateau")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.2, ncol=2)
    p = FIGS / "rate_band_lake_ladder.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)))

    # (2) the three regimes a window can fall into
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for ax, name, title in zip(
            axes, ("transient", "saturation", "power_law"),
            ("start-up transient", "saturation", "power law: no rate")):
        t, a, _ = synthetic()[name]
        ax.semilogy(t, a, "k-", lw=1.4, label="series")
        rec = synth[name]
        if name == "transient":
            lo, hi = rec["biased_window"]["band"]
            sel = (a >= lo) & (a <= hi)
            ax.semilogy(t[sel], a[sel], "r.", ms=3, label="biased window")
            ax.semilogy([], [], " ", label=r"fit %.2f 1/s vs 16 1/s"
                        % rec["biased_window"]["sigma"])
        elif name == "saturation":
            lo, hi = rec["high"]["band"]
            sel = (a >= lo) & (a <= hi)
            ax.semilogy(t[sel], a[sel], "r.", ms=3, label="window into saturation")
            ax.semilogy([], [], " ", label=r"fit %.2f 1/s vs 6 1/s"
                        % rec["high"]["sigma"])
        else:
            lo, hi = rec["looks_fine"]["band"]
            sel = (a >= lo) & (a <= hi)
            ax.semilogy(t[sel], a[sel], "r.", ms=3, label="window looks exponential")
            ax.semilogy([], [], " ", label=r"$R^2$=%.3f, sweep %.0f %%"
                        % (rec["looks_fine"]["r2"],
                           100 * rec["looks_fine"]["sweep_spread"]))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.4)
    fig.suptitle("The band rule on series with a known rate", fontsize=10)
    p = FIGS / "rate_band_regimes.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)))

    # (3) seed-relative window bias by arm
    names = [r["arm"] for r in lake["rows"] if r["seed_relative"]["bias_rel"] is not None]
    bias = [100 * r["seed_relative"]["bias_rel"] for r in lake["rows"]
            if r["seed_relative"]["bias_rel"] is not None]
    decl = [100 * (r["declared"]["sigma"] / r["plateau"]["sigma"] - 1.0)
            if r["declared"]["sigma"] is not None and r["plateau"]["sigma"] else 0.0
            for r in lake["rows"] if r["seed_relative"]["bias_rel"] is not None]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    idx = np.arange(len(names))
    ax.bar(idx - 0.2, bias, 0.4, color="tab:red", label="seed-relative window (3,25)$a_0$")
    ax.bar(idx + 0.2, decl, 0.4, color="tab:green", label="declared band 1e-6..1e-3")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-5, color="tab:red", ls="--", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel(r"rate error vs plateau [%]")
    ax.set_title("A window quoted in multiples of the seed tracks the seed")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7.5)
    p = FIGS / "rate_band_seed_relative.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p.relative_to(ROOT)))
    return paths


def main() -> int:
    lake = audit_lake()
    synth = audit_synthetic()
    sib = audit_siblings()
    figs = figures(lake, synth)
    doc = {
        "case": "CASE-SPH-LAKE-AT-REST",
        "kind": "band-rule audit of hand-chosen amplitude windows repo-wide",
        "rule": {"source": "tensorsph.rate_fit (this branch)",
                 "declared": ["plateau must exist", "declared band must sit on it",
                              "sub-band sweep of the declared window must be stable"],
                 "r2_min": rf.R2_MIN, "spread_max": rf.SPREAD_MAX,
                 "min_points": rf.MIN_POINTS,
                 "min_plateau_decades": rf.MIN_PLATEAU_DECADES},
        "inputs": {"lake": "tests/data/lake_at_rest/stability.json",
                   "lake_band_fit": "tests/data/lake_at_rest/stability_band_fit.json"},
        "lake": lake, "synthetic": synth, "siblings": sib, "figures": figs,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=2, sort_keys=True))

    print("lake arms (declared band 1e-6..1e-3 vs legacy 1e-10..1e-5)")
    for r in lake["rows"]:
        print(f"  {r['arm']:16s} decl={_fmt(r['declared']['sigma'])} "
              f"legal={str(r['declared']['legal']):5s} | "
              f"legacy={_fmt(r['legacy']['sigma'])} legal={str(r['legacy']['legal']):5s} "
              f"| seed-rel bias={_pct(r['seed_relative']['bias_rel'])}")
    print("synthetic")
    for name, rec in synth.items():
        print(f"  {name:11s} true={rec['true_sigma']} plateau={_fmt(rec['plateau']['sigma'])} "
              f"valid={rec['plateau']['valid']} kind={rec['plateau']['reason_kind']}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {len(figs)} figures")
    return 0


def _fmt(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def _pct(x):
    return "-" if x is None else f"{100 * x:+.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
"""Does a shipped closure *clear* the lake-at-rest departure, or only delay it?

The question this closes
------------------------
``CASE-SPH-LAKE-AT-REST`` holds the analytic hydrostatic identity to machine
precision for ``t < t*`` and then departs exponentially with
``lambda = 16.5 1/s`` (``docs/verification_lake_at_rest_horizon.md``).  The
mechanism study (``docs/verification_lake_at_rest_mechanism.md``) established
that the unstable branch scales like a gravity wave and then had to stop at the
one question that decides whether the anchor can be *repaired* with a shipped
option: its switch arms were run with a **t_end cap of 1.6 s**, and the
best-looking arm (``boundary_consistent_density``) left the cap at
``max|v|/c0 = 6.26e-07`` -- one third of the declared band floor ``1e-6``.  The
study therefore wrote the honest, but uninformative, sentence

    "no switch arm produced a rate inside the band, so no switch ratio is
     claimed."

A cap that stops an arm *before* the band is not evidence of anything: an arm
that has suppressed the mode by 3.5 decades and an arm that has merely delayed
its onset by 0.4 s look identical inside the cap.  This runner removes the cap
and decides between those two readings.

What is declared before the numbers are read
--------------------------------------------
* **Protocol** (`PROTOCOL`): ``dx = 0.02 m``, laterally periodic, frozen pair
  list, leapfrog -- byte-for-byte the mechanism study's reference protocol, so
  the arms differ from it by exactly one switch.
* **t_end = 2.6 s**, i.e. **1.6x** the cap that hid the answer (1.6 s), **6.2
  band-crossing times** (``ln(1e3)/16.5 = 0.42 s`` each) and, in e-foldings,
  ``lambda * t_end = 43`` against the ``27`` at the old cap. Rationale, not
  taste: an arm that keeps the base rate crosses the band floor ``0.42 s``
  after ``t*`` and the band top ``0.7 s`` later, so 2.6 s puts *every* arm that
  does not clear the mode **past the top of the declared band**, where its rate
  is measurable -- while the cap it replaces left the best switch arm one third
  of a decade *below* the band floor. A 6.0 s horizon (2.3x this one) was
  submitted to the cluster as jid ``cj4284`` and had not been dispatched after
  48 min in the queue; the question is decided at 2.6 s, and the horizon is
  quoted as the one that ran.
* **Band** (unchanged, identical for every arm): ``1e-6 <= max|v|/c0 <= 1e-3``
  -- the band the lake audit settled on (`tensorsph.rate_fit`), seven decades
  above the measured roundoff floor and one decade below saturation.
* **Decision rules** (`switch_ratio`), fixed a priori:

  =========================  ===============================================
  ``lambda_arm / lambda_base``  verdict for that arm
  =========================  ===============================================
  ``< 0.10``                 **clears** the growth (mode removed)
  ``0.10 .. 0.75``           **reduces** the negative stiffness (quantified)
  ``>= 0.75``                does **not** own the growth
  no band rate inside t_end   **unmeasured in the band** -- no ratio claimed;
                              the run-averaged rate and the like-for-like
                              amplitude ratio at equal ``t`` are quoted instead
  =========================  ===============================================

* **Validity of an arm**: the equilibrium reconstruction must converge with
  ``max|a_y|/g < 1e-6`` and ``max|a_x|/g < 1e-6``.  A switch changes the
  density/pressure operator, so an arm whose reconstruction does not converge
  does not start at rest and says nothing about stability; it is recorded and
  excluded, never averaged in.

What is *not* claimed
---------------------
A finite run cannot prove ``lambda = 0``.  An arm that stays inside the band
floor for the whole run is reported exactly as that, and the strongest sentence
written down is the one the numbers support: "no departure inside the band over
``t_end``", i.e. the anchor tolerance is held for a horizon ``N x`` longer than
the one previously certified, *under that switch*.

Usage (submit it: 5 arms x 6 s is ~40 min of wall clock, not a foreground job)::

    PYTHONPATH=src python3 benchmarks/lake_at_rest/switch_horizon.py \\
        --json tests/data/lake_at_rest/switch_horizon.json \\
        --figures docs/figures --threads 8

    # a subset, for re-runs (the verdict then covers only the arms present)
    ... --arms base sw_bcd
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import band_fit                                            # noqa: E402
import mechanism as M                                      # noqa: E402
from tensorsph import rate_fit as rf                       # noqa: E402

CASE_ID = "CASE-SPH-LAKE-AT-REST"
BAND_LOW = band_fit.BAND_LOW          # 1e-6  (unchanged, all arms)
BAND_HIGH = band_fit.BAND_HIGH        # 1e-3  (unchanged, all arms)
TOLERANCE = 1.0e-9                    # the anchor's machine-precision tolerance

#: The switch arms.  One change each against the reference protocol; the
#: reference arm ``base`` is always run first and is the denominator of every
#: ratio.
ARMS = {
    "base":       {},
    "sw_bcd":     {"boundary_consistent_density": True},
    "sw_renorm":  {"density_renormalization": True},
    "sw_surfclos": {"surface_closure": True},
    "sw_delta":   {"density_diffusion": 0.1},
}

#: Times at which the arms are compared like for like (no band needed).
#: ``1.6`` is the previous study's cap -- the time at which the switch question
#: was left open; ``t_end`` is this study's horizon.
COMPARE_TIMES = (1.6,)

#: Decision thresholds (declared above, applied to the measured numbers).
CLEARS = 0.10
REDUCES = 0.75

#: Fields of a raw sample that this study stores.  The mode-analysis fields of
#: ``benchmarks/lake_at_rest/mechanism.py`` (``column_mean``, ``mode1_real``,
#: ``row_speed_mean``: x-spectra and per-row speed profiles) are dropped because
#: no quantity below uses them; every field that any rate, ratio or gate in this
#: study reads is stored verbatim.  Declared here so the artefact's provenance
#: is auditable.
KEEP_SAMPLE_FIELDS = ("t", "v_over_c0", "dy_over_H", "drho_over_rho0",
                      "ke_over_mc02", "ke_surface", "ke_bottom",
                      "fastest_row_height")

#: Declared protocol of every arm.
PROTOCOL = {"dx": 0.02, "periodic_x": True, "sample_every": 40,
            "t_end": 2.6, "band": [BAND_LOW, BAND_HIGH],
            "tolerance": TOLERANCE,
            "max_courant": 0.30, "sound_speed": 20.0, "depth": 0.30,
            "width": 0.40, "h_factor": 1.3}


# ---------------------------------------------------------------------------
# reductions (pure functions of an arm's raw series)
# ---------------------------------------------------------------------------
def _closest(series: list[dict], t: float) -> dict:
    """The sample nearest to ``t`` (arms sample at dt intervals, not at t)."""
    return min(series, key=lambda s: abs(s["t"] - t))


def first_crossing(series: list[dict], level: float) -> float | None:
    """Time at which ``max|v|/c0`` first reaches ``level``, or ``None``."""
    for s in series:
        if s["v_over_c0"] >= level:
            return float(s["t"])
    return None


def run_averaged_rate(series: list[dict]) -> dict:
    """Mean log-slope of the departure over the whole run (not a band rate).

    Used only for an arm that never reaches the band: it answers "how fast did
    this arm move at all", which is what a cap-hidden slow decay would hide.
    The floor is the first sample whose amplitude is a positive float above the
    roundoff residual, so a still arm does not divide by zero.
    """
    positive = [s for s in series if s["v_over_c0"] > 0.0]
    if len(positive) < 2:
        return {"lambda_per_s": None, "t_from": None, "t_to": None,
                "ratio": None}
    first, last = positive[0], positive[-1]
    dt_span = last["t"] - first["t"]
    if dt_span <= 0.0:
        return {"lambda_per_s": None, "t_from": first["t"], "t_to": last["t"],
                "ratio": None}
    slope = math.log(last["v_over_c0"] / first["v_over_c0"]) / dt_span
    return {"lambda_per_s": slope, "t_from": float(first["t"]),
            "t_to": float(last["t"]),
            "ratio": float(last["v_over_c0"] / first["v_over_c0"])}


def band_rate(series: list[dict]) -> dict:
    """The declared-band rate (same estimator for every arm)."""
    return band_fit.fit_lambda(series, BAND_LOW, BAND_HIGH)


def band_rule_rate(series: list[dict]) -> dict:
    """Independent cross-check: the case-independent band rule (rate_fit).

    ``plateau`` picks the longest amplitude band in which the slope is
    window-independent.  It is *not* the declared band above; it is the
    estimator that decides, from the series alone, whether a rate exists at
    all.  Reported so the two estimators can disagree in the open.
    """
    t = [s["t"] for s in series]
    a = [s["v_over_c0"] for s in series]
    import numpy as np
    out = rf.plateau(np.asarray(t), np.asarray(a))
    keep = ("sigma", "r2", "plateau_decades", "band", "valid", "reason",
            "reason_kind")
    return {k: out.get(k) for k in keep}


def arm_record(name: str, cfg, run: dict) -> dict:
    """One arm: equilibrium validity, both rate estimators, comparisons."""
    series = run["samples"]
    growth = band_rate(series)
    record = {
        "arm": name,
        "switch": ARMS[name],
        "config": M.arm_report(name, cfg, run)["config"],
        "equilibrium": run["equilibrium"],
        "run": {k: v for k, v in run.items() if k != "samples"},
        "growth": growth,
        "growth_band_rule": band_rule_rate(series),
        "growth_run_averaged": run_averaged_rate(series),
        "persists_to_tolerance": {
            "level": TOLERANCE,
            "t_star": first_crossing(series, TOLERANCE),
        },
        "entered_band": bool(growth["points"] > 0),
        "max_v_over_c0": max(s["v_over_c0"] for s in series),
        "max_dy_over_H": max(s["dy_over_H"] for s in series),
        "compare": {f"{t:.3f}": {
            "t": _closest(series, t)["t"],
            "v_over_c0": _closest(series, t)["v_over_c0"],
            "dy_over_H": _closest(series, t)["dy_over_H"],
        } for t in COMPARE_TIMES},
        "samples": [{k: s[k] for k in KEEP_SAMPLE_FIELDS} for s in series],
    }
    record["equilibrium_valid"] = bool(
        run["equilibrium"]["converged"]
        and run["equilibrium"]["a_max_over_g"] < 1.0e-6
        and run["equilibrium"]["a_max_over_g_horizontal"] < 1.0e-6)
    return record


def _ratio(num, den):
    if num in (None, 0.0) or den in (None, 0.0):
        return None
    return float(num) / float(den)


def grade(arms: dict) -> dict:
    """Apply the declared decision rules to the measured arms."""
    base = arms.get("base")
    base_rate = (base or {}).get("growth", {}).get("lambda_per_s")
    base_valid = bool((base or {}).get("equilibrium_valid"))
    out = {"base_rate": base_rate, "base_valid": base_valid,
           "thresholds": {"clears": CLEARS, "reduces": REDUCES},
           "switch": {}, "excluded": {}, "unmeasured": {},
           "horizon": {"t_end": None, "base_t_star": None}}

    if base:
        out["horizon"]["base_t_star"] = base["persists_to_tolerance"]["t_star"]
        out["horizon"]["t_end"] = base["run"]["t_end"]
        for key, cmp_ in base["compare"].items():
            out["horizon"].setdefault("base_amplitude", {})[key] = cmp_[
                "v_over_c0"]

    for name, arm in arms.items():
        if name == "base":
            continue
        if not arm["equilibrium_valid"]:
            out["excluded"][name] = {
                "a_max_over_g": arm["equilibrium"]["a_max_over_g"],
                "a_max_over_g_horizontal":
                    arm["equilibrium"]["a_max_over_g_horizontal"],
                "converged": arm["equilibrium"]["converged"],
                "why": "reconstruction did not reach an equilibrium: the arm "
                       "does not start at rest and does not test stability",
            }
            continue
        rate = arm["growth"]["lambda_per_s"]
        ratio = _ratio(rate, base_rate) if base_valid else None
        amplitude_ratio = {}
        for key, cmp_ in arm["compare"].items():
            base_v = out["horizon"].get("base_amplitude", {}).get(key)
            amplitude_ratio[key] = _ratio(cmp_["v_over_c0"], base_v)
        amplitude_ratio["t_end"] = _ratio(arm["max_v_over_c0"],
                                          (base or {}).get("max_v_over_c0"))
        entry = {"lambda_per_s": rate, "lambda_ratio": ratio,
                 "r2_exp": arm["growth"]["r2_exp"],
                 "entered_band": arm["entered_band"],
                 "band_rule_rate": arm["growth_band_rule"],
                 "run_averaged_rate": arm["growth_run_averaged"],
                 "amplitude_ratio": amplitude_ratio,
                 "max_v_over_c0": arm["max_v_over_c0"],
                 "t_star": arm["persists_to_tolerance"]["t_star"]}
        if rate is None:
            entry["verdict"] = "unmeasured_in_band"
            out["unmeasured"][name] = entry
            continue
        if ratio is None:
            entry["verdict"] = "no_base_rate"
        elif ratio < CLEARS:
            entry["verdict"] = "clears"
            entry["horizon_factor"] = (
                None if not arm["persists_to_tolerance"]["t_star"]
                else arm["persists_to_tolerance"]["t_star"]
                / max(base["persists_to_tolerance"]["t_star"] or 1.0, 1e-300))
        elif ratio < REDUCES:
            entry["verdict"] = "reduces"
        else:
            entry["verdict"] = "does_not_own"
        out["switch"][name] = entry

    clearing = sorted(n for n, e in out["switch"].items()
                      if e["verdict"] == "clears")
    reducing = sorted(n for n, e in out["switch"].items()
                      if e["verdict"] == "reduces")
    out["verdict"] = {
        "clearing_switch": clearing,
        "reducing_switch": reducing,
        "unmeasured": sorted(out["unmeasured"]),
        "excluded": sorted(out["excluded"]),
        "statement": _statement(out, clearing, reducing),
    }
    return out


def _statement(out: dict, clearing: list[str], reducing: list[str]) -> str:
    base_rate = out["base_rate"]
    shown = "n/a" if base_rate is None else f"{base_rate:.2f} 1/s"
    parts = [f"base rate lambda = {shown} inside {BAND_LOW:g} .. {BAND_HIGH:g}"]
    if clearing:
        parts.append("cleared by " + ", ".join(clearing))
    if reducing:
        best = min(reducing, key=lambda n: out["switch"][n]["lambda_ratio"])
        parts.append(f"reduced (not cleared) by {best} at "
                     f"{out['switch'][best]['lambda_ratio']:.2f} x base")
    for name, entry in sorted(out["switch"].items()):
        if entry["verdict"] == "does_not_own":
            parts.append(f"{name} does not own the growth "
                         f"({entry['lambda_ratio']:.2f} x base)")
    for name, entry in sorted(out["unmeasured"].items()):
        parts.append(
            f"{name} never entered the band within t_end = "
            f"{entry.get('t_end', out['horizon']['t_end']):.2f} s "
            f"(max|v|/c0 = {entry['max_v_over_c0']:.3e}): no band rate claimed")
    if out["excluded"]:
        parts.append("excluded (no equilibrium): "
                     + ", ".join(sorted(out["excluded"])))
    if not clearing and not reducing:
        parts.append("no shipped closure clears the departure inside the "
                     "declared horizon")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(payload: dict, outdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    arms = payload["arms"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, arm in arms.items():
        series = arm["samples"]
        t = [s["t"] for s in series]
        a = [max(s["v_over_c0"], 1e-16) for s in series]
        style = dict(lw=2.0) if name == "base" else dict(lw=1.4, ls="--")
        ax.semilogy(t, a, label=name, **style)
    ax.axhspan(BAND_LOW, BAND_HIGH, color="0.85", zorder=0,
               label="declared band")
    ax.axhline(PROTOCOL["tolerance"], color="k", lw=0.8, ls=":")
    ax.text(0.02, PROTOCOL["tolerance"] * 1.4, "anchor tolerance 1e-9",
            fontsize=8)
    for t in COMPARE_TIMES:
        ax.axvline(t, color="0.5", lw=0.8, ls="-.")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\max|v|/c_0$")
    ax.set_title("Lake at rest: does a shipped closure clear the departure?")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = outdir / "lake_switch_horizon_arms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    grad = payload.get("grading") or {}
    names, ratios = [], []
    for name, entry in grad.get("switch", {}).items():
        if entry["lambda_ratio"] is not None:
            names.append(name)
            ratios.append(entry["lambda_ratio"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if names:
        bars = ax.bar(range(len(names)), ratios, color="#4c72b0")
        ax.axhline(CLEARS, color="tab:green", lw=1.2, ls="--",
                   label=f"clears (< {CLEARS:g})")
        ax.axhline(REDUCES, color="tab:red", lw=1.2, ls="--",
                   label=f"does not own (>= {REDUCES:g})")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        for b, r in zip(bars, ratios):
            ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\lambda_{\rm arm} / \lambda_{\rm base}$")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no switch arm produced a band rate",
                ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Switch ratio against the reference arm")
    fig.tight_layout()
    path = outdir / "lake_switch_horizon_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--t-end", type=float, default=PROTOCOL["t_end"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dx", type=float, default=PROTOCOL["dx"])
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    names = args.arms or list(ARMS)
    if "base" not in names:
        # every ratio needs its denominator; a subset run is allowed (the arms
        # are graded independently), but say so -- the artefact then carries no
        # denominator and grades no switch ratio
        print("warning: running without the reference arm: no switch ratio will "
              "be graded", flush=True)
    protocol = dict(PROTOCOL)
    protocol.update({"dx": args.dx, "t_end": args.t_end,
                     "threads": args.threads})
    base_cfg = {"t_end": args.t_end, "periodic_x": True, "dx": args.dx}

    started = time.time()
    records = {}
    for name in names:
        cfg = M.build_config(ARMS[name], base_cfg)
        t0 = time.time()
        run = M.integrate(cfg, dict(ARMS[name], sample=PROTOCOL["sample_every"]))
        record = arm_record(name, cfg, run)
        record["runtime_s"] = time.time() - t0
        records[name] = record
        rate = record["growth"]["lambda_per_s"]
        print(f"  {name:12s} lambda={'n/a' if rate is None else f'{rate:8.3f}'}"
              f"  max|v|/c0={record['max_v_over_c0']:.3e}"
              f"  t*={record['persists_to_tolerance']['t_star']}"
              f"  eq_valid={record['equilibrium_valid']}"
              f"  {record['runtime_s']:.0f}s", flush=True)

    payload = {
        "case": CASE_ID,
        "kind": "does a shipped closure clear the lake-at-rest departure "
                "(or only delay it)?",
        "band": [BAND_LOW, BAND_HIGH],
        "protocol": protocol,
        "torch": torch.__version__,
        "runtime_s": time.time() - started,
        "arms": records,
    }
    payload["grading"] = grade(records)
    print("\n" + payload["grading"]["verdict"]["statement"])
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {args.json}  ({payload['runtime_s']:.0f} s)")
    if args.figures:
        for path in figures(payload, args.figures):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Is the frame-level peak residual of the wedge-entry case *physics* or *noise*?

Why this module exists
----------------------
CASE-SPH-WEDGE-ENTRY-2D has been graded four times with four different peak
estimators (pooled 120-bin profile m17548, pooled sample-gated profile m17899,
resolution-band profile m22695, cell footprint m21677) and the reported number
moves between ``6.8`` and ``13.4`` q-units -- a factor two -- while every round
compares a *measured* max against a *point* value of Wagner's profile.  Two
rounds of audit already separated the parts of that gap that are the estimator:
``ensemble_peak.py`` (pooling) and ``estimator_bias.py`` / ``footprint_ledger.py``
(a forward model of the measurement chain on a known field, at two grids).

What is still missing -- and what this module adds
------------------------------------------------
1. **A same-operator reference at the frame level.**  The committed artefacts
   (``docs/artifacts/case_sph_wedge_entry_2d_ensemble/ensemble_dx*_c0v20.json``)
   carry, for each of 61 frames, the peak of a *single-frame* 12-bin profile
   (``frame_scan[*].cp_peak_single_frame``).  That statistic has a reference of
   its own: put the exact Wagner field on the *same* wall-node grid, bin it with
   the *same* 12 equal bins of ``tau in [0, 1.2]``, keep the bins with centre
   ``<= 1`` and take the maximum of the bin means.  The result ``A12(t)`` is what
   the estimator would read on the exact solution, so ``E = obs/A12 - 1`` is a
   residual with the discretisation response of the operator removed.  Nothing
   here needs a solver run: the wall nodes are rebuilt from the committed
   geometry (``wedge_lattice``: ``|x_k| = dx((k+1/2) cos b - (1/2) sin b)``,
   ``|x_k| <= c(t) = C V t``) and the count is checked against the committed
   ``n_wet`` of every frame.
2. **The scaling of that residual with the grid.**  Two hypotheses are
   separated by the exponent of ``E ~ dx^p``:

   * H_noise: the residual is the upward bias of a *maximum over bin means* in a
     noisy pressure field.  A bin holds ``n_i ~ dx^-1`` nodes, so the bin mean
     carries ``sigma/sqrt(n_i)`` noise and ``E[max over 10 bins]`` scales as
     ``dx^(+1/2)``.  Prediction: ``p ~ +0.5`` and the residual grows on coarser
     grids -- i.e. it is an instrument floor, and refining pays only as
     ``dx^(1/2)``;
   * H_discretisation: the residual is the scheme's own convergence error, which
     for this second-order-in-space WCSPH scheme should fall at least as fast as
     ``dx^(+1)``.  Prediction: ``p >= 1``.

   The two are falsifiable against the same four numbers, and the *sign* of the
   residual (positive on all four grids, see ``regrade.json``) already excludes
   a pure discretisation error (which would be sign-free).
3. **How many frames a decision would need.**  ``median`` over frames is used as
   the location because the per-frame peak is heavy tailed (CoV 40-50 %,
   ``stationarity.json``); the location's standard error is estimated with a
   *moving-block* bootstrap (frames are consecutive samples of one trajectory,
   so an i.i.d. bootstrap understates the error) and the effective sample size
   ``N_eff = n (1-rho)/(1+rho)`` is reported next to it.

Predeclared rules (fixed before any number was looked at)
--------------------------------------------------------
* wet-node gate ``n_wet >= 20``: the 12-bin profile has 10 participating bins
  (centres ``<= 1``), and the case is symmetric, so 20 wetted nodes is the
  smallest count that puts at least one node in every participating bin on each
  half.  A gate sweep ``(12, 20, 30, 60)`` is reported as a systematic term; the
  declared reading uses the a-priori gate ``20``, not the sweep minimum.
* location = median of the per-frame residuals; dispersion = IQR and CoV;
  uncertainty = ``max(moving-block bootstrap half width, |early-late|/2)`` over
  the gated frames.
* a grid is *judgeable* when that uncertainty is ``<= GATE`` (15 %, the case
  tolerance, unchanged); the ladder verdict is ``ladder_closure.verdict()``, the
  same rule every other round of this case uses -- this module does not invent a
  scoring rule.
* the exponent fit is ordinary least squares of ``ln|E|`` on ``ln dx`` over the
  four grid medians, with the 5..95 % bootstrap band of the medians propagated
  (resampling the frames, refitting per resample).

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/noise_floor_peak.py           # JSON to stdout
    python benchmarks/case_sph_wedge_entry_2d/noise_floor_peak.py --write   # + artefact
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorsph.wedge_entry import wagner_c_coeff, wagner_p_over_q  # noqa: E402

ENSEMBLE_DIR = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_ensemble"
OUT_DIR = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_noise_floor"

GATE = 0.15                 # case tolerance, unchanged (ladder_closure.GATE)
N_BINS = 12                 # the single-frame profile operator of ensemble_peak
TAU_MAX = 1.2               # binning range of ensemble_peak.binned
TAU_HI = 1.0                # participating bins: centre <= TAU_HI
WET_GATE = 20               # predeclared (see docstring)
GATE_SWEEP = (12, 20, 30, 60)
BLOCK = 5                   # moving-block length (sqrt(61) ~ 8; 5 keeps >= 12 blocks)
BOOT = 2000
SEED = 20261114

GRIDS = ((0.004, "dx4mm"), (0.002, "dx2mm"), (0.001, "dx1mm"), (0.0005, "dx0.5mm"))


# --------------------------------------------------------------------------- #
# geometry: the analytic wall-node grid of the committed run
# --------------------------------------------------------------------------- #

def wall_node_taus(dx: float, beta: float, C: float, V: float, t: float,
                   extra_nodes: int = 0) -> list[float]:
    """``tau = |x_k| / (C V t)`` of the first wall layer inside the footprint.

    One half of the body (the case is symmetric); ``|x_k|`` is
    ``wedge_lattice``-exact: ``s = (k + 1/2) dx`` along the face and the first
    row sits ``n = dx/2`` into the body, so
    ``|x_k| = s cos(beta) - (dx/2) sin(beta)``.  The second half contributes the
    same ``|x_k|`` values, which is why the node *count* is twice this length
    (the bin means do not change under the duplication).
    """
    c = C * V * t
    if c <= 0.0:
        return []
    taus = []
    k = 0
    while True:
        x = (k + 0.5) * dx * math.cos(beta) - 0.5 * dx * math.sin(beta)
        if x > c:
            break
        taus.append(x / c)
        k += 1
    if extra_nodes > 0:
        for _ in range(extra_nodes):
            x = (k + 0.5) * dx * math.cos(beta) - 0.5 * dx * math.sin(beta)
            taus.append(x / c)
            k += 1
    elif extra_nodes < 0:
        taus = taus[:max(0, len(taus) + extra_nodes)]
    return taus


def analytic_binned_peak(taus, beta: float) -> float:
    """The frame-level statistic applied to the exact Wagner field.

    Line-for-line the operator of ``ensemble_peak.py``: equal bins of width
    ``TAU_MAX/N_BINS`` in ``tau``, bin value = mean of the field over the nodes
    in the bin, peak = max over bins whose centre is ``<= TAU_HI`` (``min_count``
    = 1, exactly as ``gated_peak`` is called for ``frame_scan``).
    """
    sums = [0.0] * N_BINS
    cnts = [0] * N_BINS
    width = TAU_MAX / N_BINS
    for tau in taus:
        if tau != tau or tau < 0.0 or tau > TAU_MAX:
            continue
        k = min(int(tau / TAU_MAX * N_BINS), N_BINS - 1)
        sums[k] += wagner_p_over_q(tau, beta)
        cnts[k] += 1
    best = float("nan")
    for k in range(N_BINS):
        if not cnts[k] or (k + 0.5) * width > TAU_HI:
            continue
        m = sums[k] / cnts[k]
        if not (best == best) or m > best:
            best = m
    return best


# --------------------------------------------------------------------------- #
# per-grid extraction
# --------------------------------------------------------------------------- #

def load_grid(dx: float, tag: str, ensemble_dir: Path = ENSEMBLE_DIR) -> dict:
    path = ensemble_dir / f"ensemble_{tag.replace('dx', 'dx')}_c0v20.json"
    if not path.exists():                      # tolerate the naming variants
        cands = sorted(ensemble_dir.glob(f"ensemble_{tag}_c0v20.json")) or \
            sorted(ensemble_dir.glob("ensemble_*c0v20.json"))
        path = next((p for p in cands if tag in p.name), path)
    data = json.loads(path.read_text())
    beta = math.radians(data["config"]["beta_deg"])
    C = wagner_c_coeff(beta)
    rows = []
    for fr in data["frame_scan"]:
        taus = wall_node_taus(data["config"]["dx"], beta, C,
                              data["config"]["velocity"], fr["t"])
        a12 = analytic_binned_peak(taus, beta)
        obs = fr["cp_peak_single_frame"]
        ok = (a12 == a12 and a12 > 0.0 and obs == obs)
        rows.append({
            "t": fr["t"],
            "n_wet": fr["n_wet"],
            "n_nodes_rebuilt": 2 * len(taus),
            "a12": a12,
            "obs_peak": obs,
            "residual": (obs / a12 - 1.0) if ok else float("nan"),
            "cp_centre": fr["cp_centre"],
            "max_rho_ratio": fr["max_rho_ratio"],
            "tau_at_peak": fr["tau_at_peak"],
        })
    return {"tag": tag, "dx": dx, "_V": data["config"]["velocity"], "C": C,
            "path": str(path.relative_to(ROOT)),
            "beta_deg": data["config"]["beta_deg"], "c0_factor": data["config"]["c0_factor"],
            "wagner_point_peak": data["wagner_cp_max"],
            "frames": rows}


def cutoff_sensitivity(grid: dict, offsets=(-2, 0, 2)) -> dict:
    """How much the reading moves when the wet/dry cutoff moves by a few nodes.

    The measurement side is untouched (the committed ``obs_peak`` is read as it
    is); only the reference ``A12`` is rebuilt, so this bounds the error the
    geometric rebuild can inject into the residual.
    """
    dx, beta = grid["dx"], math.radians(grid["beta_deg"])
    C = wagner_c_coeff(beta)
    meds = {}
    for off in offsets:
        vals = []
        for fr in grid["frames"]:
            if fr["n_wet"] < WET_GATE or fr["residual"] != fr["residual"]:
                continue
            base = fr["_recon"] if "_recon" in fr else None
            taus = wall_node_taus(dx, beta, C, grid["_V"], fr["t"], extra_nodes=off)
            a12 = analytic_binned_peak(taus, beta)
            if a12 == a12 and a12 > 0 and fr["obs_peak"] == fr["obs_peak"]:
                vals.append(fr["obs_peak"] / a12 - 1.0)
        meds[str(off)] = _median(vals)
    finite = [v for v in meds.values() if v == v]
    return {"medians": meds,
            "max_abs_shift": max(finite) - min(finite) if len(finite) > 1 else float("nan")}


def selectivity_checks(grid: dict) -> dict:
    """Nothing below is a scoring rule -- these are fidelity checks of the rebuild."""
    exact = [r for r in grid["frames"] if r["n_nodes_rebuilt"] == r["n_wet"]]
    off = [r["n_nodes_rebuilt"] - r["n_wet"] for r in grid["frames"]
           if r["n_nodes_rebuilt"] != r["n_wet"]]
    return {
        "n_frames": len(grid["frames"]),
        "n_wet_reproduced_exactly": len(exact),
        "max_abs_node_count_error": max((abs(o) for o in off), default=0),
        "a12_min": min((r["a12"] for r in grid["frames"] if r["a12"] == r["a12"]), default=float("nan")),
        "a12_max": max((r["a12"] for r in grid["frames"] if r["a12"] == r["a12"]), default=float("nan")),
        "a12_over_point_peak": [r["a12"] / grid["wagner_point_peak"]
                                for r in grid["frames"] if r["a12"] == r["a12"]],
    }


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def _median(v):
    return statistics.median(v) if v else float("nan")


def moving_block_bootstrap(values, block: int = BLOCK, rounds: int = BOOT, seed: int = SEED):
    """5..95 % band of the median under a moving-block resample (time correlation)."""
    n = len(values)
    if n < 8:
        return {"n": n, "p05": float("nan"), "p50": float("nan"), "p95": float("nan"),
                "half_width": float("nan")}
    rng = random.Random(seed)
    n_blocks = max(2, int(math.ceil(n / block)))
    stats = []
    for _ in range(rounds):
        sample = []
        while len(sample) < n:
            start = rng.randrange(0, max(1, n - block + 1))
            sample.extend(values[start:start + block])
        stats.append(_median(sample[:n]))
    stats.sort()
    p05, p50, p95 = (stats[int(0.05 * (len(stats) - 1))], _median(stats),
                     stats[int(0.95 * (len(stats) - 1))])
    return {"n": n, "p05": p05, "p50": p50, "p95": p95,
            "half_width": 0.5 * (p95 - p05)}


def lag1_autocorr(values) -> float:
    n = len(values)
    if n < 4:
        return float("nan")
    m = statistics.fmean(values)
    num = sum((values[i] - m) * (values[i + 1] - m) for i in range(n - 1))
    den = sum((v - m) ** 2 for v in values)
    return num / den if den else float("nan")


def summarize(residuals, gate: float = GATE, wet_gate: int = WET_GATE) -> dict:
    """Location, dispersion and declared uncertainty of the gated frame residuals."""
    used = [r for r in residuals
            if r["n_wet"] >= wet_gate and r["residual"] == r["residual"]]
    vals = [r["residual"] for r in used]
    if not vals:
        return {"n_used": 0, "median": float("nan"), "judgeable": False}
    med = _median(vals)
    q1, q3 = (sorted(vals)[len(vals) // 4], sorted(vals)[3 * len(vals) // 4])
    boot = moving_block_bootstrap(vals)
    h = len(vals) // 2
    early = _median(vals[:h])
    late = _median(vals[-h:]) if h else float("nan")
    split = abs(late - early) / 2.0 if h else float("nan")
    rho = lag1_autocorr(vals)
    n_eff = len(vals) * (1.0 - rho) / (1.0 + rho) if rho == rho and abs(rho) < 1.0 else len(vals)
    terms = [t for t in (boot["half_width"], split) if t == t]
    unc = max(terms) if terms else float("nan")
    return {
        "wet_gate": wet_gate,
        "n_used": len(vals),
        "n_frames_gated_out": len(residuals) - len(vals),
        "median": med,
        "mean": statistics.fmean(vals),
        "iqr": q3 - q1,
        "cov": statistics.pstdev(vals) / abs(statistics.fmean(vals)) if statistics.fmean(vals) else float("nan"),
        "half_split_early": early,
        "half_split_late": late,
        "half_split_term": split,
        "bootstrap": boot,
        "lag1_autocorr": rho,
        "n_eff": n_eff,
        "frames_for_gate": _frames_for_gate(statistics.pstdev(vals), med),
        "uncertainty": unc,
        "judgeable": bool(unc == unc and unc <= gate),
    }


def _frames_for_gate(sd: float, med: float) -> float:
    """Frames a median would need for a 15 % standard error, from this scatter."""
    if not (sd == sd) or med == 0.0:
        return float("nan")
    # se = 1.2533 sd / sqrt(n) (median of a normal sample); solve se = GATE*|med|
    target = GATE * abs(med)
    return (1.2533 * sd / target) ** 2


def within_grid_scaling(frames, wet_gate: int = WET_GATE) -> dict:
    """Slope of ``ln|E|`` on ``ln n_wet`` inside one grid (second noise test).

    The extreme-value model says a bin mean over ``n_i ~ dx^-1`` nodes carries
    ``sigma/sqrt(n_i)`` noise, so inside one grid ``|E| ~ n_wet^(-1/2)``: the
    prediction is ``slope = -0.5``.  A discretisation error cannot produce this
    within-grid trend, because ``n_wet`` grows with ``t`` while the analytic
    profile is self-similar in ``tau`` -- the scheme's error is a function of
    ``dx``, not of the frame index.
    """
    used = [(f["n_wet"], f["residual"]) for f in frames
            if f["n_wet"] >= wet_gate and f["residual"] == f["residual"]
            and abs(f["residual"]) > 0.0]
    if len(used) < 6:
        return {"slope": float("nan"), "n": len(used), "predicted_slope": -0.5}
    xs = [math.log(n) for n, _ in used]
    ys = [math.log(abs(e)) for _, e in used]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    return {"slope": slope, "n": len(used), "predicted_slope": -0.5}


def acoustic_correlation(frames, wet_gate: int = WET_GATE) -> dict:
    """Spearman rank correlation of ``E`` with the acoustic density error.

    ``max_rho_ratio - 1`` is the committed per-frame measure of how far the
    weakly compressible solver's density is from the incompressible limit.  If
    the peak residual were a discretisation error of the *profile* it should not
    track that number; if it is pressure noise it should.
    """
    used = [(f["residual"], f["max_rho_ratio"] - 1.0) for f in frames
            if f["n_wet"] >= wet_gate and f["residual"] == f["residual"]
            and f["max_rho_ratio"] == f["max_rho_ratio"]]
    if len(used) < 6:
        return {"rho_s": float("nan"), "n": len(used)}
    return {"rho_s": _spearman([a for a, _ in used], [b for _, b in used]),
            "n": len(used)}


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def fit_exponent(points, gate: float = GATE) -> dict:
    """Least-squares slope of ``ln|median|`` on ``ln dx`` (points: (dx, median))."""
    pts = [(math.log(dx), math.log(abs(m))) for dx, m in points
           if m == m and m != 0.0]
    if len(pts) < 2:
        return {"slope": float("nan"), "n": len(pts)}
    n = len(pts)
    mx = statistics.fmean(p[0] for p in pts)
    my = statistics.fmean(p[1] for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    intercept = my - slope * mx
    pred = [intercept + slope * p[0] for p in pts]
    resid = [p[1] - q for p, q in zip(pts, pred)]
    ss = sum(r * r for r in resid)
    return {"slope": slope, "intercept": intercept, "n": n,
            "rms_ln_residual": math.sqrt(ss / n),
            "predicted_slope_noise": 0.5, "predicted_slope_discretisation": 1.0}


def bootstrap_exponent(grid_residuals, gate: float = GATE) -> dict:
    """Slope band when the frames of every grid are resampled together."""
    rng = random.Random(SEED + 1)
    slopes = []
    for _ in range(400):
        pts = []
        for dx, vals in grid_residuals:
            if not vals:
                continue
            sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
            pts.append((dx, _median(sample)))
        s = fit_exponent(pts)["slope"]
        if s == s:
            slopes.append(s)
    if not slopes:
        return {"p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    slopes.sort()
    return {"p05": slopes[int(0.05 * (len(slopes) - 1))],
            "p50": _median(slopes),
            "p95": slopes[int(0.95 * (len(slopes) - 1))]}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def build(ensemble_dir: Path = ENSEMBLE_DIR) -> dict:
    grids = [load_grid(dx, tag, ensemble_dir) for dx, tag in GRIDS]
    grids = [g for g in grids if g["frames"]]
    out = {"case": "CASE-SPH-WEDGE-ENTRY-2D",
           "script": "benchmarks/case_sph_wedge_entry_2d/noise_floor_peak.py",
           "design": {"gate": GATE, "n_bins": N_BINS, "tau_max": TAU_MAX,
                      "tau_hi": TAU_HI, "wet_gate": WET_GATE,
                      "gate_sweep": list(GATE_SWEEP), "block": BLOCK,
                      "bootstrap": BOOT, "seed": SEED,
                      "statistic": "median over gated frames of obs_peak/A12 - 1",
                      "reference": "Wagner field through the same 12-bin wall-node operator",
                      "verdict_rule": "ladder_closure.verdict(), unchanged"},
           "grids": []}
    for g in grids:
        chk = selectivity_checks(g)
        summ = summarize(g["frames"])
        sweep = {str(w): summarize(g["frames"], wet_gate=w) for w in GATE_SWEEP}
        a12_rel = chk.pop("a12_over_point_peak")
        g_out = {"tag": g["tag"], "dx": g["dx"], "artifact": g["path"],
                 "beta_deg": g["beta_deg"], "C": g["C"], "velocity": g["_V"],
                 "c0_factor": g["c0_factor"],
                 "wagner_point_peak": g["wagner_point_peak"],
                 "fidelity": chk,
                 "a12_over_point_peak_median": _median(a12_rel),
                 "reading": summ,
                 "cutoff_sensitivity": cutoff_sensitivity(g),
                 "within_grid_scaling": within_grid_scaling(g["frames"]),
                 "acoustic_correlation": acoustic_correlation(g["frames"]),
                 "gate_sweep": sweep,
                 "frames": g["frames"]}
        out["grids"].append(g_out)

    # ladder: same rule as every other round of this case
    rows = [{"tag": g["tag"], "dx": g["dx"], "residual": g["reading"]["median"],
             "uncertainty": g["reading"]["uncertainty"],
             "judgeable": g["reading"]["judgeable"]} for g in out["grids"]]
    import ladder_closure  # noqa: E402  (house decision rule, not a new one)
    out["ladder"] = ladder_closure.verdict(rows)
    out["exponent"] = fit_exponent([(g["dx"], g["reading"]["median"]) for g in out["grids"]])
    out["exponent_bootstrap"] = bootstrap_exponent(
        [(g["dx"], [f["residual"] for f in g["frames"]
                    if f["n_wet"] >= WET_GATE and f["residual"] == f["residual"]])
         for g in out["grids"]])
    # what the fitted law implies for the grid a 15 % residual would need
    exp = out["exponent"]
    fine = min(out["grids"], key=lambda g: g["dx"])
    if exp["slope"] == exp["slope"] and fine["reading"]["median"] > 0.0:
        out["dx_for_gate"] = fine["dx"] * (GATE / fine["reading"]["median"]) ** (1.0 / exp["slope"])
    else:
        out["dx_for_gate"] = float("nan")
    return out


def _print_brief(out: dict) -> None:
    print(f"{'grid':>8} {'n':>4} {'median':>9} {'IQR':>7} {'unc':>7} {'judge':>6} {'A12/peak':>9}")
    for g in out["grids"]:
        r = g["reading"]
        print(f"{g['tag']:>8} {r['n_used']:>4} {r['median']*100:+8.1f}% {r['iqr']*100:6.1f}% "
              f"{r['uncertainty']*100:6.1f}% {str(r['judgeable']):>6} "
              f"{g['a12_over_point_peak_median']:9.3f}")
    e = out["exponent"]
    b = out["exponent_bootstrap"]
    print(f"exponent p = {e['slope']:+.3f} (bootstrap p05..p95 "
          f"{b['p05']:+.3f}..{b['p95']:+.3f}); dx for a {GATE:.0%} residual = "
          f"{out['dx_for_gate']*1e3:.2f} mm")
    print("ladder:", out["ladder"]["verdict"], "--", out["ladder"]["reason"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ensemble-dir", default=str(ENSEMBLE_DIR))
    ap.add_argument("--write", action="store_true", help="write the artefact JSON")
    ap.add_argument("--brief", action="store_true", help="human summary only")
    args = ap.parse_args(argv)
    out = build(Path(args.ensemble_dir))
    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = OUT_DIR / "noise_floor.json"
        dest.write_text(json.dumps(out, indent=1))
        print(f"wrote {dest.relative_to(ROOT)}")
    if args.brief:
        _print_brief(out)
    elif not args.write:
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
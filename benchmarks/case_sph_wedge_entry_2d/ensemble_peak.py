"""Does a self-similar *frame ensemble* make the peak wall pressure judgeable?

Context (measured, not assumed)
-------------------------------
``run.py --stage accept`` and ``metric_resolution.py`` (m17548) established two
facts for CASE-SPH-WEDGE-ENTRY-2D at ``beta = 30 deg, V = 1 m/s``:

* the resolution-robust quantities pass the 15 % gate at ``dx = 2 mm`` --
  wetted half-width ``c/(C V t)`` -0.5 %, stagnation-line ``Cp(0)`` -10.5 %,
  peak abscissa ``tau_peak`` 0.92-0.94 vs the analytic 0.930;
* the *peak* ``Cp`` is **not judgeable at dx = 2 mm**: with ``profile_bins = 120``
  the profile has more bins than wetted nodes, so the "profile peak" is one
  particle's pressure noise (+234 % vs the analytic 8.4022), and the estimator
  spreads by 3.0x across bin counts -- an uncertainty 20x the 15 % tolerance.

m17548 left two routes: (a) write the bin rule down (resolution-consistent
binning + more frames in the ensemble) or (b) go to ``dx <= 1 mm`` (compute
bound).  This script tests (a), which costs one trajectory per grid instead of a
grid refinement.

Hypotheses, declared before any run
----------------------------------
H1 (the metric becomes judgeable).  The wall field is self-similar in
``tau = |x| / (C V t)``, so frames at different ``t`` are legitimate ensemble
members.  Pooling 61 frames (vs the 5-frame window used so far) at a *sample
gated* bin count (every bin must hold ``>= MIN_BIN_SAMPLES`` pooled samples)
makes the estimator's spread -- max/min over the bin-count grid -- smaller than
the 15 % acceptance tolerance, at ``dx = 2 mm``, with no solver change.
H2 (falsified if the spread stays above 15 %).  Then the peak needs
``dx <= 1 mm`` and the case must keep reporting it as WITHHELD at this grid.

A second, independent question is asked of the same trajectory: which frames
carry the spike?  If the ensemble peak is set by the first frames (the
incompressible solution has a pressure *step* at ``t = 0+``, which a
weakly compressible solver can only build through an acoustic transient of
order ``rho c0 V``, i.e. ``Cp ~ 2 c0 / V``), then a startup window exists that
must be excluded by rule rather than by taste.  ``frame_scan`` reports the
per-frame peak so the choice is data-driven and auditable.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py \
        --dx 0.002 --frames 61 --out artifacts/case_sph_wedge_entry_2d_ensemble

Writes one JSON artefact per grid: the pooled profile, the per-frame peak scan,
and the estimator spread for a growing frame count.  Nothing here changes the
solver, the case definition or a threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tensorsph.wedge_entry import (  # noqa: E402
    WedgeConfig,
    WedgeImpact,
    with_realisation,
    wagner_c_coeff,
    wagner_p_over_q,
    wagner_peak_over_q,
)

# --- predeclared estimator design (fixed before the first run) ------------- #
FRAME_COUNTS = (7, 13, 25, 41, 61)   # growing ensemble size (>5 = old window)
BIN_GRID = (6, 8, 10, 12, 16, 20, 30)
# Candidate bin counts for the *resolution band* estimator.  BIN_GRID stops at
# 30 bins, which cannot represent the physically admissible band at the two
# finest grids (a band of width [1,4] particle spacings needs ~60 bins at
# dx = 1 mm and ~120 at dx = 0.5 mm, see ``resolution_bin_band``); the band was
# therefore truncated to one side and, at dx = 0.5 mm, collapsed to a single
# member whose "spread" is 0 by construction.  BAND_GRID spans the band up to
# 240 bins and is a superset of BIN_GRID so the historical H1 statistic is
# unchanged.  (m22695)
BAND_GRID = (6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60, 80, 120, 160, 240)
MIN_BIN_SAMPLES = 4                  # a bin needs >= 4 pooled samples
TAU_MAX = 1.2
SPREAD_TOL = 0.15                    # acceptance tolerance of the case (15 %)
BOOT_RESAMPLES = 400                 # frame bootstrap for the matched reading
BOOT_SEED = 20261113
FOOTPRINT_CELLS = (2.0, 4.0, 8.0, 16.0)   # sensor footprints, in cells
# Startup-window rule: a frame is "acoustically relaxed" when the pressure wave
# has crossed the wetted width at least once, t >= c(t) / c0 -> tau_relax =
# t c0 / c = c0 / (C V), which for c0/V = 20, C = 2.72 gives t/t_end ~ 3.7/none
# -- i.e. always true here, so acoustic relaxation is NOT a frame filter.  The
# remaining candidate is the particle-scale transient, t >= N dx / c0.
ACOUSTIC_CROSSINGS = 3.0


def wall_samples(sim: WedgeImpact):
    """Per-node ``(tau, cp)`` of the *contacting* wall nodes, current state.

    Same definitions as ``WedgeImpact.diagnostics``: tau is normalised by the
    analytic wetted edge ``C V t``, ``cp = p_wall / q`` uses the Adami wall
    extrapolation, contact means kernel support above ``cfg.contact_support``.
    """
    cfg = sim.cfg
    state = sim._evaluate(sim._last_dt)
    p_wall = state["p_wall"][: sim.n_wedge]
    den = state["den"][: sim.n_wedge]
    surf = sim.is_first
    x = (sim.wedge_local[:, 0] + sim.body_offset[0])[surf]
    p = p_wall[surf]
    contact = den[surf] > cfg.contact_support * sim.support_ref
    c_w = wagner_c_coeff(cfg.beta) * sim.V * sim.time
    if c_w <= 0.0 or not bool(contact.any()):
        return [], []
    tau = (x.abs()[contact] / c_w).tolist()
    cp = (p[contact] / sim.q).tolist()
    return tau, cp


def binned(taus, cps, n_bins: int):
    """Pooled profile: ``(centres, means, counts)`` over ``tau in [0, TAU_MAX]``."""
    sums = [0.0] * n_bins
    cnts = [0] * n_bins
    for tau, cp in zip(taus, cps):
        if not (tau == tau) or not (cp == cp) or tau < 0.0 or tau > TAU_MAX:
            continue
        k = min(int(tau / TAU_MAX * n_bins), n_bins - 1)
        sums[k] += cp
        cnts[k] += 1
    width = TAU_MAX / n_bins
    centres = [(k + 0.5) * width for k in range(n_bins)]
    means = [sums[k] / cnts[k] if cnts[k] else float("nan") for k in range(n_bins)]
    return centres, means, cnts


def gated_peak(centres, means, cnts, min_count: int = MIN_BIN_SAMPLES,
               tau_hi: float = 1.0):
    """Peak of the pooled profile over ``tau <= tau_hi`` on bins holding data.

    ``min_count`` is the sample gate: a bin that pooled fewer than that many
    samples is not allowed to set the peak (this is the rule whose absence made
    the 120-bin "peak" a single-particle reading).
    """
    best = float("nan")
    best_tau = float("nan")
    best_n = 0
    for c, m, n in zip(centres, means, cnts):
        if not (m == m) or c > tau_hi or n < min_count:
            continue
        if not (best == best) or m > best:
            best, best_tau, best_n = m, c, n
    return best, best_tau, best_n


def resolution_bin_band(c_last: float, dx: float):
    """Bin counts whose bin width in *metres* stays in ``[1 dx, 4 dx]``.

    Justified a priori, before any measurement: a bin narrower than one particle
    spacing is a single particle (the m17548 defect), and a bin wider than four
    spacings cannot resolve the profile's rise over the spray root.  This is the
    uncertainty band of the *reported* peak; the full ``BIN_GRID`` spread is kept
    as the predeclared H1 statistic.
    """
    lo = int(math.floor(TAU_MAX * c_last / (4.0 * dx)))
    hi = int(math.ceil(TAU_MAX * c_last / (1.0 * dx)))
    lo_b, hi_b = max(4, lo), min(240, hi)
    return lo_b, hi_b, [n for n in BAND_GRID if lo_b <= n <= hi_b]


def jackknife_halves(shots):
    """Split the frame list in two halves (early / late) -- same estimator."""
    n = len(shots)
    return shots[: n // 2], shots[n // 2:]


def pooled(shots):
    taus: list[float] = []
    cps: list[float] = []
    for s in shots:
        taus.extend(s["tau"])
        cps.extend(s["cp"])
    return taus, cps


def spread(values):
    """``(min, max, median, (max-min)/|median|)``; NaN-safe."""
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    lo, hi = min(vals), max(vals)
    med = statistics.median(vals)
    rel = (hi - lo) / abs(med) if med else float("nan")
    return lo, hi, med, rel


def collect(dx: float, velocity: float, frames: int, c0_factor: float,
            threads: int, seed: int | None = None,
            scale: float = 1.0) -> dict:
    """Run the case once.

    ``seed`` (m22826) selects an *independent realization* through the two
    sub-cell discrete-phase offsets of ``tensorsph.wedge_entry``; ``None``
    reproduces the unperturbed case bit for bit, which is what every committed
    artefact up to m22798 was produced with.
    """
    torch.set_num_threads(max(1, int(threads)))
    cfg = WedgeConfig(beta_deg=30.0, velocity=velocity, dx=dx, frames=frames,
                      alpha_visc=0.03, sound_speed_factor=c0_factor,
                      max_courant=0.35)
    if seed is not None:
        cfg = with_realisation(cfg, seed, scale)
    cfg.validate()
    t0 = time.time()
    sim = WedgeImpact(cfg)
    C = wagner_c_coeff(cfg.beta)
    t_end = cfg.resolved_t_end()
    times = [t_end * k / (frames - 1) for k in range(frames)]

    shots = []
    target = 0
    steps = 0
    while sim.time < t_end - 1.0e-15:
        dt = min(sim.dt(), t_end - sim.time)
        if target < len(times):
            dt = min(dt, max(times[target] - sim.time, 1.0e-18))
        sim.advance(dt)
        steps += 1
        if target < len(times) and sim.time >= times[target] - 1.0e-15:
            diag = sim.diagnostics()
            tau, cp = wall_samples(sim)
            shots.append({"t": sim.time, "tau": tau, "cp": cp,
                          "n_wet": len(tau), "c_sph": diag["c"],
                          "cp_footprint_max": diag["p_max_footprint"] / sim.q,
                          "cp_centre": diag["p_centre"] / sim.q,
                          "max_rho_ratio": diag["max_rho_ratio"]})
            target += 1
    wall = time.time() - t0
    cfg_out = {"beta_deg": 30.0, "velocity": velocity, "dx": dx,
               "frames": frames, "c0_factor": c0_factor,
               "courant": 0.35, "alpha_visc": 0.03}
    if seed is not None:
        # provenance of the realization: without these three numbers a reading
        # cannot be reproduced, and an ensemble of unknown phases is not an
        # ensemble
        cfg_out["phase_x_dx"] = float(cfg.phase_x_dx)
        cfg_out["gap_jitter_dx"] = float(cfg.gap_jitter_dx)
        cfg_out["realisation_seed"] = int(seed)
        cfg_out["realisation_scale"] = float(scale)
    return {"cfg": cfg_out,
            "steps": steps, "wall_seconds": wall, "t_end": t_end,
            "C": C, "c0": sim.c0, "q": sim.q,
            "wagner_cp_max": wagner_peak_over_q(cfg.beta),
            "wagner_cp_centre": wagner_p_over_q(0.0, cfg.beta),
            "wagner_tau_peak": math.sqrt(max(0.0, 1.0 - 1.0 / C ** 2)),
            "shots": shots}


def analyse(case: dict, bin_grid=None) -> dict:
    """Estimator spread for a growing frame count + per-frame scan + jackknife."""
    grid = tuple(bin_grid) if bin_grid else tuple(BIN_GRID)
    shots = case["shots"]
    C = case["C"]
    ref = case["wagner_cp_max"]
    n_avail = len(shots)

    by_frames = {}
    for n_frames in FRAME_COUNTS:
        if n_frames > n_avail:
            continue
        sel = shots[-n_frames:]          # last n_frames of the window
        taus, cps = pooled(sel)
        peaks, rows = [], []
        for n_bins in grid:
            centres, means, cnts = binned(taus, cps, n_bins)
            peak, tau_at, n_at = gated_peak(centres, means, cnts)
            peaks.append(peak)
            rows.append({"n_bins": n_bins, "bin_width_tau": TAU_MAX / n_bins,
                         "cp_peak_gated": peak, "tau_at_peak": tau_at,
                         "samples_at_peak": n_at,
                         "n_bins_setting_peak": sum(
                             1 for m, n, c in zip(means, cnts, centres)
                             if m == m and n >= MIN_BIN_SAMPLES and c <= 1.0)})
        lo, hi, med, rel = spread(peaks)
        c_last = sel[-1]["c_sph"] if sel else float("nan")
        rb_lo, rb_hi, rb_bins = resolution_bin_band(c_last, case["cfg"]["dx"])
        # The band statistic must be measured at the band's own bin counts, not
        # read out of ``rows`` (which only holds BIN_GRID <= 30): otherwise the
        # candidate list is silently truncated to the coarse half of the band.
        rb_band_peaks = []
        for nb in rb_bins:
            pk = gated_peak(*binned(taus, cps, nb))[0]
            rb_band_peaks.append({"n_bins": nb, "cp_peak_gated": pk})
        rb_peaks = [r["cp_peak_gated"] for r in rb_band_peaks]
        rb_est = sum(1 for v in rb_peaks if v == v) >= 2
        if rb_est:
            rb_lo_v, rb_hi_v, rb_med, rb_rel = spread(rb_peaks)
        else:
            # A one-member "band" has no spread: 0.0 would be a fabricated zero
            # (m22695 defect), so the quantity is declared not estimable.
            rb_lo_v, rb_hi_v, rb_med, rb_rel = spread(rb_peaks)
            rb_rel = float("nan")
        half = n_frames // 2
        jk = {}
        for tag, sub in (("early_half", sel[:half]), ("late_half", sel[half:])):
            t2, c2 = pooled(sub)
            p2 = [gated_peak(*binned(t2, c2, nb))[0] for nb in (10, 12, 16)]
            jk[tag] = spread(p2)[2]
        jk_rel = (abs(jk["late_half"] - jk["early_half"])
                  / max(abs(jk["late_half"]), 1e-12))
        # Declared uncertainty = the stricter of the two committed readings,
        # i.e. max(resolution-band spread, frame half-split).  ``judgeable``
        # below keeps the historical rule (full bin-grid spread only) so that
        # the committed artefacts stay reproducible; m22545 showed that rule
        # disagrees with the stricter one on 2 of 4 grids (see
        # docs/case_sph_wedge_entry_2d_stationarity_m22545.md), so
        # ``judgeable_stationarity`` is the field to read.
        _terms = [v for v in (rb_rel, jk_rel) if v == v]
        unc_declared = max(_terms) if _terms else float("nan")
        by_frames[str(n_frames)] = {
            "n_frames": n_frames, "pooled_samples": len(cps),
            "mean_wet_nodes": statistics.mean(s["n_wet"] for s in sel),
            "rows": rows,
            "peak_min": lo, "peak_max": hi, "peak_median": med,
            "spread_rel": rel,
            "err_vs_wagner_median": (med - ref) / ref if med == med else float("nan"),
            "resolution_band": {"n_bins_range": [rb_lo, rb_hi],
                                "n_bins_used": rb_bins,
                                "band_peaks": rb_band_peaks,
                                "n_members": sum(1 for v in rb_peaks if v == v),
                                "estimable": bool(rb_est),
                                "candidate_grid": "BAND_GRID",
                                "band_truncated": bool(rb_hi > max(BAND_GRID)),
                                "band_clipped_low": bool(rb_lo < min(BAND_GRID)),
                                "peak_min": rb_lo_v, "peak_max": rb_hi_v,
                                "peak_median": rb_med, "spread_rel": rb_rel,
                                "err_vs_wagner_median": (
                                    (rb_med - ref) / ref if rb_med == rb_med
                                    else float("nan"))},
            "jackknife_halves": jk, "jackknife_rel": jk_rel,
            "uncertainty_declared": unc_declared,
            "judgeable": bool(rel == rel and rel < SPREAD_TOL),
            "judgeable_stationarity": bool(unc_declared == unc_declared
                                           and unc_declared < SPREAD_TOL),
        }

    # ---- matched-bandwidth reading (m21677) ---------------------------- #
    # The committed BIN_GRID stops at 30 bins.  A bin one *surface cell* wide
    # (the natural measurement cell) needs n_bins = TAU_MAX c_end / dx, i.e. 60
    # bins at dx = 1 mm and 120 at dx = 0.5 mm, so the two finest grids could
    # never be read at matched bandwidth and were silently over-smoothed.  The
    # block below records that reading for every grid, whatever the grid list
    # is, plus a frame bootstrap (resampling the 61 self-similar frames with
    # replacement) so the number carries its own uncertainty.
    sel = shots[-FRAME_COUNTS[-1]:] if len(shots) > FRAME_COUNTS[-1] else list(shots)
    c_end = C * case["cfg"]["velocity"] * case["t_end"]
    n_cells = TAU_MAX * c_end / case["cfg"]["dx"]
    n_bins_m = max(4, int(round(n_cells)))
    taus, cps = pooled(sel)
    peak_m, tau_m, n_at_m = gated_peak(*binned(taus, cps, n_bins_m))
    boot = []
    n_f = len(sel)
    if n_f >= 8:
        rng = random.Random(BOOT_SEED)
        for _ in range(BOOT_RESAMPLES):
            sample = [sel[rng.randrange(n_f)] for _ in range(n_f)]
            t2, c2 = pooled(sample)
            pk = gated_peak(*binned(t2, c2, n_bins_m))[0]
            if pk == pk:
                boot.append(pk)
    boot_sorted = sorted(boot)
    def _q(f):
        return (boot_sorted[min(len(boot_sorted) - 1, int(round(f * (len(boot_sorted) - 1))))]
                if boot_sorted else float("nan"))
    sweep_bins = sorted({int(n) for n in list(grid) + [n_bins_m // 2, n_bins_m,
                                                       int(round(1.5 * n_bins_m))]})
    matched = {
        "n_cells": n_cells,
        "n_bins_matched": n_bins_m,
        "peak": peak_m, "tau_at_peak": tau_m, "samples_at_peak": n_at_m,
        "err_vs_wagner_point": ((peak_m - ref) / ref if peak_m == peak_m
                                else float("nan")),
        "bootstrap": {"resamples": BOOT_RESAMPLES, "seed": BOOT_SEED,
                      "n_valid": len(boot), "p05": _q(0.05), "p50": _q(0.5),
                      "p95": _q(0.95),
                      "rel_half_width": ((_q(0.95) - _q(0.05)) / 2.0 / _q(0.5)
                                         if _q(0.5) == _q(0.5) and _q(0.5) else
                                         float("nan"))},
        "bandwidth_sweep": [
            {"n_bins": nb,
             "bin_width_cells": TAU_MAX * c_end / nb / case["cfg"]["dx"],
             "cp_peak_gated": gated_peak(*binned(taus, cps, nb))[0]}
            for nb in sweep_bins],
    }

    # ---- finite-area-averaged peak (m21677) ---------------------------- #
    # A point maximum of a square-root-singular field is not a measurable
    # quantity; a transducer reports a finite-area average.  Same operator on
    # both sides (see footprint_peak.py), sliding window so the reading is not
    # an artefact of where the bin edges fall.
    import footprint_peak as _fpm
    fp_rows = _fpm.footprint_peaks(taus, cps, case["cfg"]["dx"], c_end,
                                   FOOTPRINT_CELLS)
    fp_boot = []
    if n_f >= 8:
        rng = random.Random(BOOT_SEED)
        for _ in range(BOOT_RESAMPLES):
            sample = [sel[rng.randrange(n_f)] for _ in range(n_f)]
            t2, c2 = pooled(sample)
            fp_boot.append(_fpm.footprint_peaks(t2, c2, case["cfg"]["dx"], c_end,
                                                FOOTPRINT_CELLS))
    footprint = {"footprints_cells": list(FOOTPRINT_CELLS), "rows": fp_rows,
                 "bootstrap": {str(r["footprint_cells"]):
                               _fpm.footprint_uncertainty(fp_rows,
                                                          r["footprint_cells"],
                                                          fp_boot)
                               for r in fp_rows}}

    # per-frame scan: which frames carry the spike (startup question)
    scan = []
    for shot in shots:
        centres, means, cnts = binned(shot["tau"], shot["cp"], 12)
        peak, tau_at, n_at = gated_peak(centres, means, cnts, min_count=1)
        scan.append({"t": shot["t"], "tau_norm": shot["t"],
                     "n_wet": shot["n_wet"],
                     "cp_peak_single_frame": peak, "tau_at_peak": tau_at,
                     "cp_centre": shot["cp_centre"],
                     "cp_footprint_max": shot["cp_footprint_max"],
                     "max_rho_ratio": shot["max_rho_ratio"]})
    # frame that would relax a particle-scale acoustic transient
    t_ac = ACOUSTIC_CROSSINGS * case["cfg"]["dx"] / case["c0"]
    scan_sorted = sorted(scan, key=lambda r: r["cp_peak_single_frame"], reverse=True)
    return {"by_frames": by_frames, "frame_scan": scan,
            "matched_bandwidth": matched,
            "footprint_peak": footprint,
            "t_acoustic_relax": t_ac,
            "t_end": case["t_end"],
            "acoustic_relax_fraction_of_window": t_ac / case["t_end"],
            "argmax_frame": scan_sorted[0] if scan_sorted else {},
            "wagner_cp_max": ref,
            "wagner_cp_centre": case["wagner_cp_centre"],
            "wagner_tau_peak": case["wagner_tau_peak"],
            "cp_acoustic_scale": 2.0 * case["c0"] / case["cfg"]["velocity"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, default=2.0e-3)
    ap.add_argument("--velocity", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--c0-factor", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bin-grid", default=None,
                    help="comma separated bin counts (default: BIN_GRID)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "case_sph_wedge_entry_2d_ensemble")
    # Persist the per-frame wall samples so the *uncertainty model* of the
    # footprint reading can be audited after the fact (m22416).  The committed
    # artefacts of m21677..m21953 kept only pooled readings, which made the
    # frame bootstrap unfalsifiable: with no per-frame data the independence
    # assumption it rests on could not be checked.  Off by default so every
    # previously committed artefact reproduces bit for bit.
    ap.add_argument("--dump-frames", type=Path, default=None,
                    help="also write per-frame (tau, cp) wall samples to this path")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = collect(args.dx, args.velocity, args.frames, args.c0_factor,
                   args.threads)
    bin_grid = ([int(v) for v in args.bin_grid.split(",") if v.strip()]
                if args.bin_grid else None)
    result = analyse(case, bin_grid)
    payload = {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py",
        "design": {"frame_counts": list(FRAME_COUNTS),
                   "bin_grid": list(bin_grid or BIN_GRID),
                   "min_bin_samples": MIN_BIN_SAMPLES,
                   "spread_tolerance": SPREAD_TOL,
                   "tau_max": TAU_MAX},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform(),
                        "threads": args.threads},
        **result,
        "config": case["cfg"], "steps": case["steps"],
        "wall_seconds": round(case["wall_seconds"], 1),
        "c0": case["c0"], "q": case["q"], "C": case["C"],
    }
    tag = f"dx{case['cfg']['dx']*1e3:g}mm_c0v{case['cfg']['c0_factor']:g}"
    path = out_dir / f"ensemble_{tag}.json"
    path.write_text(json.dumps(payload, indent=1))
    if args.dump_frames is not None:
        dump = Path(args.dump_frames)
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(json.dumps({
            "case": "CASE-SPH-WEDGE-ENTRY-2D",
            "script": "benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py",
            "tag": tag, "dx": case["cfg"]["dx"], "t_end": case["t_end"],
            "C": case["C"], "q": case["q"],
            "frames": [{"t": s["t"], "n_wet": s["n_wet"],
                        "max_rho_ratio": s["max_rho_ratio"],
                        "tau": s["tau"], "cp": s["cp"]} for s in case["shots"]],
        }, indent=1))
        print(f"[ensemble] frame samples -> {dump}")
    print(f"[ensemble] {tag}: steps={case['steps']} wall={case['wall_seconds']:.1f}s"
          f" -> {path}")
    for key, row in result["by_frames"].items():
        print(f"  frames={key:>2}  peak median={row['peak_median']:.3f}"
              f"  spread={row['spread_rel']*100:.1f}%"
              f"  err_vs_wagner={row['err_vs_wagner_median']*100:+.1f}%"
              f"  judgeable={row['judgeable']}")
    top = result.get("argmax_frame") or {}
    if top:
        print(f"  hottest frame: t={top['t']:.5f}s ({top['t']/case['t_end']*100:.0f}% of window)"
              f" cp_peak_single_frame={top['cp_peak_single_frame']:.2f} n_wet={top['n_wet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
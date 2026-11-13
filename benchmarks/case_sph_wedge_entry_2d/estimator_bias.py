"""Same-operator comparison: what does the peak-Cp *estimator* do to a known field?

Context (measured, not assumed)
-------------------------------
Three earlier rounds on CASE-SPH-WEDGE-ENTRY-2D established

* m17548: the peak Cp is not judgeable at ``dx = 2 mm`` because the binned
  profile peak swings by 3.0x over the bin-count grid;
* m17899: pooling 61 self-similar frames under a sample gate moves the dx = 2 mm
  reading from ``+234 %`` to ``-30.4 %`` and the dx = 4 mm reading to ``+23.3 %``
  -- i.e. the same estimator lands on *opposite* sides of the analytic value on
  the two grids, so the peak was declared not judgeable and the case kept
  ``WITHHELD``.

Both rounds compared a *measured* number against the *point value* of Wagner's
profile.  Those are not the same functional: the estimator (a) reads wall nodes
spaced ``dx`` apart, (b) reads an Adami kernel extrapolation of the field, not a
point value, and (c) takes a maximum over bin means.  Comparing a point value
with a discretised estimator therefore mixes estimator bias with physics error.
This script measures the estimator bias *for a known field*, with no parameter
tuned to the shoreline data, and then re-does the comparison in the same
operator on both sides (reference and simulation).

Predeclared model (constants below, fixed before the first run)
---------------------------------------------------------------
The forward model of the measurement chain is built only from quantities the
solver already declares:

* wall nodes on the first body layer, ``|x_k| = dx ((k+1/2) cos b - (1/2) sin b)``
  (``wedge_lattice`` in ``src/tensorsph/wedge_entry.py``);
* a wet node is one with ``|x_k| <= c(t)``, ``c = C V t`` (the estimator
  normalises tau by the analytic ``c``, so the same ``c`` is used here);
* the field seen by a node is Adami's kernel extrapolation,
  ``p_w = sum_f p_f W / sum_f W`` with the solver's own 2-D cubic spline and
  ``h = 1.3 dx`` (``WedgeConfig``), summed over the fluid lattice rows inside
  ``2h``;
* the truth is Wagner (2) from ``tensorsph.wedge_entry.wagner_p_over_q`` (the
  same function the campaign uses as its reference), sampled at the fluid
  particles' abscissae;
* the estimator is a line-for-line copy of ``ensemble_peak.py``
  (``TAU_MAX = 1.2``, sample gate 4, peak over bins with ``tau <= 1``).

Hypotheses (declared before running)
------------------------------------
H1: the estimator bias is large enough to explain the cross-grid sign flip, i.e.
    applying the *same* estimator to the analytic field reproduces the sign flip
    between dx = 4 mm and dx = 2 mm.
H2 (H1 falsified): the analytic field through the same estimator is grid
    consistent, so the sign flip is a property of the SPH data, not of the
    estimator.

The verdict on the case is then read off the *same-operator residual*
``(obs - synth)/synth``, which is what actually competes with the 15 % gate.

Usage
-----
::

    python benchmarks/case_sph_wedge_entry_2d/estimator_bias.py \
        --artifacts docs/artifacts/case_sph_wedge_entry_2d_ensemble \
        --out docs/artifacts/case_sph_wedge_entry_2d_bias

Reads only committed SPH aggregates (``by_frames`` rows + per-frame ``n_wet``);
runs no SPH step, so it costs seconds and needs no cluster.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.wedge_entry import (  # noqa: E402
    wagner_c_coeff,
    wagner_p_over_q,
    wagner_peak_over_q,
    wagner_validity_tau,
)

# --- predeclared estimator design (copy of ensemble_peak.py) ---------------- #
FRAME_COUNTS = (7, 13, 25, 41, 61)
BIN_GRID = (6, 8, 10, 12, 16, 20, 30)
MIN_BIN_SAMPLES = 4
TAU_MAX = 1.2
TAU_HI = 1.0
SPREAD_TOL = 0.15

# --- predeclared measurement-operator model --------------------------------- #
H_OVER_DX = 1.3          # solver support radius / particle spacing (WedgeConfig)
FLUID_ROWS = 3           # fluid lattice rows inside 2h = 2.6 dx (support cutoff)
NOISE_REL = 0.30         # declared per-node relative scatter for the noise model
NOISE_SEEDS = 200
BETA = 30.0
VELOCITY = 1.0
FRAMES_TOTAL = 61

SUPPORT_HALF = 2.0       # kernel support in units of h
SIGMA_2D = 10.0 / (7.0 * math.pi)   # 2-D cubic spline normalisation (x 1/h^2)


def kernel_2d(r: float, h: float) -> float:
    """2-D cubic spline kernel, identical shape/normalisation to the solver."""
    q = r / h
    if q < 1.0:
        shape = 1.0 - 1.5 * q * q + 0.75 * q ** 3
    elif q < 2.0:
        shape = 0.25 * (2.0 - q) ** 3
    else:
        return 0.0
    return SIGMA_2D * shape / (h * h)


def wall_abscissae(dx: float, beta: float, n_nodes: int) -> np.ndarray:
    """``|x|`` of the first-layer wall nodes of ``wedge_lattice`` (one side)."""
    s = (np.arange(n_nodes, dtype=float) + 0.5) * dx
    n_in = 0.5 * dx
    return s * math.cos(beta) - n_in * math.sin(beta)


def smoothed_node_values(tau_nodes: np.ndarray, c: float, dx: float,
                         beta: float, field=None) -> np.ndarray:
    """Adami kernel extrapolation of the analytic field at the wall nodes.

    Fluid particles sit on the solver lattice below the face,
    ``s_j = (j + 1/2) dx`` along the face and ``(i + 1/2) dx`` away from it, for
    ``i = 0 .. FLUID_ROWS-1``; each carries the analytic ``p/q`` at its own
    abscissa.  The node value is ``sum p W / sum W`` over that stencil, which is
    the operator ``WedgeImpact`` applies (Adami et al. 2012, zero gravity and a
    non-accelerating wall in the body frame).
    """
    if field is None:
        field = lambda t: float(wagner_p_over_q(t, beta))  # noqa: E731
    h = H_OVER_DX * dx
    x_nodes = tau_nodes * c
    # node position in the face frame: invert ``|x| = s cos(b) - (1/2) dx sin(b)``
    s_nodes = (x_nodes + 0.5 * dx * math.sin(beta)) / math.cos(beta)
    span = int(math.ceil(SUPPORT_HALF * H_OVER_DX)) + 1
    js = np.arange(-span, span + 1)
    vals = np.zeros_like(tau_nodes)
    for idx, (x, s_n) in enumerate(zip(x_nodes, s_nodes)):
        num = den = 0.0
        for i in range(FLUID_ROWS):
            n_f = -(i + 0.5) * dx                       # fluid side of the face
            for j in js:
                s_f = s_n + j * dx
                if s_f <= 0.0:
                    continue
                x_f = abs(s_f * math.cos(beta) + n_f * math.sin(beta))
                r = math.hypot(j * dx, n_f)             # node -> fluid particle
                w = kernel_2d(r, h)
                if w <= 0.0:
                    continue
                num += float(field(x_f / c)) * w
                den += w
        vals[idx] = num / den if den > 0.0 else float("nan")
    return vals


def frame_samples(dx: float, t: float, beta: float, use_smoothing: bool):
    """``(tau, cp)`` of the analytically wet wall nodes at time ``t``."""
    C = wagner_c_coeff(beta)
    c = C * VELOCITY * t
    if c <= 0.0:
        return np.zeros(0), np.zeros(0)
    n_nodes = int(math.floor(c / dx)) + 3
    x = wall_abscissae(dx, beta, n_nodes)
    tau = x / c
    wet = (x <= c) & (tau <= wagner_validity_tau(beta))
    tau_w = tau[wet]
    if tau_w.size == 0:
        return tau_w, np.zeros(0)
    if use_smoothing:
        cp = smoothed_node_values(tau_w, c, dx, beta)
    else:
        cp = np.array([float(wagner_p_over_q(s, beta)) for s in tau_w])
    # ``wedge_lattice`` lays *both* faces (+/-); tau is an ``|x|`` so the two
    # sides are identical samples and double the per-bin sample count, which
    # matters for the 4-sample gate.  Counted explicitly, not by assumption.
    return np.concatenate((tau_w, tau_w)), np.concatenate((cp, cp))


def _bin_index(tau: np.ndarray, n_bins: int) -> np.ndarray:
    """``k = min(int(tau / TAU_MAX * n_bins), n_bins - 1)`` for valid samples."""
    k = np.minimum((tau / TAU_MAX * n_bins).astype(int), n_bins - 1)
    ok = np.isfinite(tau) & (tau >= 0.0) & (tau <= TAU_MAX)
    return k[ok]


def estimator_peak(taus: np.ndarray, cps: np.ndarray, n_bins: int):
    """Pooled-profile peak, copy of ``ensemble_peak.binned`` + ``gated_peak``."""
    k = _bin_index(taus, n_bins)
    cnts = np.bincount(k, minlength=n_bins)
    sums = np.bincount(k, weights=cps[np.isfinite(taus) & (taus >= 0.0)
                                      & (taus <= TAU_MAX)], minlength=n_bins)
    width = TAU_MAX / n_bins
    centres = (np.arange(n_bins) + 0.5) * width
    means = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    use = np.isfinite(means) & (centres <= TAU_HI) & (cnts >= MIN_BIN_SAMPLES)
    if not use.any():
        return float("nan"), float("nan")
    idx = np.flatnonzero(use)[np.argmax(means[use])]
    return float(means[idx]), float(centres[idx])


def spread(values):
    """``(min, max, median, (max-min)/|median|)``; NaN-safe (as ensemble_peak)."""
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    lo, hi, med = min(vals), max(vals), statistics.median(vals)
    return lo, hi, med, (hi - lo) / abs(med) if med else float("nan")


def frames(t_end: float, n: int):
    """The last ``n`` of the case's ``FRAMES_TOTAL`` output times."""
    return [t_end * k / (FRAMES_TOTAL - 1) for k in range(FRAMES_TOTAL)][-n:]


class SynthChain:
    """Cached sampling chain: one pass over frames, then cheap re-estimation."""

    def __init__(self, dx: float, n_frames: int, t_end: float,
                 use_smoothing: bool):
        self.dx = dx
        self.beta = math.radians(BETA)
        self.frames = []
        for t in frames(t_end, n_frames):
            self.frames.append(frame_samples(dx, t, self.beta, use_smoothing))

    def estimate(self, noise: float = 0.0, rng: random.Random | None = None) -> dict:
        taus, cps = [], []
        for tau, cp in self.frames:
            if noise > 0.0 and rng is not None and cp.size:
                eps = np.array([rng.gauss(0.0, 1.0) for _ in range(cp.size)])
                cp = cp * (1.0 + noise * eps)
            taus.append(tau)
            cps.append(cp)
        tau_all = np.concatenate(taus) if taus else np.zeros(0)
        cp_all = np.concatenate(cps) if cps else np.zeros(0)
        rows = []
        for n_bins in BIN_GRID:
            peak, tau_at = estimator_peak(tau_all, cp_all, n_bins)
            rows.append({"n_bins": n_bins, "cp_peak_gated": peak,
                         "tau_at_peak": tau_at})
        lo, hi, med, rel = spread([r["cp_peak_gated"] for r in rows])
        n_wet = [len(t) for t, _ in self.frames]
        return {"rows": rows, "peak_min": lo, "peak_max": hi,
                "peak_median": med, "spread_rel": rel,
                "pooled_samples": int(cp_all.size),
                "mean_wet_nodes": statistics.mean(n_wet) if n_wet else 0.0}


def synth_estimates(dx: float, n_frames: int, t_end: float, use_smoothing: bool,
                    noise: float = 0.0, seed: int | None = None):
    chain = SynthChain(dx, n_frames, t_end, use_smoothing)
    return chain.estimate(noise=noise, rng=random.Random(seed) if seed is not None else None)


def apex_check(dx: float, t_end: float, beta: float, artifact: dict,
               n_frames: int = 30) -> dict:
    """Model-vs-data check where the operator is *provably* unbiased.

    At ``tau -> 0`` Wagner's profile is flat (``dCp/dtau = 0``), so kernel
    extrapolation and particle sampling cannot bias the value: the analytic
    point value, the smoothed model and the measured ``cp_centre`` must all
    agree unless the *solver* has a scale error.  Whatever gap is left is a
    solver error, not a measurement-operator artefact -- this is the calibration
    that licenses reading the peak residual at ``tau = 0.93`` as physics.
    """
    scan = artifact["frame_scan"]
    tail = scan[-n_frames:] if len(scan) >= n_frames else scan
    obs = [r["cp_centre"] for r in tail if r["cp_centre"] == r["cp_centre"]
           and r["n_wet"] > 0]
    model_vals = []
    for t in frames(t_end, len(tail)):
        tau, cp = frame_samples(dx, t, beta, True)
        if tau.size and cp.size:
            model_vals.append(float(cp[int(np.argmin(tau))]))
    point = float(wagner_p_over_q(0.0, beta))
    obs_mean = statistics.mean(obs) if obs else float("nan")
    mod_mean = statistics.mean(model_vals) if model_vals else float("nan")
    return {"wagner_point": point, "model_smoothed_apex": mod_mean,
            "obs_apex_mean": obs_mean,
            "model_vs_point": mod_mean / point - 1.0 if mod_mean == mod_mean else float("nan"),
            "obs_vs_point": obs_mean / point - 1.0 if obs_mean == obs_mean else float("nan"),
            "n_frames_used": len(obs)}


def frame_peak_stats(artifact: dict) -> dict:
    """Distribution of the committed per-frame single-frame peak (12 bins, ungated).

    Quantifies how far the *typical* frame sits from the operator-expected peak:
    if the median frame peak already exceeds it, the excess is not an extreme
    value of the pooled maximum but a property of the trajectory.
    """
    scan = [r["cp_peak_single_frame"] for r in artifact["frame_scan"]]
    vals = sorted(v for v in scan if v == v and v > 0.0)
    if not vals:
        return {"n_frames_used": 0}

    def q(f):
        return vals[min(len(vals) - 1, int(round(f * (len(vals) - 1))))]

    return {"n_frames_used": len(vals), "min": vals[0], "p25": q(0.25),
            "median": q(0.5), "p75": q(0.75), "max": vals[-1]}


def validate_sampling_model(dx: float, t_end: float, artifact: dict) -> dict:
    """Model wet-node count vs the committed per-frame ``n_wet``."""
    scan = artifact["frame_scan"]
    model = [len(frame_samples(dx, t, math.radians(BETA), True)[0]) for t in
             [t_end * k / (FRAMES_TOTAL - 1) for k in range(FRAMES_TOTAL)]]
    obs = [r["n_wet"] for r in scan]
    n = min(len(model), len(obs))
    diffs = [m - o for m, o in zip(model[:n], obs[:n])]
    return {"per_frame_model": model[:n], "per_frame_obs": obs[:n],
            "max_abs_diff": max((abs(d) for d in diffs), default=0),
            "mean_abs_diff": statistics.mean(abs(d) for d in diffs) if diffs else 0,
            "obs_monotone_trend_note": "n_wet grows with t (wetted width c=C V t)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ROOT / "docs/artifacts/"
                                               "case_sph_wedge_entry_2d_ensemble"))
    ap.add_argument("--out", default=str(ROOT / "docs/artifacts/"
                                        "case_sph_wedge_entry_2d_bias"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    beta = math.radians(BETA)
    ref_point = wagner_peak_over_q(beta)
    report = {"case": "CASE-SPH-WEDGE-ENTRY-2D",
              "script": "benchmarks/case_sph_wedge_entry_2d/estimator_bias.py",
              "design": {"frame_counts": list(FRAME_COUNTS),
                         "bin_grid": list(BIN_GRID),
                         "min_bin_samples": MIN_BIN_SAMPLES,
                         "tau_max": TAU_MAX, "tau_hi": TAU_HI,
                         "h_over_dx": H_OVER_DX, "fluid_rows": FLUID_ROWS,
                         "noise_rel": NOISE_REL, "noise_seeds": NOISE_SEEDS,
                         "beta_deg": BETA, "velocity": VELOCITY},
              "wagner_cp_max_point": ref_point,
              "grids": {}}

    for grid in ("dx2mm", "dx4mm"):
        art_path = Path(args.artifacts) / f"ensemble_{grid}_c0v20.json"
        art = json.loads(art_path.read_text())
        dx = art["config"]["dx"]
        t_end = art["t_end"]
        entry = {"dx": dx, "t_end": t_end,
                 "sampling_model_check": validate_sampling_model(dx, t_end, art),
                 "apex_check": apex_check(dx, t_end, beta, art),
                 "frame_peak_stats": frame_peak_stats(art),
                 "by_frames": {}}
        for n_frames in FRAME_COUNTS:
            obs = art["by_frames"].get(str(n_frames))
            if obs is None:
                continue
            chain_s = SynthChain(dx, n_frames, t_end, True)
            chain_p = SynthChain(dx, n_frames, t_end, False)
            smooth = chain_s.estimate()
            point = chain_p.estimate()
            noisy = [chain_s.estimate(noise=NOISE_REL, rng=random.Random(1000 + k))["peak_median"]
                     for k in range(NOISE_SEEDS)]
            nlo, nhi, nmed, nrel = spread(noisy)
            obs_med = obs["peak_median"]
            rows = []
            for o_row, s_row, p_row in zip(obs["rows"], smooth["rows"],
                                           point["rows"]):
                s_val = s_row["cp_peak_gated"]
                o_val = o_row["cp_peak_gated"]
                rows.append({
                    "n_bins": o_row["n_bins"],
                    "obs_cp_peak": o_val,
                    "synth_cp_peak_smoothed": s_val,
                    "synth_cp_peak_point": p_row["cp_peak_gated"],
                    "obs_vs_point_ref": (o_val - ref_point) / ref_point,
                    "obs_vs_same_operator": ((o_val - s_val) / s_val
                                             if s_val == s_val and s_val else float("nan")),
                })
            entry["by_frames"][str(n_frames)] = {
                "n_frames": n_frames,
                "obs_peak_median": obs_med,
                "obs_spread_rel": obs["spread_rel"],
                "obs_err_vs_point_ref": (obs_med - ref_point) / ref_point,
                "synth_smoothed": smooth,
                "synth_point": point,
                "synth_noisy_median_band": {"min": nlo, "max": nhi,
                                            "median": nmed, "spread_rel": nrel},
                "same_operator_residual_median": (
                    (obs_med - smooth["peak_median"]) / smooth["peak_median"]
                    if smooth["peak_median"] == smooth["peak_median"]
                    and smooth["peak_median"] else float("nan")),
                "rows": rows,
            }
        report["grids"][grid] = entry

    # ---- hypothesis test H1/H2 ------------------------------------------- #
    signs = {}
    for grid, entry in report["grids"].items():
        last = entry["by_frames"].get(str(FRAME_COUNTS[-1]))
        if last is None:
            continue
        signs[grid] = {"obs_err": last["obs_err_vs_point_ref"],
                       "synth_err": last["synth_smoothed"]["peak_median"] / ref_point - 1.0,
                       "same_operator_residual": last["same_operator_residual_median"]}
    flip_obs = (signs["dx2mm"]["obs_err"] < 0.0) != (signs["dx4mm"]["obs_err"] < 0.0)
    flip_synth = (signs["dx2mm"]["synth_err"] < 0.0) != (signs["dx4mm"]["synth_err"] < 0.0)
    report["hypothesis"] = {
        "signs": signs,
        "obs_has_sign_flip_between_grids": flip_obs,
        "synth_has_sign_flip_between_grids": flip_synth,
        "H1_reproduced_by_estimator_alone": bool(flip_obs and flip_synth),
        "verdict": ("H1: the estimator alone reproduces the cross-grid sign flip"
                    if (flip_obs and flip_synth) else
                    "H2: the analytic field is grid-consistent through the same "
                    "estimator, so the observed sign flip is a data property"),
    }

    # ---- verdict: same-operator residual vs the 15 % gate ---------------- #
    verdict = {}
    for grid, entry in report["grids"].items():
        last = entry["by_frames"][str(FRAME_COUNTS[-1])]
        res = last["same_operator_residual_median"]
        sp = last["synth_smoothed"]["spread_rel"]
        verdict[grid] = {
            "residual": res,
            "absent_bias_spread": sp,
            "within_15pct": bool(abs(res) <= SPREAD_TOL),
            "grid_consistent_with_other_grid": None,
        }
    if "dx2mm" in verdict and "dx4mm" in verdict:
        r2, r4 = verdict["dx2mm"]["residual"], verdict["dx4mm"]["residual"]
        diff = abs(r2 - r4)
        verdict["dx2mm"]["grid_consistent_with_other_grid"] = bool(diff <= SPREAD_TOL)
        verdict["dx4mm"]["grid_consistent_with_other_grid"] = bool(diff <= SPREAD_TOL)
        verdict["cross_grid_residual_gap"] = diff
        # indicative only: |e(4mm)| / |e(2mm)| = 2^p
        if r2 and r4:
            verdict["observed_order_indicative"] = math.log2(abs(r4) / abs(r2))

    report["verdict"] = verdict

    text = json.dumps(report, indent=1, sort_keys=False)
    (out_dir / "bias_report.json").write_text(text)
    print(text[:200])
    print(f"\nwrote {out_dir / 'bias_report.json'}")
    print(json.dumps(report["hypothesis"], indent=1))
    print(json.dumps(report["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
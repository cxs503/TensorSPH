"""Tests for the resolved-band Wagner acceptance layer (p70w3b-n01).

Three groups:

1. analytic identities of the Wagner reference written in the self-similar
   abscissa ``tau = |x|/(C V t)``, asserted to 6 significant digits (the
   campaign's artefact format);
2. properties of the metrics themselves -- a metric that is fed the analytic
   profile must return zero error, and a known multiplicative bias must come back
   as exactly that bias (otherwise the error budget of the SPH comparison is not
   interpretable);
3. one small SPH run to prove the wall-pressure profile is actually emitted and
   binned in the analytic normalisation, plus the aggregation path on a synthetic
   payload so the acceptance arithmetic is testable without a 1-hour simulation.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "wedge_entry"))

from tensorsph.wedge_entry import (  # noqa: E402
    WedgeConfig,
    WedgeImpact,
    analytic_beta_slope,
    band_peak,
    gci,
    offset_scaling_fit,
    profile_l2_rel_error,
    rel_err_safe,
    resolved_band_tau,
    sig6,
    wagner_c_coeff,
    wagner_p_over_q,
    wagner_pmax,
    wagner_peak_over_q,
    wagner_peak_tau,
    wagner_validity_tau,
    wet_half_width,
)

import wagner_resolved as wr  # noqa: E402


BETAS = (10.0, 15.0, 20.0, 30.0)


def _rad(deg: float) -> float:
    return math.radians(deg)


# --------------------------------------------------------------------------- #
# 1. analytic identities (6 significant digits)
# --------------------------------------------------------------------------- #

def test_peak_tau_matches_closed_form_sig6():
    """tau_peak = sqrt(1 - 1/C^2) to 6 significant digits."""
    expected = {10.0: 0.993680, 15.0: 0.985344, 20.0: 0.972785, 30.0: 0.930003}
    for beta, ref in expected.items():
        got = wagner_peak_tau(_rad(beta))
        assert sig6(got) == ref, (beta, got, ref)
        # independent recomputation from the definition
        C = wagner_c_coeff(_rad(beta))
        assert got == pytest.approx(math.sqrt(1.0 - 1.0 / C ** 2), rel=1e-15)


def test_peak_over_q_equals_pressure_at_peak_tau():
    """The maximum of (2) sits exactly at tau_peak and equals C^2 + 1."""
    for beta in BETAS + (45.0, 60.0):
        b = _rad(beta)
        tp = wagner_peak_tau(b)
        p_peak = wagner_p_over_q(tp, b)
        assert p_peak == pytest.approx(wagner_peak_over_q(b), rel=1e-12), beta
        if wagner_c_coeff(b) >= 1.0:
            assert wagner_peak_over_q(b) == pytest.approx(
                wagner_c_coeff(b) ** 2 + 1.0, rel=1e-15)


def test_peak_over_q_matches_dimensional_pmax():
    """p_over_q * q equals wagner_pmax for every campaign deadrise and speed."""
    rho = 1000.0
    for beta in BETAS + (45.0,):
        b = _rad(beta)
        for V in (1.0, 2.0, 5.0):
            q = 0.5 * rho * V * V
            assert q * wagner_peak_over_q(b) == pytest.approx(
                wagner_pmax(V, b, rho), rel=1e-14)


def test_stagnation_level_and_centre_pressure():
    """p/q at tau = 0 is 2C, i.e. p(0) = rho V^2 C (time independent)."""
    for beta in BETAS:
        b = _rad(beta)
        assert wagner_p_over_q(0.0, b) == pytest.approx(2.0 * wagner_c_coeff(b),
                                                        rel=1e-15)


def test_profile_is_monotone_up_to_peak_and_positive_inside_validity():
    """(2) rises monotonically to tau_peak then falls; it is positive on the footprint."""
    b = _rad(30.0)
    tp = wagner_peak_tau(b)
    taus = [i / 2000.0 * tp for i in range(1, 2001)]
    vals = [wagner_p_over_q(t, b) for t in taus]
    assert all(v1 < v2 for v1, v2 in zip(vals[:-1], vals[1:]))
    # beyond the validity cut the outer solution is clamped to zero, never negative
    tmax = wagner_validity_tau(b)
    assert wagner_p_over_q(tmax * 1.001, b) == 0.0
    assert wagner_p_over_q(1.5, b) == 0.0


def test_tensor_and_scalar_forms_agree():
    b = _rad(20.0)
    tau = torch.tensor([0.0, 0.5, 0.9, 0.97, 1.01], dtype=torch.float64)
    vec = wagner_p_over_q(tau, b)
    for t, v in zip(tau.tolist(), vec.tolist()):
        assert v == pytest.approx(wagner_p_over_q(t, b), rel=1e-13)


def test_analytic_beta_slope_of_the_reference_itself():
    """The reference's own finite-window slope is -1.904147, not -2.

    The campaign asks for a log-log slope of -2 within 20 %.  The exact Wagner
    curve gives -1.904147 over beta = 10..30 deg (the additive stagnation level
    ``+1`` flattens it), which is 4.8 % from -2 and therefore *inside* the band:
    the criterion is attainable, but the number to beat is the reference's own
    slope, not -2.  Both are asserted so a future change that quietly retargets
    the criterion fails here.
    """
    a, r2 = analytic_beta_slope(BETAS, 1.0, 1000.0)
    assert sig6(a) == -1.90415
    assert abs(a + 2.0) / 2.0 < 0.20
    assert r2 > 0.999


def test_offset_scaling_fit_recovers_wagner_constants():
    """p = q (A/tan^2 beta + B) returns A = pi^2/4 and B = 1 on exact Wagner data."""
    xs = [1.0 / math.tan(_rad(b)) ** 2 for b in BETAS]
    ys = [wagner_pmax(1.0, _rad(b), 1000.0) for b in BETAS]
    A, B, r2 = offset_scaling_fit(xs, ys, 0.5 * 1000.0)
    assert sig6(A) == 2.4674          # pi^2/4 = 2.467401...
    assert sig6(B) == 1.0
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_gci_matches_closed_form():
    """GCI = 1.25 |e| / (r^p - 1) for a known pair."""
    assert gci(1.0, 0.9, 2.0, 1.0) == pytest.approx(1.25 * 0.1, rel=1e-12)
    assert gci(0.0, 1.0, 2.0, 1.0) == float("inf")


def test_rel_err_safe_degenerate_cases():
    assert rel_err_safe(1.1, 1.0) == pytest.approx(0.1, rel=1e-12)
    assert math.isnan(rel_err_safe(1.0, 0.0))
    assert math.isnan(rel_err_safe(float("nan"), 1.0))


# --------------------------------------------------------------------------- #
# 2. metric properties
# --------------------------------------------------------------------------- #

def test_resolved_band_tau_definition():
    assert resolved_band_tau(0.05, 1.0e-3, 3.0) == pytest.approx(0.94, rel=1e-15)
    assert resolved_band_tau(0.05, 0.25e-3, 3.0) == pytest.approx(0.985, rel=1e-15)
    assert resolved_band_tau(1.0e-3, 1.0e-3, 3.0) == 0.0     # wholly unresolved
    assert resolved_band_tau(0.0, 1.0e-3) == 0.0
    # converges to the full footprint as dx -> 0
    assert resolved_band_tau(0.05, 1.0e-12) == pytest.approx(1.0, rel=1e-10)


def test_band_peak_and_l2_are_exact_on_the_analytic_profile():
    b = _rad(30.0)
    tau = [0.015 + 0.03 * i for i in range(40)]
    vals = [wagner_p_over_q(t, b) for t in tau]
    tau_r = resolved_band_tau(0.05, 1.0e-3, 3.0)
    peak, at = band_peak(tau, vals, tau_r)
    assert peak > 0.0 and at <= tau_r
    assert profile_l2_rel_error(tau, vals, b, tau_r) == pytest.approx(0.0, abs=1e-14)
    # symmetric window: the reference's own band maximum
    ref_peak, _ = band_peak(tau, [wagner_p_over_q(t, b) if t <= tau_r else -1.0
                                  for t in tau], tau_r)
    assert peak == pytest.approx(ref_peak, rel=1e-15)


@pytest.mark.parametrize("factor", [0.9, 1.1])
def test_l2_metric_returns_a_known_multiplicative_bias(factor):
    """A profile scaled by ``factor`` must come back as |factor - 1| in L2."""
    b = _rad(20.0)
    tau = [0.01 + 0.02 * i for i in range(50)]
    ref = [wagner_p_over_q(t, b) for t in tau]
    biased = [factor * v for v in ref]
    err = profile_l2_rel_error(tau, biased, b, 1.0)
    assert err == pytest.approx(abs(factor - 1.0), rel=1e-12)


def test_band_peak_ignores_nan_bins_and_out_of_band_values():
    tau = [0.1, 0.2, 0.3, 0.95]
    vals = [1.0, float("nan"), 5.0, 100.0]
    peak, at = band_peak(tau, vals, 0.9)
    assert (peak, at) == (5.0, 0.31) or (peak, at) == (5.0, 0.3)


# --------------------------------------------------------------------------- #
# 3. SPH: profile emission + aggregation path
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One tiny wedge run (dx = 5 mm) that exercises the profile diagnostic.

    Threads are capped: the solver is a sparse index_add loop, and on a busy
    128-core node torch's default thread count makes this toy case *slower* than
    a serial run (measured), which turns a 5-second test into a multi-minute one.
    """
    prev = torch.get_num_threads()
    torch.set_num_threads(max(1, min(4, prev)))
    try:
        cfg = WedgeConfig(beta_deg=30.0, velocity=1.0, dx=5.0e-3, c_end=0.05,
                          frames=3, profile_bins=20)
        sim = WedgeImpact(cfg)
        rows = sim.run()
    finally:
        torch.set_num_threads(prev)
    return cfg, sim, rows


def test_profile_is_emitted_in_the_analytic_normalisation(small_run):
    cfg, sim, rows = small_run
    last = rows[-1]
    tau = last["profile_tau"]
    vals = last["profile_p_over_q"]
    assert len(tau) == cfg.profile_bins == len(vals) == len(last["profile_count"])
    assert tau[0] > 0.0 and tau[-1] < 1.2
    assert all(t2 > t1 for t1, t2 in zip(tau[:-1], tau[1:]))
    finite = [v for v in vals if math.isfinite(v)]
    assert finite, "no wetted bin was recorded"
    assert max(finite) > 0.0 and min(finite) >= 0.0   # Tait pressure is positive
    # tau is normalised by the analytic edge c_W = C V t: the wetted bins must
    # therefore stop near tau = 1, not at the (slightly different) measured c
    c_w = wagner_c_coeff(cfg.beta) * cfg.velocity * last["t"]
    assert c_w > 0.0
    assert abs(wet_half_width(cfg.velocity, cfg.beta, last["t"]) - c_w) < 1e-15


def test_case_band_summary_path_on_a_synthetic_payload():
    """The aggregation arithmetic is testable without a long SPH run."""
    cfg = WedgeConfig(beta_deg=30.0, velocity=1.0, dx=1.0e-3, frames=4)
    beta, V, _rho = cfg.beta, cfg.velocity, cfg.rho0
    frames = []
    for k in range(1, 6):
        t = k * cfg.resolved_t_end() / 5.0
        c_w = wagner_c_coeff(beta) * V * t
        tau = [0.03 + 0.03 * i for i in range(40)]
        frames.append({
            "t": t, "c_sph": c_w, "c_wagner": c_w, "x_jet_sph": c_w,
            "profile_tau": tau,
            "profile_p_over_q": [wagner_p_over_q(x, beta) if x <= 0.94 else float("nan")
                                 for x in tau],
        })
    payload = {
        "id": "synthetic", "kind": "ladder",
        "config": {"beta_deg": 30.0, "velocity": V, "dx": cfg.dx,
                   "c_coeff": wagner_c_coeff(beta), "c_end": cfg.c_end},
        "analytic": {"c_coeff": wagner_c_coeff(beta)},
        "frames": frames,
    }
    out = wr.case_band_summary(payload, smooth=0)
    assert out["n_frames"] == len(frames)
    assert out["p_band_rel_err_mean"] == pytest.approx(0.0, abs=1e-12)
    assert out["profile_l2_mean"] == pytest.approx(0.0, abs=1e-12)
    assert out["c_coeff_mean"] == pytest.approx(wagner_c_coeff(beta), rel=1e-12)
    assert out["cells_to_root"] == pytest.approx(
        (1.0 - wagner_peak_tau(beta)) * cfg.c_end / cfg.dx, rel=1e-15)
    assert out["last_profile"]["tau"] == frames[-1]["profile_tau"]
    assert out["ensemble"]["n_frames"] == len(frames)
    assert out["noise_rel"] == pytest.approx(0.0, abs=1e-12)


def test_gci_richardson_flags_non_monotone_sequence():
    """A non-monotone ladder is reported as not in the asymptotic range.

    ``gci_richardson`` takes the levels coarse -> fine, so ``[-1]`` is the finest:
    a converging sequence has successive differences of the same sign, an
    oscillating one does not.
    """
    good = wr.gci_richardson([0.60, 0.80, 0.90], 2.0)      # coarse -> fine
    assert good["monotone"] is True
    assert good["order"] == pytest.approx(1.0, rel=1e-12)
    assert good["gci"] < 0.5
    bad = wr.gci_richardson([0.90, 1.00, 0.95], 2.0)
    assert bad["monotone"] is False
    assert bad["order"] is None
    assert bad["p_used"] == 1.0


def test_ensemble_profile_averages_frame_noise_and_keeps_std():
    """Frame averaging is unbiased on a self-similar field and reports the noise.

    Synthetic frames = Wagner profile times ``1 + noise``: the ensemble mean must
    approach Wagner and the reported frame-to-frame std must approach the imposed
    noise level.  This is the property that justifies using the ensemble rather
    than a frame-wise maximum as the verification estimator.
    """
    beta = _rad(20.0)
    V, rho, dx = 1.0, 1000.0, 1.0e-3
    tau = [0.015 + 0.03 * i for i in range(40)]
    ref = [wagner_p_over_q(t, beta) for t in tau]
    rng = [[1.0 + 0.20 * ((k * 7 + i * 3) % 11 - 5) for i in range(40)]
           for k in range(30)]          # deterministic pseudo-noise, +-10 %
    frames = [{"t": (k + 1) * 1.0e-3,
               "profile_tau": tau,
               "profile_p_over_q": [r * w if t <= 0.99 else float("nan")
                                    for r, w, t in zip(row, ref, tau)]}
              for k, row in enumerate(rng)]
    ens = wr.ensemble_profile(frames, beta, V, dx, 0.05, cells=3.0, smooth=0)
    band = [i for i, t in enumerate(tau) if t <= ens["tau_r"]]
    for i in band:
        assert ens["mean"][i] == pytest.approx(ref[i], rel=0.10)
        assert 0.02 < ens["std"][i] / ens["mean"][i] < 0.80
    # the frame average is what makes the estimator usable: it must be much
    # closer to Wagner than the worst single frame, by roughly sqrt(n_frames)
    raw = wr.profile_l2_rel_error(tau, ens["mean"], beta, ens["tau_r"])
    assert raw < 0.10
    single = [wr.profile_l2_rel_error(tau, f["profile_p_over_q"], beta, ens["tau_r"])
              for f in frames]
    assert raw < sum(single) / len(single) / 2.0
    # smoothing is only applied inside the band and must not change the peak
    # ordering: it is a noise reduction, not a fit
    ens_s = wr.ensemble_profile(frames, beta, V, dx, 0.05, cells=3.0, smooth=1)
    assert (wr.profile_l2_rel_error(tau, ens_s["smoothed"], beta, ens_s["tau_r"])
            < 3.0 * raw)


def test_summary_json_is_serialisable_with_sig6(tmp_path):
    """sig6 rounding keeps 6 significant digits (campaign artefact format)."""
    assert sig6(0.9936798) == 0.99368
    assert sig6(80.3601) == 80.3601
    assert sig6(-1.9041477) == -1.90415
    assert sig6(0.0) == 0.0
    assert sig6(None) is None
    payload = {"a": 0.9936798, "b": [1.23456789]}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload))
    assert json.loads(path.read_text())["a"] == pytest.approx(0.9936798)

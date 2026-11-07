"""Calibration tests for the wave-tank measurement chain (the gate numbers).

Criteria (b), (c) and (e) of ``benchmarks/wave_tank/summarise.py`` are not
measurements of the solver; they are readings of three estimators
(``two_probe_dispersion``, ``spatial_wave_number``, ``detrended_reflection``)
applied to probe records. A reading can only be trusted if the same estimator
recovers a *known* answer, so every test here feeds it a closed-form wave field:
``eta = a e^{-alpha x} cos(wt - kx) + R a e^{-alpha x} cos(wt + kx + psi)`` with
``k``, ``alpha``, ``R`` and ``psi`` imposed.

The full sweep (bias envelopes, noise floor, probe layouts) lives in
``benchmarks/wave_tank/calibrate_estimators.py`` and is reported in
``docs/verification_wave_tank_estimators.md``; these are the same claims at a
size that fits in CI.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.wave_tank.calibrate_estimators import (
    DEPTH,
    HEIGHT,
    measure,
    probe_series,
    sample_times,
    wave_number,
)
from benchmarks.wave_tank import run as wave_tank_run
from benchmarks.wave_tank.summarise import case_row
from tensorsph.wave_tank import (
    WaveTankConfig,
    angular_frequency,
    detrended_reflection,
    harmonic_fit,
    spatial_wave_number,
    two_probe_dispersion,
)

KH = 3.0
K = wave_number()
OMEGA = angular_frequency(K, DEPTH)
WAVELENGTH = 2.0 * math.pi / K
PRODUCTION_OFFSETS = (0.4, 0.7, 1.0, 1.3)          # CASES in run.py
DECAY10_OFFSETS = (0.5, 3.0, 6.0, 9.0, 9.4, 9.8, 10.2)


def production_positions(offsets=PRODUCTION_OFFSETS):
    return [offset * WAVELENGTH for offset in offsets]


# --------------------------------------------------------------------------- #
# Positive control: a signal whose answer is known exactly
# --------------------------------------------------------------------------- #

def test_estimators_recover_a_known_decaying_train_to_round_off():
    """Noiseless decaying train: both estimators must return the imposed answer."""
    alpha = 2.0 / (10.0 * WAVELENGTH)               # 20 % loss per wavelength
    positions = production_positions()
    times = sample_times(OMEGA)
    series = probe_series(times, positions, K, OMEGA, alpha=alpha)
    got = measure(positions, series, times, OMEGA)
    assert got["two_probe"]["k_measured"] == pytest.approx(K, rel=1e-9)
    assert got["spatial"]["k"] == pytest.approx(K, rel=1e-9)
    assert got["spatial"]["alpha"] == pytest.approx(alpha, rel=1e-8)
    assert got["spatial"]["celerity_error"] < 1e-8
    assert got["two_probe"]["celerity_error"] < 1e-8
    # Celerity is compared with the dispersion relation at the *measured* k, so
    # the exact answer is the only way to score zero here.
    assert got["spatial"]["alias_risk"] is None      # no prior supplied


def test_envelope_formula_reads_decay_as_reflection_while_detrended_does_not():
    """The documented defect of the campaign formula, pinned as a number."""
    positions = production_positions()
    times = sample_times(OMEGA)
    rows = {}
    for loss in (0.0, 0.05, 0.20, 0.30):
        alpha = -math.log(1.0 - loss) / WAVELENGTH
        series = probe_series(times, positions, K, OMEGA, alpha=alpha)
        got = measure(positions, series, times, OMEGA)
        rows[loss] = (got["reflection"]["envelope"], got["reflection"]["detrended"])
    assert rows[0.0][0] == pytest.approx(0.0, abs=1e-12)
    # Pure decay, no reflector: the envelope formula invents a standing wave.
    assert rows[0.05][0] == pytest.approx(0.023, abs=5e-3)
    assert rows[0.20][0] == pytest.approx(0.100, abs=5e-3)
    assert rows[0.30][0] > 0.15
    # The log-linear detrend removes it exactly.
    for loss in (0.0, 0.05, 0.20, 0.30):
        assert abs(rows[loss][1]) < 1e-9
    # Monotone in the decay rate, which is what makes it a diagnostic.
    assert rows[0.05][0] < rows[0.20][0] < rows[0.30][0]


def test_recovered_decay_ratio_matches_the_imposed_one():
    """The amplitude column the report quotes is the imposed ratio."""
    positions = production_positions()
    times = sample_times(OMEGA)
    for loss in (0.05, 0.2, 0.4):
        alpha = -math.log(1.0 - loss) / WAVELENGTH
        series = probe_series(times, positions, K, OMEGA, alpha=alpha)
        got = measure(positions, series, times, OMEGA)
        assert got["reflection"]["height_ratio_per_wavelength"] == pytest.approx(
            1.0 - loss, rel=1e-9)


# --------------------------------------------------------------------------- #
# Reflection: the estimator-only bias (finite sampling of the envelope)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("true_r", [0.02, 0.1, 0.4])
def test_detrended_reflection_undercounts_by_a_bounded_amount(true_r):
    """Four probes over 0.9 lambda cannot see the true extremes of the envelope.

    The measured ``R`` is therefore systematically low; the calibration fixes the
    bound (14-19 % relative) so the gate value is read as a lower bound rather
    than as the reflection coefficient.
    """
    positions = production_positions()
    times = sample_times(OMEGA)
    alpha = -math.log(0.95) / WAVELENGTH
    recovered = []
    for step in range(12):
        psi = math.pi * step / 12.0
        series = probe_series(times, positions, K, OMEGA, alpha=alpha,
                              reflection=true_r, psi=psi)
        got = measure(positions, series, times, OMEGA)
        recovered.append(got["reflection"]["detrended"])
    mean = float(np.mean(recovered))
    assert mean <= true_r * 0.999                    # always low, never high
    assert mean >= true_r * 0.75                     # but bounded: no 2x loss
    assert max(recovered) <= true_r * 1.05
    # Worst case over the standing-wave phase is ~2.2x low at R = 0.4: the gate
    # value is a lower bound on reflection, not a measurement of it.
    assert min(recovered) >= true_r * 0.45


def test_envelope_estimator_is_worse_than_detrended_when_the_wave_decays():
    """At the same true R the campaign formula loses to the detrended one."""
    positions = production_positions()
    times = sample_times(OMEGA)
    alpha = -math.log(0.90) / WAVELENGTH
    psi = 0.0
    series = probe_series(times, positions, K, OMEGA, alpha=alpha,
                          reflection=0.10, psi=psi)
    got = measure(positions, series, times, OMEGA)
    assert abs(got["reflection"]["detrended"] - 0.10) < 0.04
    assert got["reflection"]["envelope"] > got["reflection"]["detrended"]


# --------------------------------------------------------------------------- #
# Noise floor of criterion (b)
# --------------------------------------------------------------------------- #

def test_the_three_percent_dispersion_gate_is_above_the_instrument_floor():
    """White noise at 2 % of H/2 must not manufacture a 3 % celerity error.

    The harmonic fit averages every sample in the window, so the phase noise is
    ``sigma sqrt(2/N)``: the estimator, not the solver, is far from the limit.
    """
    positions = production_positions()
    times = sample_times(OMEGA)
    rng = np.random.default_rng(5)
    errors = []
    for _ in range(40):
        series = probe_series(times, positions, K, OMEGA)
        series = [record + rng.normal(0.0, 0.02 * HEIGHT / 2.0, size=record.size)
                  for record in series]
        got = measure(positions, series, times, OMEGA)
        errors.append(got["spatial"]["celerity_error"])
    assert float(np.median(errors)) < 0.005
    assert float(np.percentile(errors, 95)) < 0.01


def test_the_analytic_phase_noise_matches_the_measured_one():
    """``sigma_phi = sigma sqrt(2/N)`` predicts the observed k scatter."""
    positions = production_positions()
    times = sample_times(OMEGA)
    separation = positions[1] - positions[0]
    rng = np.random.default_rng(9)
    level = 0.02
    measured = []
    for _ in range(60):
        series = probe_series(times, positions[:2], K, OMEGA)
        series = [record + rng.normal(0.0, level * HEIGHT / 2.0, size=record.size)
                  for record in series]
        pair = two_probe_dispersion(times, series[0], series[1], separation,
                                    OMEGA, DEPTH)
        measured.append(abs(pair.k_measured - K) / K)
    predicted = level * math.sqrt(2.0 / times.size) / (K * separation)
    assert float(np.median(measured)) == pytest.approx(predicted, rel=0.5)


# --------------------------------------------------------------------------- #
# Aliasing: the failure mode the estimators used to hide
# --------------------------------------------------------------------------- #

def test_half_wavelength_spacing_flips_the_sign_and_is_rejected():
    """At exactly lambda/2 the wrapped step is pi: k inverts. Never hide it."""
    positions = [0.4 * WAVELENGTH + 0.5 * WAVELENGTH * index for index in range(4)]
    phases = [((K * x + math.pi) % (2 * math.pi)) - math.pi for x in positions]
    with pytest.raises(ValueError, match="half a wavelength"):
        spatial_wave_number(positions, [1.0] * 4, phases, OMEGA, DEPTH)
    times = sample_times(OMEGA)
    series = probe_series(times, positions, K, OMEGA)
    with pytest.raises(ValueError, match="half a wavelength"):
        two_probe_dispersion(times, series[0], series[1],
                             positions[1] - positions[0], OMEGA, DEPTH)


@pytest.mark.parametrize("spacing", [0.5, 0.6, 1.5, 2.5])
def test_the_imposed_wave_number_resolves_the_branch_and_flags_the_risk(spacing):
    """With the imposed k as prior the aliasing is resolvable and *labelled*."""
    positions = [0.4 * WAVELENGTH + spacing * WAVELENGTH * index for index in range(4)]
    times = sample_times(OMEGA)
    series = probe_series(times, positions, K, OMEGA)
    got = measure(positions, series, times, OMEGA,
                  use_spatial=spacing < 0.5)
    resolved = spatial_wave_number(positions, got["heights"], got["phases"],
                                  OMEGA, DEPTH, k_prior=K)
    assert resolved["k"] == pytest.approx(K, rel=1e-9)
    assert resolved["celerity_error"] < 1e-8
    assert resolved["alias_risk"] is True
    # The prior had to supply at least pi/(k dx) of the phase information.
    assert 0.0 < resolved["branch_tolerance"] <= 1.0


def test_without_a_prior_a_wide_layout_is_undetectable_not_guessed():
    """Honest limitation: past lambda/2 the record alone cannot tell the branch.

    A 1.5-wavelength spacing returns an aliased ``k`` with ``alias_risk`` None --
    which is exactly why the production layouts are pinned below lambda/2 and why
    ``summarise`` passes the imposed k. At exactly lambda/2 the sign flip *is*
    detectable and raises.
    """
    positions = [0.4 * WAVELENGTH + 1.5 * WAVELENGTH * index for index in range(4)]
    times = sample_times(OMEGA)
    series = probe_series(times, positions, K, OMEGA)
    got = measure(positions, series, times, OMEGA)
    assert "rejected" in got["spatial"]               # cannot be resolved alone
    assert "rejected" in got["two_probe"]
    heights = [harmonic_fit(times, record, OMEGA).height for record in series]
    with pytest.raises(ValueError, match="half a wavelength"):
        spatial_wave_number(positions, heights, got["phases"], OMEGA, DEPTH)


def test_production_dispersion_layout_is_unambiguous():
    """The four-probe dispersion layout is safe with and without the prior."""
    positions = production_positions()
    times = sample_times(OMEGA)
    series = probe_series(times, positions, K, OMEGA)
    got = measure(positions, series, times, OMEGA)
    assert got["spatial"]["k"] == pytest.approx(K, rel=1e-9)
    with_prior = spatial_wave_number(positions, got["heights"], got["phases"],
                                     OMEGA, DEPTH, k_prior=K)
    assert with_prior["alias_risk"] is False
    assert with_prior["k"] == pytest.approx(K, rel=1e-9)


def test_decay10_layout_needs_the_prior_for_its_decay_column():
    """The ten-wavelength case is spaced 2.5 lambda: amplify-only fallback."""
    positions = [offset * WAVELENGTH for offset in DECAY10_OFFSETS]
    times = sample_times(OMEGA)
    alpha = -math.log(0.95) / WAVELENGTH
    series = probe_series(times, positions, K, OMEGA, alpha=alpha)
    got = measure(positions, series, times, OMEGA)
    # The amplitude-only decay fit never touches the phase, so it is exact.
    assert got["reflection"]["height_ratio_per_wavelength"] == pytest.approx(
        0.95, rel=1e-9)
    assert got["reflection"]["detrended"] < 0.01
    # The phase-based k on this layout is only usable with the imposed prior.
    assert "rejected" in got["spatial"]
    heights = got["heights"]
    with pytest.raises(ValueError, match="half a wavelength"):
        spatial_wave_number(positions, heights, got["phases"], OMEGA, DEPTH)
    with_prior = spatial_wave_number(positions, got["heights"], got["phases"],
                                     OMEGA, DEPTH, k_prior=K)
    assert with_prior["k"] == pytest.approx(K, rel=1e-9)
    assert with_prior["alias_risk"] is True


# --------------------------------------------------------------------------- #
# The production chain: analyse_regular -> case_row (the gate numbers)
# --------------------------------------------------------------------------- #

def synthetic_run(offsets=PRODUCTION_OFFSETS, alpha=0.0, reflection=0.0):
    """A WaveTankRun-shaped stub: probe records with a known answer."""
    positions = [offset * WAVELENGTH for offset in offsets]
    times = sample_times(OMEGA, periods=1.2)
    series = probe_series(times, positions, K, OMEGA, alpha=alpha,
                          reflection=reflection)
    return SimpleNamespace(probe_x=positions, times=list(times),
                           probes=[list(record) for record in series],
                           simulated_time=float(times[-1]))


def production_config(**overrides):
    settings = dict(length=2.4 * WAVELENGTH, depth=DEPTH, dx=0.02 * DEPTH,
                    kh=KH, wave_height=HEIGHT, wave_kind="regular",
                    maker="relaxation", sponge_wavelengths=0.9,
                    probe_offsets=PRODUCTION_OFFSETS, ramp_periods=0.5,
                    max_courant=0.45)
    settings.update(overrides)
    return WaveTankConfig(**settings)


def test_analyse_regular_reads_the_known_answer_through_the_full_chain():
    """End-to-end: probe records -> analyse_regular -> the gate columns."""
    cfg = production_config()
    run = synthetic_run(alpha=-math.log(0.95) / WAVELENGTH)
    analysis = wave_tank_run.analyse_regular(run, cfg)
    assert analysis["dispersion"]["k_measured"] == pytest.approx(K, rel=1e-9)
    assert analysis["dispersion"]["celerity_error"] < 1e-8
    assert analysis["dispersion"]["alias_risk"] is False
    # One implementation only: the payload must carry the library value.
    heights = [fit["height"] for fit in analysis["probe_fits"]]
    expected, decay = detrended_reflection(run.probe_x, heights, cfg.wavelength())
    assert analysis["reflection"]["R_detrended"] == pytest.approx(expected, rel=1e-12)
    assert analysis["reflection"]["height_ratio_per_wavelength"] == pytest.approx(
        decay, rel=1e-12)


def test_a_one_in_ten_reflection_breaks_the_three_percent_dispersion_gate():
    """Contamination the estimator cannot remove: reflection, not noise.

    A four-probe array spanning 0.9 lambda samples less than one period of the
    ``lambda/2`` standing-wave pattern, so the reflected component is not averaged
    out but read as a phase tilt. The measured bias is a few percent at
    ``R = 0.1`` -- larger than the 3 % gate and much larger than the noise floor
    (0.01 % at 1 % probe noise), so a reflection that survives the sponge layer is
    the binding error of criterion (b).
    """
    cfg = production_config()
    baseline = None
    for true_r in (0.0, 0.05, 0.10, 0.20):
        run = synthetic_run(alpha=-math.log(0.95) / WAVELENGTH, reflection=true_r)
        analysis = wave_tank_run.analyse_regular(run, cfg)
        error = analysis["dispersion"]["celerity_error"]
        if baseline is None:
            baseline = error
            assert error < 1e-8                 # R = 0: the estimator is exact
            continue
        assert abs(analysis["dispersion"]["k_measured"] - K) / K > 0.01
    assert true_r == 0.20                       # loop ran to the end


def test_case_row_carries_an_exact_dispersion_gate_for_a_synthetic_tank():
    """summarise.case_row, the function that writes the verdict, is calibrated."""
    cfg = production_config()
    run = synthetic_run(alpha=-math.log(0.95) / WAVELENGTH)
    analysis = wave_tank_run.analyse_regular(run, cfg)
    payload = {
        "case": "dispersion_kh3",
        "config": {"kh": KH, "dx_over_depth": 0.02, "c0_factor": 10.0,
                   "height_over_depth": 0.1, "dt": cfg.dt,
                   "depth": DEPTH, "wavelength": cfg.wavelength()},
        "run": {"steps_taken": 1, "particles": 1, "wall_seconds": 0.0,
                "max_courant": 0.0, "max_compression": 0.0, "mass_error": 0.0,
                "volume_error": 0.0, "escaped": 0},
        "analysis": analysis,
        "passed": True,
    }
    row = case_row(payload)
    # The gate value of criterion (b) is the spatial estimator, and on a known
    # signal it reads zero: the estimator contributes no error of its own.
    assert row["celerity_error_spatial"] < 1e-8
    assert row["alias_risk"] is False
    assert row["reflection_detrended"] is not None
    # Criterion (c)'s amplitude column falls back to the phase-free estimate.
    assert row["decay_ratio_per_wavelength"] == pytest.approx(0.95, rel=1e-8)
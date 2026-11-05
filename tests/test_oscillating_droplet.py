"""Tests for the oscillating-drop verification (gap-analysis row B3, open half).

Three layers, from cheapest to most expensive:

1. **Analytic layer** -- the 2-D Rayleigh-Lamb frequency coefficient is
   re-derived from the curvature of a perturbed circle by finite differences,
   so the reference used by the benchmark cannot be a transcription error, and
   the 3-D/2-D formula difference (15.5 % in frequency) is pinned.
2. **Estimator layer** -- the initial ellipse conserves area and the
   second-moment estimator returns the seeded mode-2 amplitude.
3. **Signal layer** -- the damped-cosine fit and the zero-crossing estimator
   recover a synthetic ``A e^{-lambda t} cos(omega t + phi) + c``.
4. **Physics layer** -- one real relaxation; the gate is recorded in
   ``docs/verification_oscillating_droplet.md``.

Run only the cheap layers with ``pytest -m 'not slow'``.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "oscillating_droplet"
GATE_PCT = 15.0


def _bench():
    spec = importlib.util.spec_from_file_location("oscillating_run",
                                                  BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _bench()


# --------------------------------------------------------------------------- #
# 1. analytic layer
# --------------------------------------------------------------------------- #
def curvature_of_polar(r, dr, ddr):
    """``kappa = (r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^{3/2}`` (plane curve)."""
    return (r * r + 2.0 * dr * dr - r * ddr) / (r * r + dr * dr) ** 1.5


def test_laplace_curvature_expansion_gives_the_rayleigh_lamb_coefficient():
    """``kappa(eps) = (1/R)[1 + (m^2-1) eps]`` at ``theta = 0``, ``eps`` dimensionless.

    This is the linearisation that turns the Laplace pressure perturbation into
    the restoring force of eq. (2) of the benchmark docstring:
    ``p' = sigma (m^2-1) eps / R``, hence ``omega^2 = sigma m (m^2-1)/(rho R^3)``.
    It is checked numerically rather than trusted from the analytic sketch,
    because an ``m^2-2`` slip in this expansion would move the m = 2 frequency
    by 18 % -- larger than the gate.
    """
    R = 0.02
    eps = 1.0e-6
    for m in (2, 3, 4):
        def curv(theta, eta):
            # r(theta) = R(1 + eta cos m theta) with its exact derivatives.
            r = R * (1.0 + eta * math.cos(m * theta))
            dr = -R * eta * m * math.sin(m * theta)
            ddr = -R * eta * m * m * math.cos(m * theta)
            return curvature_of_polar(r, dr, ddr)

        # d(kappa)/d(eps) at theta = 0 (where sin(m theta) = 0), eps -> 0
        slope = (curv(0.0, eps) - curv(0.0, -eps)) / (2.0 * eps)
        assert slope == pytest.approx((m * m - 1.0) / R, rel=1.0e-6)
        # Combine with the Bernoulli balance p' = sigma kappa' eps and the
        # irrotational field phi = A r^m cos(m theta) -> omega^2 = sigma m
        # kappa' / (rho R^2); that must reproduce the reference law bit for bit.
        omega_sq = 7.2 * m * slope / (1000.0 * R ** 2)
        omega = math.sqrt(omega_sq)
        assert omega == pytest.approx(
            math.sqrt(7.2 * m * (m * m - 1.0) / (1000.0 * R ** 3)), rel=1e-9)
        # ... and the circle itself is recovered for eta = 0.
        assert curv(0.7, 0.0) == pytest.approx(1.0 / R, rel=1.0e-9)


def test_two_dimensional_and_three_dimensional_laws_differ_by_the_gate(bench):
    """Guards against silently using the 3-D sphere law on a 2-D drop."""
    R, sigma, rho = 0.02, 7.2, 1000.0
    two = bench.analytic_omega(R, sigma, rho)
    three = bench.analytic_omega_3d(R, sigma, rho)
    assert two == pytest.approx(math.sqrt(6.0 * sigma / (rho * R ** 3)))
    assert three == pytest.approx(math.sqrt(8.0 * sigma / (rho * R ** 3)))
    # A 15.5 % frequency difference: the choice is not cosmetic.
    assert (three / two - 1.0) * 100.0 == pytest.approx(
        (math.sqrt(8.0 / 6.0) - 1.0) * 100.0, rel=1e-12)
    assert (three / two - 1.0) * 100.0 > GATE_PCT


def test_reference_frequency_and_period(bench):
    """Pinned reference values for the reported configuration (R = 2 cm)."""
    omega = bench.analytic_omega(0.02, 7.2, 1000.0)
    assert omega == pytest.approx(73.4847, rel=1e-4)
    assert 2.0 * math.pi / omega == pytest.approx(0.0855, rel=1e-3)


# --------------------------------------------------------------------------- #
# 2. estimator layer
# --------------------------------------------------------------------------- #
def test_initial_ellipse_conserves_area_and_carries_the_seeded_mode(bench):
    R, dx, eps = 0.02, 0.002, 0.10
    pos, mass = bench.elliptical_drop(R, dx, R * 1.0, eps)  # rho0 = R only sets mass
    assert mass.shape == (pos.shape[0],)
    amp = bench.mode_amplitude(pos, mass)
    # Area: lattice cell area dx * (sqrt(3)/2 dx) times the particle count.
    area = pos.shape[0] * dx * dx * math.sqrt(3.0) / 2.0
    assert area == pytest.approx(math.pi * R * R, rel=0.03)
    # Semi-axes: a = R(1+eps), b = R/(1+eps), a*b = R^2 exactly.
    assert amp["semi_major"] / amp["semi_minor"] == pytest.approx(
        (1.0 + eps) ** 2, rel=0.03)
    assert amp["r_eff"] == pytest.approx(R, rel=0.02)
    assert amp["s_moment"] == pytest.approx(eps, rel=0.05)


def test_mode_estimator_is_invariant_to_a_translation(bench):
    R, dx = 0.02, 0.003
    pos, mass = bench.elliptical_drop(R, dx, 1000.0, 0.1)
    a = bench.mode_amplitude(pos, mass)["s_moment"]
    b = bench.mode_amplitude(pos + torch.tensor([3.0, -1.5], dtype=pos.dtype),
                             mass)["s_moment"]
    assert a == pytest.approx(b, rel=1e-12)


def test_area_conserving_axes_not_the_naive_ones(bench):
    """``b = R(1-eps)`` would break the reference area by ``O(eps^2)``."""
    R, eps = 0.02, 0.1
    a, b = R * (1.0 + eps), R / (1.0 + eps)
    naive_b = R * (1.0 - eps)
    assert a * b == pytest.approx(R * R, rel=1e-15)
    assert a * naive_b / (R * R) - 1.0 == pytest.approx(-eps ** 2, rel=1e-12)


# --------------------------------------------------------------------------- #
# 3. signal layer
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic():
    omega, lam, amp, phi, offset = 73.4847, 3.0, 0.1, 0.4, 0.01
    t = [i * 3.9e-5 for i in range(6600)]   # ~3 periods
    s = [offset + amp * math.exp(-lam * ti) * math.cos(omega * ti + phi)
         for ti in t]
    return t, s, {"omega": omega, "lambda": lam, "amp": amp, "offset": offset}


def test_damped_cosine_fit_recovers_a_known_oscillation(bench, synthetic):
    t, s, truth = synthetic
    fit = bench.damped_cosine_fit(t, s, truth["omega"])
    assert (fit["omega"] - truth["omega"]) / truth["omega"] < 0.005
    assert fit["lambda_per_s"] == pytest.approx(truth["lambda"], rel=0.10)
    assert fit["r2"] > 0.999
    assert fit["amplitude0"] == pytest.approx(truth["amp"], rel=0.02)


def test_zero_crossing_period_recovers_the_period(bench, synthetic):
    t, s, truth = synthetic
    zc = bench.zero_crossing_period(t, s)
    assert zc["n_crossings"] >= 2
    assert (zc["omega"] - truth["omega"]) / truth["omega"] < 0.01


def test_peak_decay_recovers_the_damping_rate(bench, synthetic):
    t, s, truth = synthetic
    pk = bench.peak_decay(t, s)
    assert pk["n_peaks"] >= 3
    assert pk["lambda_per_s"] == pytest.approx(truth["lambda"], rel=0.10)


def test_fit_refuses_to_hallucinate_a_frequency_from_a_non_oscillating_trace(bench):
    """Noise without an oscillation must not produce a high ``r2``."""
    torch.manual_seed(0)
    t = [i * 3.9e-5 for i in range(2000)]
    s = [float(torch.randn(())) for _ in t]
    fit = bench.damped_cosine_fit(t, s, 73.4847)
    assert fit["r2"] < 0.05


# --------------------------------------------------------------------------- #
# 4. physics layer (slow)
# --------------------------------------------------------------------------- #
#: History, not a gate.  The mode-2 frequency error of this CI case, measured
#: while clearing the m3312 reds, is -31.78 % on loong64 and +8.64 % on the
#: x86_64 GitHub runner for the same code.  Neither number is a statement about
#: the model's accuracy: both recorded traces of this configuration violate
#: energy conservation (kinetic energy peaks ~450x above the surface energy
#: released by the seeded ellipse, onset ~1.1 reference periods), so the fitted
#: frequency is a function of the fit window -- it moves by 40+ percentage
#: points on one and the same trace -- and of the rounding-seeded growth rate,
#: not of the physics.  Pinning such a number under a *strict* xfail is what
#: turned this file red on every architecture whose trace reproduced the pin
#: (strict xfail + reproducible number = XPASS), so the live tests below assert
#: only invariants that survive a change of architecture, thread count or
#: reduction order.  The window-dependence study runs offline on the committed
#: traces in ``tests/test_oscillating_droplet_instability.py``.
RECORDED_MODE2_ERR_PCT = 31.78      # |error|, loong64 full window (history)
RECORDED_MODE2_ERR_X86_PCT = 8.64   # x86_64 GitHub runner (history)

#: Energy budget factor for the divergence test.  The measured peak is ~450x the
#: released surface energy.  The threshold is deliberately loose (2x, i.e. still
#: an unambiguous violation of the ``KE <= E_avail`` bound) so that neither a
#: re-tune nor a change of architecture can trip the test spuriously, while any
#: scheme that conserves energy -- the re-baseline trigger -- still fails it.
DIVERGENCE_FACTOR = 2.0


def available_surface_energy(radius: float, sigma: float, eps: float,
                             n_samples: int = 20000) -> float:
    """``E_avail = sigma (P_ellipse - P_circle)``: hard bound on the m = 2 KE.

    The seeded ellipse conserves area, so ``a = R (1 + eps)`` and
    ``b = R / (1 + eps)``; its perimeter is integrated numerically rather than
    from a truncated series, so the bound does not inherit an expansion error.
    """
    a, b = radius * (1.0 + eps), radius / (1.0 + eps)
    h = 2.0 * math.pi / n_samples
    perimeter = sum(math.hypot(a * math.sin(i * h), b * math.cos(i * h))
                    for i in range(n_samples)) * h
    return sigma * (perimeter - 2.0 * math.pi * radius)


@pytest.fixture(scope="module")
def release_case(bench):
    """One real release: the CI cost is why this is shared by the two tests.

    The configuration keeps the wall clock under the slow-test timeout: the cost
    of an oscillation scales as ``c0 sqrt(rho R / sigma)`` in steps, so a smaller
    drop with the same ``R/dx`` is cheaper.  The reported numbers use R = 2 cm.
    """
    return bench.run_case(radius=0.01, dx=0.001, sigma=7.2, rho0=1000.0,
                          sound_speed=20.0, viscosity=5e-5, eps=0.10,
                          periods=1.8)


def fit_error_at(case: dict, bench, periods: "float | None") -> float:
    """Mode-2 fit error over the first ``periods`` reference periods, in [%].

    ``periods=None`` fits the whole recorded trace.  This is the same estimator
    the benchmark reports (``damped_cosine_fit`` on the second moment), so the
    two windows below are admissible readings of one and the same trace.
    """
    trace = case["diagnostics"]["trace"]
    omega_ref = case["reference"]["omega_rad_per_s"]
    limit = (float("inf") if periods is None
             else periods * case["reference"]["period_s"])
    t = [ti for ti in trace["t"] if ti <= limit]
    fit = bench.damped_cosine_fit(t, list(trace["s_second_moment"])[:len(t)],
                                  omega_ref)
    return (fit["omega"] - omega_ref) / omega_ref * 100.0


@pytest.mark.slow
def test_release_case_cannot_support_a_15_percent_frequency_claim(release_case,
                                                                  bench):
    """Two architecture-independent proofs that this case is not a measurement.

    Route A -- energy conservation.  The kinetic energy of the m = 2 mode may not
    exceed the surface energy released by the seeded ellipse; a trace that does
    has no physical mode-2 frequency to fit.
    Route B -- reproducibility.  The *same* trace yields fits that differ by more
    than the 15 % gate depending on an admissible choice of fit window, so no
    single number is quotable either.

    Either route alone justifies the WITHHELD verdict of
    ``docs/verification_oscillating_droplet_instability.md``, and both survive a
    change of architecture, thread count or reduction order -- which the old
    ``-31.78 % +- 2 %`` strict-xfail pin did not: a strict xfail around a
    reproducible number is an XPASS (a failure) on every machine that reproduces
    it.  A scheme that actually fixes the divergence has to fail here, which is
    the reminder to re-baseline against the 15 % gate instead of trusting the
    history pin.
    """
    setup = release_case["setup"]
    energy = available_surface_energy(float(setup["radius_m"]),
                                      float(setup["sigma_N_per_m"]),
                                      float(setup["initial_eps"]))
    ke_max = max(release_case["diagnostics"]["trace"]["kinetic_energy_J"])
    assert energy > 0.0, "the seeded ellipse must store surface energy"
    creation = ke_max / energy
    spread = abs(fit_error_at(release_case, bench, None)
                 - fit_error_at(release_case, bench, 0.75))
    assert creation > DIVERGENCE_FACTOR or spread > GATE_PCT, (
        f"the case became measurable: the kinetic energy peak is only "
        f"{creation:.2f}x the released surface energy (runs must stay below "
        f"1x) and two admissible fit windows of the same trace disagree by "
        f"{spread:.1f} pt, i.e. below the {GATE_PCT:.0f} % gate.  Re-baseline "
        "the mode-2 frequency against the gate and update "
        "docs/verification_oscillating_droplet_instability.md")
    # The run fails on the instability, not on a mass leak.
    assert release_case["diagnostics"]["mass_relative_drift"] < 1e-12


@pytest.mark.slow
def test_reported_frequency_is_the_fit_of_the_live_trace(release_case, bench):
    """The quoted error must follow from this run's trace, not from history."""
    trace = release_case["diagnostics"]["trace"]
    omega_ref = release_case["reference"]["omega_rad_per_s"]
    fit = bench.damped_cosine_fit(list(trace["t"]),
                                  list(trace["s_second_moment"]), omega_ref)
    recomputed = (fit["omega"] - omega_ref) / omega_ref * 100.0
    frequency = release_case["frequency"]
    assert frequency["fit_second_moment_rel_error_pct"] == pytest.approx(
        recomputed, abs=1e-9)
    # Both estimators are recorded; neither may silently become NaN.
    assert math.isfinite(frequency["fit_second_moment_rel_error_pct"])
    assert math.isfinite(frequency["fit_kinetic_energy_rel_error_pct"])
    assert frequency["within_15pct"] is (
        abs(frequency["fit_second_moment_rel_error_pct"]) < GATE_PCT)

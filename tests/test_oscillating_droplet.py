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
# 2026-11-05 (m4484): the two physics-layer tests used to pin an
# architecture-specific number: the *same* code measures +8.64 % for the
# full-window mode-2 fit on x86_64 GitHub CI and -31.78 % on the loong64
# reference container, so they were parked as strict-xfail.  That is red CI in
# both directions and carries no physics: on loong64 the pin was met and the
# strict xfail turned into an XPASS failure, on x86_64 it did not.  The cause is
# not the fit, it is the solution: the recorded traces violate energy
# conservation -- KE_max exceeds the surface energy released by the seeded
# ellipse by ~4.7e2 (R = 1 cm, this case) and ~4.5e2 (R = 2 cm) -- so the
# second-moment series diverges from ~1.1 reference periods and the trajectory
# is round-off seeded.  Neither recorded number is a path-independent
# observable, so the measurement is WITHHELD, not PASS and not FAIL.
#
# The two tests below now assert the *invariant* claims, which hold on every
# architecture and are falsifiable in the useful direction:
#   1. the release case creates energy, so no frequency may be quoted from it;
#   2. refitting the same trace on prefixes of the window moves the answer by
#      more than the 15 % gate, so no gate verdict may be quoted either.
# A solver fix that makes the case stable and energy-conserving breaks (1) and
# (2), which forces a re-baseline of ``docs/verification_oscillating_droplet.md``
# (it still records FAIL, -43.5 %, for the R = 2 cm case) instead of silently
# flipping a pin.  Evidence: docs/verification_oscillating_droplet_instability.md
# and benchmarks/oscillating_droplet_instability/report.json.
#: fit windows, in reference periods (the set used by the instability study)
WINDOW_PERIODS = (0.5, 0.75, 1.0, 1.8)
#: KE_max / released surface energy above which the case is energy-creating.
#: Measured ~4.7e2; 1e1 leaves a two-decade margin that discretisation noise of
#: a stable scheme cannot reach.
ENERGY_CREATION_FACTOR = 10.0


def _ellipse_perimeter(a: float, b: float, n: int = 4096) -> float:
    """Perimeter of the half-axes ``(a, b)`` ellipse by midpoint quadrature."""
    dphi = 2.0 * math.pi / n
    total = 0.0
    for k in range(n):
        phi = (k + 0.5) * dphi
        total += math.hypot(a * math.sin(phi), b * math.cos(phi))
    return total * dphi


def _available_energy_j(case: dict) -> float:
    """Surface energy released when the seeded ellipse relaxes to a circle.

    ``sigma * (P_ellipse - P_circle)`` with area-conserving half axes
    ``R(1 + eps)``, ``R / (1 + eps)``: a hard upper bound on the kinetic energy
    an energy-conserving solver can extract from this initial state.
    """
    setup = case["setup"]
    radius, eps = setup["radius_m"], setup["initial_eps"]
    a, b = radius * (1.0 + eps), radius / (1.0 + eps)
    released = _ellipse_perimeter(a, b) - 2.0 * math.pi * radius
    assert released > 0.0
    return setup["sigma_N_per_m"] * released


def _window_error_pct(bench, case: dict, window_periods: float) -> tuple[float, float]:
    """Mode-2 frequency error (percent) refitted on a prefix of the trace."""
    omega_ref = case["reference"]["omega_rad_per_s"]
    trace = case["diagnostics"]["trace"]
    if math.isinf(window_periods):
        ts, ss = list(trace["t"]), list(trace["s_second_moment"])
    else:
        t_end = window_periods * case["reference"]["period_s"]
        pairs = [(ti, si) for ti, si in zip(trace["t"],
                                            trace["s_second_moment"])
                 if ti <= t_end]
        ts = [ti for ti, _ in pairs]
        ss = [si for _, si in pairs]
    assert len(ts) >= 20, f"window {window_periods} periods is too short"
    fit = bench.damped_cosine_fit(ts, ss, omega_ref)
    return (fit["omega"] - omega_ref) / omega_ref * 100.0, float(
        fit.get("r2", float("nan")))


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


def test_release_case_creates_energy_so_no_frequency_can_be_quoted(release_case):
    """Energy creation is proof of numerical invalidity, on every architecture.

    The trace is *not* merely inaccurate: kinetic energy passes the hard bound
    set by the surface energy released by the seeded ellipse, so the amplitude
    series is a round-off-seeded blow-up and its fitted frequency is a property
    of the round-off, not of the model.  This is the load-bearing claim of the
    instability audit; it replaces the old architecture pin because the factor
    is O(1e2) and cannot flip between x86_64 and loong64.
    """
    budget = _available_energy_j(release_case)
    ke_max = max(release_case["diagnostics"]["trace"]["kinetic_energy_J"])
    ratio = ke_max / budget
    assert ratio > ENERGY_CREATION_FACTOR, (
        f"KE_max / released surface energy = {ratio:.1f} <= "
        f"{ENERGY_CREATION_FACTOR}: the case no longer creates energy, so its "
        "frequency may now be measurable -- re-baseline "
        "docs/verification_oscillating_droplet.md and this test")
    # The blow-up is not a mass leak: mass is conserved to machine precision, so
    # the energy defect belongs to the pressure/surface-tension coupling.
    assert release_case["diagnostics"]["mass_relative_drift"] < 1e-12


@pytest.mark.slow
def test_mode2_frequency_is_a_function_of_the_fit_window(bench, release_case):
    """The 'measured' frequency moves by more than the gate with the window.

    Therefore no PASS/FAIL verdict may be quoted for the mode-2 frequency of
    this case: the 15 % gate is inside the fit-window spread.  Recorded numbers
    (+8.64 % x86_64, -31.78 % loong64) are two points of this spread, which is
    why they disagreed; the fit itself still recovers a visible oscillation
    (every window keeps a high ``r2``), so the spread is a property of the
    solution, not a broken estimator.
    """
    errors = {}
    for window_periods in WINDOW_PERIODS:
        errors[window_periods] = _window_error_pct(bench, release_case,
                                                   window_periods)
    spread = max(e for e, _ in errors.values()) - min(
        e for e, _ in errors.values())
    assert spread > GATE_PCT, (
        f"fit-window spread {spread:.1f} pp <= {GATE_PCT} % gate: the "
        "measurement became quotable -- re-baseline "
        "docs/verification_oscillating_droplet.md and this test")
    assert min(r2 for _, r2 in errors.values()) > 0.5, errors
    # The recorded number must be reproducible from the recorded trace: no
    # hand-edited pin survives in the report.
    recorded = release_case["frequency"]["fit_second_moment_rel_error_pct"]
    full_window, _ = _window_error_pct(bench, release_case, math.inf)
    assert recorded == pytest.approx(full_window, abs=1e-9)

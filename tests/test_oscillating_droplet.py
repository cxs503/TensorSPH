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
import platform
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
#: The m3312 pins measured this cheap case while clearing CI reds.  The run is
#: reproducible *within* a platform (bit-for-bit identical for 1, 4, 8 and 16
#: torch threads -- there is no RNG, no wall-clock dependence and no atomic
#: reduction in the integrator), but not *across* platforms: see below.  The
#: reference case R = 2 cm is recorded at -43.5 % in
#: ``docs/verification_oscillating_droplet.md`` section 3.1 / 5, i.e. the model
#: is a documented FAIL against the 15 % bar -- this cheap case is not better.
# The cheap CI case (R = 1 cm, dx = 1 mm, 1.8 periods, eps = 0.10) does not
# complete a clean oscillation: it diverges acoustically well before the end of
# the window (``max|v|`` reaches ~1.6 c0 in the run behind the pin below; the
# R = 2 cm release-scale case reaches 3e5 x the capillary speed, see section 7
# of ``docs/verification_oscillating_droplet.md``).  The number pinned here is
# therefore a **pseudo-frequency of a diverging trace**, not a physical
# measurement of the m = 2 Rayleigh-Lamb mode.
#
# A pseudo-frequency is not platform-portable *by construction*: two runs of
# identical code on identical input diverge at slightly different steps, and
# the damped-cosine estimator then locks onto a different pseudo-mode of a
# trace whose fit r2 is only ~0.79.  Measured spread on the same commit:
# -31.78 % (self-hosted loong64 runner -- this repository's CI) vs +8.64 %
# (GitHub-hosted x86_64 runner, 2026-11-04).  Neither value is
# architecture-canonical, so the pin is **per platform**: it stays 2 pp tight
# on each recorded platform (tight drift detection is the whole point of the
# pin) and an unrecorded platform is reported as xfail, never as a silent pass.
# What is *not* platform-specific -- and what the second test below asserts --
# is the verdict: this quantity is not validated on any platform.
RECORDED_MODE2_ERR_PCT = {
    "loongarch64": -31.78,   # self-hosted ci-loong64 runner (used by CI)
    "x86_64": +8.64,         # GitHub-hosted x86_64 runner, recorded 2026-11-04
}
RECORDED_MODE2_TOL_PCT = 2.0


def recorded_mode2_err() -> float:
    """Recorded mode-2 pseudo-frequency error [%] for this platform, else xfail."""
    arch = platform.machine()
    if arch not in RECORDED_MODE2_ERR_PCT:
        pytest.xfail(
            f"no recorded mode-2 pin for platform {arch!r}: this quantity is a "
            "pseudo-frequency of a diverging trace (see the comment above), so "
            "it must be recorded from a real run before it can be asserted")
    return RECORDED_MODE2_ERR_PCT[arch]


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


def test_pin_table_matches_the_evidence_json():
    """The per-platform pins and the recorded evidence may not drift apart.

    ``benchmarks/oscillating_droplet/ci_pin_evidence.json`` is written
    programmatically by the run itself (no hand-copied numbers), so it is the
    machine-checkable witness of *why* the table above holds these values.
    Cheap (no simulation): it guards the bookkeeping, the physics tests below
    guard the measurement.

    The comparison is done against the platform the evidence was *recorded* on,
    not against this machine: the file carries one trace (loongarch64), so
    demanding that x86_64's pin equal it would be red on any x86_64 runner for
    no physical reason.  The consistency claim is platform-independent and
    therefore checked everywhere.
    """
    import json

    evidence = json.loads((BENCH / "ci_pin_evidence.json").read_text())
    measured = evidence["measured"]
    evidence_arch = evidence["platform"]
    assert evidence_arch in RECORDED_MODE2_ERR_PCT, (
        f"evidence was recorded on {evidence_arch!r}, which has no pin in "
        "RECORDED_MODE2_ERR_PCT -- add the pin, do not drop the evidence")
    assert RECORDED_MODE2_ERR_PCT[evidence_arch] == pytest.approx(
        measured["fit_rel_error_pct"], abs=0.01), (
        f"pin {RECORDED_MODE2_ERR_PCT[evidence_arch]:+.2f} % vs evidence "
        f"{measured['fit_rel_error_pct']:+.2f} % on {evidence_arch}")
    assert evidence["recorded_platform_pins_pct"] == {
        arch: pytest.approx(pin, abs=0.01)
        for arch, pin in RECORDED_MODE2_ERR_PCT.items()}, (
        "the pin table inside the evidence file disagrees with the one in this "
        "module -- re-record instead of editing one of the two by hand")
    assert measured["fit_r2"] < 0.9, (
        "a fit r2 below 0.9 is what makes this number a pseudo-frequency; if "
        "the fit improved, re-baseline the pins and the doc instead of the "
        "threshold")


def test_recorded_mode2_frequency_deficiency_is_reproduced(release_case):
    """Pin the *measured* mode-2 pseudo-frequency on this platform.

    The pin is not a physics claim: the trace diverges, so the recorded FAIL
    verdict lives in ``docs/verification_oscillating_droplet.md`` and in
    ``test_oscillation_frequency_gate_verdict_stays_fail`` below.  What this
    test buys is drift detection: if the model (or the estimator) moves on a
    recorded platform, the pseudo-frequency moves and CI goes red, which is the
    only cheap signal a diverging configuration can give.
    """
    recorded = recorded_mode2_err()
    err = release_case["frequency"]["fit_second_moment_rel_error_pct"]
    assert err == pytest.approx(recorded, abs=RECORDED_MODE2_TOL_PCT), (
        f"mode-2 pseudo-frequency drifted on {platform.machine()}: measured "
        f"{err:+.2f} %, recorded {recorded:+.2f} % +- "
        f"{RECORDED_MODE2_TOL_PCT} pp.  Re-record the pin for this platform "
        "(and only for this platform) together with "
        "docs/verification_oscillating_droplet.md.")


@pytest.mark.slow
def test_oscillation_frequency_gate_verdict_stays_fail(release_case):
    """The recorded platform verdict is FAIL on *every* platform.

    Deliberately phrased as a verdict test rather than as "|err| > gate on this
    machine": x86_64 records +8.64 % (inside the 15 % gate) while loong64
    records -31.78 %, i.e. the two platforms disagree by ~40 pp, which is
    exactly why no single-platform number may be read as "the physics passes".
    The verdict rests on facts that are platform-independent: (a) the release
    scale benchmark is -43.5 % and this CI case -31.8 % on the CI platform,
    (b) both runs diverge (section 7 of the doc), and (c) neither platform is
    within the 3 % standard.  A real fix (a converging run) has to fail here
    first -- that failure is the reminder to re-render the document.
    """
    recorded = recorded_mode2_err()
    err = release_case["frequency"]["fit_second_moment_rel_error_pct"]
    report = ROOT / "docs" / "verification_oscillating_droplet.md"
    text = report.read_text()
    assert "FAIL" in text and "-43.5" in text, (
        "the document must keep recording the release-scale deficiency "
        "(-43.5 %) and the FAIL verdict")
    assert release_case["frequency"]["within_3pct"] is False, (
        f"mode-2 pseudo-frequency {err:+.2f} % claims the 3 % standard on "
        f"{platform.machine()}; that is a diverging-trace artefact -- "
        "re-render docs/verification_oscillating_droplet.md before relaxing")
    assert err == pytest.approx(recorded, abs=RECORDED_MODE2_TOL_PCT), (
        f"measured {err:+.2f} % vs recorded {recorded:+.2f} % on "
        f"{platform.machine()}")
    assert release_case["diagnostics"]["mass_relative_drift"] < 1e-12


# --------------------------------------------------------------------------- #
# 5. properties ported from the m2919 lineage (PR #87 semantic union)
# --------------------------------------------------------------------------- #
# The pre-merge m2919 branch of this benchmark shipped its own implementation
# (``--case`` CLI, ``lobed_droplet``, ``fit_damped_oscillation``).  The merge
# keeps the main-line harness (it is the functional superset: four studies plus
# the document generator) and carries the m2919 *properties* over here, so
# nothing the m2919 tests guarded is lost.  Name map:
#   drop_frequency_2d(sigma, rho, R, n)  -> analytic_omega(R, sigma, rho, m=n)
#   drop_frequency_3d(sigma, rho, R, n)  -> analytic_omega_3d(R, sigma, rho, m=n)
#   lobed_droplet(...)                   -> lobed_droplet(...)  (same signature)
def test_mode_one_is_a_translation_and_has_no_frequency(bench):
    """``m = 1`` is a rigid translation: the formula must refuse it, not return 0."""
    with pytest.raises(ValueError):
        bench.analytic_omega(0.02, 7.2, 1000.0, 1)
    with pytest.raises(ValueError):
        bench.analytic_omega(0.02, 7.2, 1000.0, 0)


def test_frequency_scales_as_sqrt_sigma_and_r_minus_three_halves(bench):
    base = bench.analytic_omega(0.02, 7.2, 1000.0, 2)
    assert bench.analytic_omega(0.02, 4.0 * 7.2, 1000.0, 2) == pytest.approx(
        2.0 * base, rel=1e-12)
    assert bench.analytic_omega(0.04, 7.2, 1000.0, 2) == pytest.approx(
        base / 2.0 ** 1.5, rel=1e-12)


def test_branches_agree_for_large_azimuthal_modes(bench):
    m = 200
    ratio = (bench.analytic_omega_3d(0.02, 7.2, 1000.0, m)
             / bench.analytic_omega(0.02, 7.2, 1000.0, m))
    assert ratio == pytest.approx(1.0, rel=0.02)   # both ~ m^3


def test_mode_estimator_flips_sign_under_a_ninety_degree_rotation(bench):
    """``s ~ x^2 - y^2``: the azimuthal symmetry the mode-2 estimator relies on."""
    pos, mass = bench.elliptical_drop(0.02, 0.002, 1000.0, 0.05)
    s0 = bench.mode_quadrupole(pos, mass)
    assert s0 > 0.0
    rot = torch.stack((-pos[:, 1], pos[:, 0]), dim=1)
    assert bench.mode_quadrupole(rot, mass) == pytest.approx(-s0, rel=1e-9)
    # the eigen estimator is |s|: it must NOT see the flip, which is exactly why
    # the frequency fit uses it and the phase-sensitive runs use the quadrupole
    assert bench.mode_amplitude(rot, mass)["s_moment"] == pytest.approx(
        bench.mode_amplitude(pos, mass)["s_moment"], rel=1e-9)


@pytest.mark.parametrize("m", [2, 3, 5])
def test_generalised_lobe_geometry_fills_the_analytic_lobe_area(bench, m):
    """``lobed_droplet`` (m2919 lineage) must enclose ``pi R^2 (1 + eps^2/2)``."""
    radius, dx, rho0, eps = 0.02, 0.001, 1000.0, 0.05
    pos, mass = bench.lobed_droplet(radius, dx, rho0, eps, m)
    dy = dx * math.sqrt(3.0) / 2.0
    cell = dx * dy
    area = pos.shape[0] * cell
    target = math.pi * radius ** 2 * (1.0 + 0.5 * eps ** 2)
    assert area == pytest.approx(target, rel=0.05)
    r = torch.linalg.vector_norm(pos, dim=1)
    # the outermost lobe tip is at R (1 + eps); nothing may poke outside it
    assert float(r.max()) <= radius * (1.0 + eps) + dx
    assert float(r.max()) >= radius * (1.0 + eps) - 2.0 * dx
    assert float(mass.sum()) == pytest.approx(rho0 * area, rel=1e-12)


@pytest.mark.slow
def test_without_surface_tension_the_mode_does_not_oscillate(bench):
    """Negative control ported from m2919: no sigma, no capillary frequency."""
    # Coarse lattice on purpose: the negative control asks whether the mode can
    # track omega_2 at all without surface tension, not for a converged number.
    rep = bench.run_case(radius=0.02, dx=0.004, sigma=0.0, eps=0.10, steps=1200)
    trace = rep["diagnostics"]["trace"]
    omega_ref = bench.analytic_omega(0.02, 7.2, 1000.0, 2)
    span = trace["t"][-1] - trace["t"][0]
    assert span > 0.5 * 2.0 * math.pi / omega_ref, (
        f"the control only covers {span:.4f} s, less than half a period")
    omega = bench.zero_crossing_period(trace["t"], trace["s_second_moment"])["omega"]
    assert not math.isfinite(omega) or abs(omega - omega_ref) / omega_ref > 0.3, (
        f"a drop without surface tension returned omega = {omega} rad/s, "
        f"within 30 % of the capillary value {omega_ref:.1f} rad/s")

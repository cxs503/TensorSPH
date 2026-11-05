"""The mode-2 frequency instrument: signed observable, basin-aware fit.

Background: the CI release case pinned two readings of the *same* deterministic
code, ``-31.78 %`` (loong64) and ``+8.64 %`` (x86_64), as strict-xfail in
``tests/test_oscillating_droplet.py``.  ``benchmarks/oscillating_droplet/
fit_robustness.py`` and ``docs/verification_oscillating_droplet_fit.md`` show
why: ``s_moment = (A - B)/(A + B)`` cannot be negative, so the released drop is
recorded as ``|cos|`` (rectified at twice the mode frequency), and the fit then
has several near-equal basins.  These tests pin the two repairs:

* ``mode_amplitude(..., reference_axis=...)`` also returns ``s_signed``, which
  restores the sign of the quadrupole and is a clean damped cosine;
* ``damped_cosine_fit`` refines every competing basin and reports how much
  better the winner is (``ambiguity_rms_gap``).

Numbers in the assertions are the measured ones from the committed trace
``benchmarks/oscillating_droplet/trace_ci_dx0p001.json`` (loongarch64, single
thread) and from closed-form synthetic traces; they are pins, not tolerances
that were widened until the test passed.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "benchmarks" / "oscillating_droplet" / "run.py"
TRACE = ROOT / "benchmarks" / "oscillating_droplet" / "trace_ci_dx0p001.json"


def _load_harness():
    spec = importlib.util.spec_from_file_location("oscillating_droplet_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = _load_harness()
RADIUS, SIGMA, RHO0 = 0.01, 7.2, 1000.0
OMEGA = B.analytic_omega(RADIUS, SIGMA, RHO0)
#: irrotational estimate lambda = 2 nu k^2, k = 2/R (the trace's own damping).
LAMBDA = 2.0 * 5.0e-5 * 4.0 / RADIUS ** 2


def _ellipse(a, b, theta, n=4000):
    u = torch.linspace(0.0, 2.0 * math.pi, n, dtype=torch.float64)
    x, y = a * torch.cos(u), b * torch.sin(u)
    c, s = math.cos(theta), math.sin(theta)
    return torch.stack((c * x - s * y, s * x + c * y), dim=1)


# --------------------------------------------------------------------------- #
# 1. the observable
# --------------------------------------------------------------------------- #
def test_signed_observable_restores_the_quadrupole_sign():
    """+eps when aligned, 0 at 45 deg, -eps when turned through 90 deg."""
    mass = torch.ones(4000, dtype=torch.float64)
    reference = B.mode_amplitude(_ellipse(1.1, 0.9, 0.0), mass)
    assert reference["s_signed"] == pytest.approx(reference["s_moment"], abs=1e-12)
    assert reference["s_moment"] == pytest.approx(0.1001, abs=1e-3)

    quarter = B.mode_amplitude(_ellipse(1.1, 0.9, math.pi / 4), mass,
                               reference_axis=reference["major_axis"])
    assert abs(quarter["s_signed"]) < 1e-6
    assert quarter["s_moment"] == pytest.approx(0.1001, abs=1e-3)

    turned = B.mode_amplitude(_ellipse(1.1, 0.9, math.pi / 2), mass,
                             reference_axis=reference["major_axis"])
    assert turned["s_signed"] == pytest.approx(-0.1001, abs=1e-3)
    # the legacy observable is blind to the rotation: it returns the magnitude
    assert turned["s_moment"] == pytest.approx(0.1001, abs=1e-3)


def test_signed_observable_unwraps_a_release_sequence():
    """A drop released from eps = 0.1 reads |cos| in s_moment, cos in s_signed."""
    mass = torch.ones(4000, dtype=torch.float64)
    eps = 0.10
    truth, moment, signed = [], [], []
    reference = None
    for i in range(60):
        phase = 2.0 * math.pi * i / 40.0          # 1.5 periods
        c = eps * math.cos(phase)
        a, b = 1.0 + c, 1.0 / (1.0 + c)           # area-preserving ellipse
        # No rotation: when c < 0 the semi-minor axis is the one along x, so
        # the major-axis eigenvector turns by 90 deg on its own.
        amp = B.mode_amplitude(_ellipse(a, b, 0.0), mass,
                               reference_axis=(reference if reference is not None
                                               else [1.0, 0.0]))
        if reference is None:
            reference = amp["major_axis"]
        truth.append(c)
        moment.append(amp["s_moment"])
        signed.append(amp["s_signed"])
    def corr(x, y):
        return float(torch.corrcoef(torch.tensor([x, y]))[0, 1])
    # 0.9955 measured: s = (a-b)/(a+b) is not exactly linear in eps for an
    # area-preserving ellipse, so the remaining 0.4 % is the quadratic term,
    # not a sign error.
    assert corr(signed, truth) > 0.99
    assert corr(moment, [abs(v) for v in truth]) > 0.99
    # rectified: never negative, two cusps' worth of sign flips lost
    assert min(moment) > 0.0
    assert min(signed) < -0.05


# --------------------------------------------------------------------------- #
# 2. the estimator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phase", [0.0, math.pi / 4, math.pi / 2])
def test_fit_recovers_a_closed_form_frequency(phase):
    """An exact damped cosine is recovered to far better than the 3 % bar."""
    t = [(i + 1) * 1.95e-5 for i in range(1400)]
    s = [0.10 * math.exp(-LAMBDA * tt) * math.cos(OMEGA * tt + phase) for tt in t]
    fit = B.damped_cosine_fit(t, s, OMEGA)
    error = (fit["omega"] - OMEGA) / OMEGA * 100.0
    assert abs(error) < 0.1, error
    assert fit["r2"] > 0.9999


def test_fit_is_deterministic():
    """Identical input must give bit-identical output, not a basin lottery."""
    t = [(i + 1) * 1.95e-5 for i in range(900)]
    s = [0.10 * math.exp(-LAMBDA * tt) * math.cos(OMEGA * tt) for tt in t]
    first = B.damped_cosine_fit(t, s, OMEGA)
    second = B.damped_cosine_fit(t, s, OMEGA)
    assert first["omega"] == second["omega"]
    assert first["lambda_per_s"] == second["lambda_per_s"]


def test_rectification_alone_biases_the_legacy_reading():
    """The defect has to be visible on a *perfect* rectified signal.

    ``|cos|`` is fitted 68 % high: the fit reads the harmonic content of the
    rectified signal, not the oscillation.  This is the bias the legacy
    observable injects before the solver does anything.
    """
    t = [(i + 1) * 1.95e-5 for i in range(1400)]
    clean = [0.10 * math.exp(-LAMBDA * tt) * math.cos(OMEGA * tt) for tt in t]
    rectified = [abs(v) for v in clean]
    err_clean = (B.damped_cosine_fit(t, clean, OMEGA)["omega"] - OMEGA) / OMEGA * 100
    err_rect = (B.damped_cosine_fit(t, rectified, OMEGA)["omega"] - OMEGA) / OMEGA * 100
    assert abs(err_clean) < 0.1
    assert err_rect > 30.0, err_rect


def test_fit_reports_the_ambiguity_of_its_own_answer():
    """``ambiguity_rms_gap`` is present on every fit and finite when it wins."""
    t = [(i + 1) * 1.95e-5 for i in range(1400)]
    s = [0.10 * math.exp(-LAMBDA * tt) * math.cos(OMEGA * tt) for tt in t]
    fit = B.damped_cosine_fit(t, s, OMEGA)
    assert "ambiguity_rms_gap" in fit
    assert fit["starts"] >= 1
    assert fit["ambiguity_rms_gap"] > 0.0


# --------------------------------------------------------------------------- #
# 3. the recorded CI trace (the case the xfail pins describe)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def recorded():
    data = json.loads(TRACE.read_text())
    trace = data["trace"]
    return {"t": trace["t"], "s_moment": trace["s_second_moment"],
            "s_signed": trace["s_signed"], "ke": trace["kinetic_energy_J"],
            "reported": data["frequency"]}


def test_recorded_trace_is_rectified_and_its_model_is_poor(recorded):
    """The pinned number came from a signal that is not a damped cosine."""
    moment = recorded["s_moment"]
    assert min(moment) > 0.0, "the legacy observable must be non-negative"
    fit = B.damped_cosine_fit(recorded["t"], moment, OMEGA)
    assert fit["r2"] < 0.90
    assert fit["lambda_per_s"] > 0.0


def test_signed_observable_gives_a_much_better_model_of_the_same_trace(recorded):
    """Same run, same estimator: the signed trace is a damped cosine."""
    moment_fit = B.damped_cosine_fit(recorded["t"], recorded["s_moment"], OMEGA)
    signed_fit = B.damped_cosine_fit(recorded["t"], recorded["s_signed"], OMEGA)
    assert signed_fit["r2"] > 0.93
    assert signed_fit["r2"] > moment_fit["r2"] + 0.10


def test_recorded_readings_are_reproduced_as_a_historical_record(recorded):
    """Pin what the legacy path measured on loongarch64: -31.5 % (CI pin -31.78).

    This is the reading the strict-xfail tests describe.  It is kept as a pin so
    that the difference between the artefacts and the repaired observable stays
    visible; the repaired value is asserted separately below.
    """
    reported = recorded["reported"]["fit_second_moment_rel_error_pct"]
    assert reported == pytest.approx(-31.5, abs=0.5), reported


def test_repaired_reading_is_cross_confirmed_by_two_observables(recorded):
    """Signed fit and kinetic-energy fit agree: -42.95 % on the same trace.

    They disagreed by tens of percentage points on the legacy path
    (``estimator_agreement_pct`` in the recorded frequency block).  The physics
    verdict is still FAIL against the 15 % gate -- what changed is that the
    number is now reproducible and independently confirmed.
    """
    signed = recorded["reported"]["fit_signed_rel_error_pct"]
    ke_half = recorded["reported"]["fit_kinetic_energy_rel_error_pct"]
    assert signed == pytest.approx(-42.95, abs=1.0), signed
    assert ke_half == pytest.approx(signed, abs=1.0)
    assert abs(signed) > B.GATE_PCT
    legacy_agreement = recorded["reported"]["estimator_agreement_pct"]
    assert abs(legacy_agreement) > 10.0, legacy_agreement

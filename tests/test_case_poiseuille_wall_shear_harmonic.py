"""Tests for the CASE-POISEUILLE-SPH harmonic wall-shear layer (gates W0-W6).

Three jobs, deliberately separated:

1. **Machinery** -- the kernel-free momentum-budget estimator
   ``tau_hat = rho H (g_hat - i omega <u>_hat)/2`` must reproduce the closed
   form ``mu (g_hat/i omega) alpha tanh(alpha H/2)`` *exactly* when it is fed the
   analytic mean velocity, and must respond to a perturbation in the mean by the
   corresponding amount.  An estimator that always passes is worse than none.
2. **Controls** -- the interior-quadratic local estimator (the steady-regime
   estimator of ``combined.py``) must be shown to be biased on the harmonic
   profile, resolution-independently, and to become exact as ``omega -> 0``.
   Without this the new gate could be a restatement of the old estimator.
3. **Published verdict** -- the committed ``out/wall_shear_harmonic.json`` must
   be reproducible from ``out/result.json``, must pass its own gates, and every
   number quoted in ``docs/case_poiseuille_wall_shear_harmonic.md`` must appear
   in the artifact (no hand-copied numbers).

Run:  pytest tests/test_case_poiseuille_wall_shear_harmonic.py -v
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "case_poiseuille"))

import wall_shear_harmonic as W  # noqa: E402  (the driver)

CASE = REPO / "benchmarks" / "case_poiseuille"
RESULT = CASE / "out" / "result.json"
ARTIFACT = CASE / "out" / "wall_shear_harmonic.json"
NOTE = REPO / "docs" / "case_poiseuille_wall_shear_harmonic.md"
FIGURE = REPO / "docs" / "figures" / "case_poiseuille_wall_shear_harmonic.png"

H, NU, G0, RHO = 0.1, 0.01, 9.81, 1000.0
OMEGA_SLOW, OMEGA_FAST = 13.0, 31.4

pytestmark = pytest.mark.skipif(not RESULT.is_file(), reason="ladder artifact absent")


# --------------------------------------------------------------------------- #
# Synthetic rows: an artifact made from the analytic profile at a real lattice
# --------------------------------------------------------------------------- #

def synthetic_rows(omega: float, n: int, scale: complex = 1.0 + 0j,
                   height: float = H, nu: float = NU, g0: float = G0):
    """Rows shaped like ``out/result.json`` but built from the closed form."""
    dx = height / n
    y = dx * 0.5 + dx * np.arange(n)
    uhat = W.analytic_profile(y, omega, height, g0, nu) * scale
    return [{"row": int(k), "y": float(y[k]), "y_over_H": float(y[k] / height),
             "amplitude": float(abs(uhat[k])),
             "phase": float(np.angle(uhat[k])),
             "amplitude_analytic": float(abs(uhat[k])),
             "phase_analytic": float(np.angle(uhat[k])),
             "fit_residual": 0.0}
            for k in range(n)]


def synthetic_artifact(omega_slow: float = OMEGA_SLOW,
                       omega_fast: float = OMEGA_FAST,
                       scale: complex = 1.0 + 0j) -> dict:
    geometry = {"height": H, "viscosity": NU, "rho0": RHO, "body_force": G0,
                "columns": 8}
    runs = []
    for n in (8, 12, 16):
        rows = synthetic_rows(omega_slow, n, scale=scale)
        runs.append({"resolution_H_over_dx": n, "omega": omega_slow, "rows": rows,
                     "womersley_number": 0.5 * H * math.sqrt(omega_slow / NU),
                     "stokes_layer": math.sqrt(2.0 * NU / omega_slow)})
    fast = {"resolution_H_over_dx": 12, "omega": omega_fast,
            "rows": synthetic_rows(omega_fast, 12, scale=scale),
            "womersley_number": 0.5 * H * math.sqrt(omega_fast / NU),
            "stokes_layer": math.sqrt(2.0 * NU / omega_fast)}
    return {"geometry": geometry, "harmonic_runs": runs, "harmonic_fast": fast}


# --------------------------------------------------------------------------- #
# 1. Machinery
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("omega", [0.5, 13.0, 31.4, 120.0])
@pytest.mark.parametrize("height,viscosity", [(0.1, 0.01), (0.05, 1e-3), (0.2, 0.05)])
def test_budget_identity_matches_the_closed_form(omega, height, viscosity):
    """(1) and (2) are the same expression -- through *different* routes.

    (1) integrates the measured mean velocity, (2) differentiates the closed
    form; a sign or ``H/2`` slip in either would break this at machine level.
    """
    mean = W.analytic_mean_amplitude(omega, height, G0, viscosity)
    via_budget = W.budget_wall_shear(mean, omega, height, G0, RHO)
    closed = W.analytic_wall_shear(omega, height, G0, viscosity, RHO)
    assert abs(via_budget - closed) / abs(closed) < 1e-12


@pytest.mark.parametrize("height,viscosity", [(0.1, 0.01), (0.05, 1e-3)])
def test_steady_limit_is_rho_g_H_over_2(height, viscosity):
    """``omega -> 0``: the oscillatory wall shear tends to the steady value."""
    closed = W.analytic_wall_shear(1e-9, height, G0, viscosity, RHO)
    assert abs(abs(closed) - RHO * G0 * height / 2.0) / (RHO * G0 * height / 2.0) < 1e-6


def test_closed_form_derivative_is_consistent_at_both_walls():
    """``tau(H) = -tau(0)``: verified by differentiating the profile numerically."""
    omega = OMEGA_SLOW
    y = np.linspace(0.0, H, 400001)
    uhat = W.analytic_profile(y, omega, H, G0, NU)
    slope = np.gradient(uhat, y)
    tau0 = RHO * NU * slope[0]
    tauH = RHO * NU * slope[-1]
    tau_hat = W.analytic_wall_shear(omega, H, G0, NU, RHO)
    assert abs(tau0 - tau_hat) / abs(tau_hat) < 1e-4
    assert abs(tauH + tau_hat) / abs(tau_hat) < 1e-4


def _quadrature_error(n: int, omega: float = OMEGA_SLOW) -> float:
    mean = W.measured_mean_amplitude(synthetic_rows(omega, n))
    exact = W.analytic_mean_amplitude(omega, H, G0, NU)
    return abs(mean - exact) / abs(exact)


def test_measured_mean_amplitude_recovers_the_section_mean():
    """The row average is a midpoint quadrature of the section mean.

    It is therefore *second-order* accurate, not exact: at ``H/dx = 8`` the
    row spacing is ``dx/delta = 0.32`` Stokes layers, so the quadrature error
    is ~1 %, and it is the only reason the synthetic analytic input does not
    reproduce the closed form to machine precision.
    """
    for n, tol in ((8, 1e-2), (12, 4e-3), (16, 2.5e-3)):
        assert _quadrature_error(n) < tol


def test_row_mean_quadrature_error_shrinks_with_resolution():
    """Midpoint quadrature: the section mean converges as ``dx^2``."""
    errs = [_quadrature_error(n) for n in (8, 16, 32, 64)]
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert errs[3] < 2e-4
    # second order: doubling the resolution must cut the error ~4x
    for coarse, fine in zip(errs, errs[1:]):
        assert 3.0 < coarse / fine < 5.0


def test_estimator_responds_to_a_perturbed_mean_velocity():
    """A 1 % change in ``<u>`` must be *visible* in the reported amplitude error.

    Sensitivity is attenuated by the quasi-steady term: ``tau = rho H (g - i omega
    <u>)/2`` so ``d tau/tau = -i omega d<u> / (g - i omega <u>)``, and at
    ``omega = 13`` the two terms partially cancel in ``|tau|``; the measured
    gain is ~0.2 % of ``|tau|`` per 1 % of ``<u>``.  The test pins that gain so
    a silent sign/kernel bug cannot pass as "insensitive by design".
    """
    base = W.build(synthetic_artifact())
    pert = W.build(synthetic_artifact(scale=1.01 + 0j))
    delta = (pert["harmonic_runs"][-1]["wall_shear_budget"]["amp_error_pct"]
             - base["harmonic_runs"][-1]["wall_shear_budget"]["amp_error_pct"])
    assert 0.15 < abs(delta) < 0.30, delta
    # and the analytic prediction of that gain, from the closed forms alone
    u = W.analytic_mean_amplitude(OMEGA_SLOW, H, G0, NU)
    v = G0 - 1j * OMEGA_SLOW * u          # g - i omega <u>, up to the rho H/2 factor
    dv = -1j * OMEGA_SLOW * (0.01 * u)    # response to a +1 % change of <u>
    gain = (np.real(np.conj(v) * dv) / abs(v) ** 2) * 100.0  # d|tau|/|tau| in %
    assert delta == pytest.approx(gain, abs=0.02)


def quasi_steady_artifact() -> dict:
    """A mean-momentum-balance-only channel: steady parabola, zero phase."""
    art = synthetic_artifact()
    steady = W.analytic_profile(np.zeros(1), 1e-9, H, G0, NU)  # unused placeholder
    for run in art["harmonic_runs"] + [art["harmonic_fast"]]:
        n = run["resolution_H_over_dx"]
        dx = H / n
        y = dx * 0.5 + dx * np.arange(n)
        u = G0 * (H - y) * y / (2.0 * NU)  # steady parabola, no phase lag
        run["rows"] = [{"row": int(k), "y": float(y[k]), "y_over_H": float(y[k] / H),
                        "amplitude": float(u[k]), "phase": 0.0,
                        "amplitude_analytic": float(u[k]), "phase_analytic": 0.0,
                        "fit_residual": 0.0} for k in range(n)]
    del steady
    return art


def test_a_quasi_steady_channel_fails_the_gate():
    """Falsification: the mean momentum balance alone must NOT pass W1-W4.

    A solver that followed the forcing with the steady parabola is the exact
    alternative hypothesis this gate exists to reject; it is rejected at
    +133 % amplitude and -7.3 deg phase.
    """
    verdict = W.build(quasi_steady_artifact())
    assert not verdict["gates"]["W1_wall_shear_amplitude_5pct"]
    assert not verdict["gates"]["W3_all_rungs_5pct"]
    assert not verdict["gates"]["W4_frequency_ratio_5pct"]
    assert not verdict["all_gates_pass"]
    assert verdict["quasi_steady_control"]["measured"]["amp_error_pct"] > 50.0


def test_synthetic_analytic_artifact_passes_every_gate():
    """Sanity: with the analytic profile as input, the estimator is exact."""
    verdict = W.build(synthetic_artifact())
    assert verdict["all_gates_pass"], verdict["gates"]
    finest = max(verdict["harmonic_runs"], key=lambda r: r["resolution_H_over_dx"])
    # residual is the row-midpoint quadrature of <u> (O(dx^2)), nothing else
    assert abs(finest["wall_shear_budget"]["amp_error_pct"]) < 2e-3
    assert abs(finest["wall_shear_budget"]["phase_error_deg"]) < 0.5


# --------------------------------------------------------------------------- #
# 2. Controls: the local quadratic estimator
# --------------------------------------------------------------------------- #

def test_local_estimator_is_exact_for_the_steady_parabola():
    """``omega -> 0``: the quadratic fit is unbiased, as ``combined.py`` relies on."""
    omega = 1e-6
    y = H / 16 * 0.5 + H / 16 * np.arange(16)
    uhat = W.analytic_profile(y, omega, H, G0, NU)
    tau = W.quadratic_local_wall_shear(y, uhat, H, RHO, NU)
    exact = W.analytic_wall_shear(omega, H, G0, NU, RHO)
    assert abs(tau - exact) / abs(exact) < 1e-3


def test_local_estimator_is_biased_in_the_harmonic_regime():
    """Gate W5: > 5 % amplitude bias, and resolution-independent (model bias)."""
    rec = W.estimator_transferability(OMEGA_SLOW, {"height": H, "viscosity": NU,
                                                   "rho0": RHO, "body_force": G0})
    assert rec["is_biased_beyond_case_tolerance"], rec
    assert rec["is_resolution_independent"], rec
    assert abs(rec["per_rung"]["16"]["phase_error_deg"]) > 10.0


def test_local_estimator_bias_grows_with_womersley_number():
    """The bias is a Stokes-layer effect: monotone in ``Wo`` over 2..40."""
    geo = {"height": H, "viscosity": NU, "rho0": RHO, "body_force": G0}
    sweep = W.sweep_estimator_bias(geo, omegas=(2.0, 5.0, 10.0, 20.0, 40.0))
    bias = [abs(s["amp_error_pct"]) for s in sweep]
    assert bias == sorted(bias), bias
    assert bias[-1] > bias[0]


# --------------------------------------------------------------------------- #
# 3. Published verdict + notes
# --------------------------------------------------------------------------- #

def test_committed_artifact_is_reproducible_from_the_ladder():
    if not ARTIFACT.is_file():
        pytest.skip("wall-shear artifact absent")
    published = json.loads(ARTIFACT.read_text())
    rebuilt = W.build(json.loads(RESULT.read_text()))
    for key in ("gates", "all_gates_pass", "estimator_transferability",
                "reference_selfcheck"):
        assert published[key] == rebuilt[key], key
    for a, b in zip(published["harmonic_runs"], rebuilt["harmonic_runs"]):
        assert a["resolution_H_over_dx"] == b["resolution_H_over_dx"]
        for field in ("wall_shear_budget", "mean_velocity",
                      "wall_shear_local_quadratic"):
            for name, value in b[field].items():
                assert a[field][name] == pytest.approx(value, rel=1e-12), name


def test_committed_artifact_passes_every_gate():
    if not ARTIFACT.is_file():
        pytest.skip("wall-shear artifact absent")
    art = json.loads(ARTIFACT.read_text())
    assert art["all_gates_pass"], art["gates"]
    assert set(art["gates"]) == {
        "W0_mean_amplitude_5pct",
        "W0b_mean_phase_5pct_of_quarter_turn",
        "W1_wall_shear_amplitude_5pct",
        "W2_wall_shear_phase_5pct_of_quarter_turn",
        "W3_all_rungs_5pct",
        "W4_frequency_ratio_5pct",
        "W5_local_estimator_control_biased_gt_5pct",
        "W6_reference_selfcheck_1e-10",
    }


def test_published_numbers_are_below_the_case_tolerance():
    """< 5 % everywhere, and the drag ratio discriminates against quasi-steady."""
    if not ARTIFACT.is_file():
        pytest.skip("wall-shear artifact absent")
    art = json.loads(ARTIFACT.read_text())
    for rec in art["harmonic_runs"]:
        assert abs(rec["wall_shear_budget"]["amp_error_pct"]) < 5.0
        assert rec["wall_shear_budget"]["phase_error_pct_quarter_turn"] < 5.0
    disc = art["frequency_discrimination"]
    assert abs(disc["ratio_error_pct"]) < 5.0
    # a quasi-steady solver would give 1.00; the measurement must not
    assert abs(disc["ratio_measured"] - 1.0) > 0.2


def test_note_quotes_the_artifact():
    """No hand-copied numbers: the note must carry the artifact's own values."""
    if not ARTIFACT.is_file() or not NOTE.is_file():
        pytest.skip("note or artifact absent")
    art = json.loads(ARTIFACT.read_text())
    text = NOTE.read_text()
    finest = max(art["harmonic_runs"], key=lambda r: r["resolution_H_over_dx"])
    for value, spec in (
            (finest["wall_shear_budget"]["measured_abs"], ".4f"),
            (finest["wall_shear_budget"]["analytic_abs"], ".4f"),
            (finest["wall_shear_budget"]["amp_error_pct"], "+.3f"),
            (finest["wall_shear_budget"]["phase_error_deg"], "+.3f"),
    ):
        assert format(value, spec) in text, (value, spec)
    assert format(art["frequency_discrimination"]["ratio_measured"], ".4f") in text
    assert format(art["frequency_discrimination"]["ratio_analytic"], ".4f") in text
    assert format(art["reference_selfcheck"]["budget_identity_rel_error"], ".2e") in text


def test_readme_quotes_the_artifact():
    """The README summary must not drift from the artifact either."""
    if not ARTIFACT.is_file():
        pytest.skip("wall-shear artifact absent")
    readme = CASE / "README.md"
    if not readme.is_file():
        pytest.skip("README absent")
    text = readme.read_text()
    art = json.loads(ARTIFACT.read_text())
    finest = max(art["harmonic_runs"], key=lambda r: r["resolution_H_over_dx"])
    for value, spec in (
            (finest["wall_shear_budget"]["measured_abs"], ".4f"),
            (finest["wall_shear_budget"]["analytic_abs"], ".4f"),
            (finest["wall_shear_budget"]["amp_error_pct"], "+.3f"),
            (finest["wall_shear_budget"]["phase_error_deg"], "+.3f"),
            (art["frequency_discrimination"]["ratio_measured"], ".4f"),
    ):
        assert format(value, spec) in text, (value, spec)


def test_note_and_figure_exist():
    assert NOTE.is_file(), NOTE
    assert FIGURE.is_file(), FIGURE
    assert FIGURE.stat().st_size > 20000  # a blank canvas would be tiny


def test_wrap_deg_is_an_angle():
    assert W.wrap_deg(0.0) == pytest.approx(0.0)
    assert W.wrap_deg(370.0) == pytest.approx(10.0)
    assert W.wrap_deg(-190.0) == pytest.approx(170.0)
    assert W.wrap_deg(180.0) == pytest.approx(-180.0)  # half-open range
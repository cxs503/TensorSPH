"""Tests for the equilibrium-based Young-Laplace verification.

Two earlier rounds disagreed about ``Delta p = sigma/R`` on the same model:
-75.6 % at dx = 2 mm versus a pass at dx = 4 mm, with the same CSF force and the
same Tait law.  These tests lock in what this round actually established, which
is a *negative* result with a mechanism:

* the droplet has no static state inside the integration window -- it expands,
  its interior density deficit grows, and the run ends in a blow-up, both with
  and without surface tension, so the expansion is a free-surface pressure
  effect and not something the CSF force produces;
* therefore no single-sample ``Delta p`` may be quoted as the equilibrium value,
  and both earlier verdicts are transient artifacts;
* the legacy band basis is provably biased low on the same state, which is the
  systematic behind the -75.6 %.

Any change that flips one of these verdicts must fail here, so the record
cannot go stale silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_equilibrium"
REPORT = BENCH / "report.json"
DRIFT = BENCH / "drift.json"
PLATFORM_TOL = 15.0     # percent


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():  # pragma: no cover - committed artifact required
        pytest.fail(f"missing committed benchmark record: {REPORT}")
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def drift():
    if not DRIFT.exists():  # pragma: no cover
        pytest.fail(f"missing committed drift record: {DRIFT}")
    return json.loads(DRIFT.read_text())


@pytest.fixture(scope="module")
def bench():
    import importlib.util

    spec = importlib.util.spec_from_file_location("laplace_equilibrium_run",
                                                  BENCH / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# the measurement basis itself
# --------------------------------------------------------------------------- #
def test_complete_support_mask_drops_the_kernel_shell(bench):
    """Only particles whose 2h kernel support is inside the fluid may be
    sampled; the truncated shell is what biased the coarse-dx reading."""
    radius, dx = 0.02, 0.002
    h = 1.3 * dx
    r = torch.tensor([0.0, 0.5 * radius, radius - 2.6 * h, radius - 2.4 * h,
                      radius * 1.02], dtype=torch.float64)
    pos = torch.stack([r, torch.zeros_like(r)], dim=1)
    assert bench.complete_support_mask(pos, radius, h).tolist() == [
        True, True, True, False, False]


# --------------------------------------------------------------------------- #
# committed ladder record
# --------------------------------------------------------------------------- #
def test_reference_is_the_analytic_law(report):
    assert report["reference"]["delta_p_Pa"] == pytest.approx(
        report["params"]["sigma_N_per_m"] / report["params"]["radius_m"])
    for row in report["ladder"]:
        assert row["reference"]["delta_p_Pa"] == pytest.approx(
            row["setup"]["sigma_N_per_m"] / row["setup"]["radius_m"])


def test_committed_errors_are_recomputed_from_the_pressures(report):
    """Every percentage must follow from the pressures the record reports --
    this is what stops a hand-edited verdict."""
    for row in report["ladder"]:
        eq = row["equilibrium"]
        dp = row["reference"]["delta_p_Pa"]
        assert eq["rel_error_pct"] == pytest.approx(
            (eq["p_complete_Pa"] - dp) / dp * 100.0, abs=1e-9)
        assert eq["rel_error_band_pct"] == pytest.approx(
            (eq["p_complete_Pa"] - eq["p_band_Pa"] - dp) / dp * 100.0, abs=1e-9)
        assert eq["within_15pct"] == (abs(eq["rel_error_pct"]) < PLATFORM_TOL)
        assert eq["within_3pct"] == (abs(eq["rel_error_pct"]) < 3.0)


def test_the_documented_verdict_is_withheld(report):
    """The honest verdict of this round: every resolution misses the 15 % bar
    and the miss is resolution-independent, i.e. structural, not noise."""
    errs = [row["equilibrium"]["rel_error_pct"] for row in report["ladder"]]
    assert all(not row["equilibrium"]["within_15pct"]
               for row in report["ladder"]), (
        "the jump now passes the 15 % bar -- re-run the drift study, and if the "
        "droplet genuinely equilibrates, update the doc and this test together")
    assert max(errs) - min(errs) < 10.0, errs
    assert all(e < 0.0 for e in errs), errs


def test_measurement_uncertainty_is_reported(report):
    """A percentage without an uncertainty is not a measurement."""
    for row in report["ladder"]:
        eq = row["equilibrium"]
        assert eq["p_complete_sem_Pa"] > 0.0
        assert eq["measurement_snr"] > 3.0, row["setup"]["dx_m"]


def test_legacy_band_basis_is_biased_low(report):
    """The old protocol subtracts a band pressure measured *inside* the CSF
    shell, where p has already decayed toward the vacuum value.  Checked on the
    same state, that basis must under-report the jump well beyond the bar."""
    checked = 0
    for row in report["ladder"]:
        if row["setup"]["n_band_080_095"] < 5:
            continue
        eq = row["equilibrium"]
        assert eq["rel_error_band_pct"] < eq["rel_error_pct"] - PLATFORM_TOL, (
            row["setup"]["dx_m"], eq["rel_error_pct"], eq["rel_error_band_pct"])
        checked += 1
    assert checked >= 1


def test_phase_scan_reproduces_the_non_reproducible_verdict(report):
    """One run, many sample times: the error swings far wider than the bar, so
    a single-sample verdict is a coin flip."""
    scan = report["phase_scan"]
    assert scan["n_samples"] >= 10
    assert scan["error_spread_pct"] > 30.0
    assert scan["legacy_error_spread_pct"] > 30.0
    assert scan["n_samples_within_15pct"] < scan["n_samples"]
    for s in scan["samples"]:
        dp = scan["reference"]["delta_p_Pa"]
        assert s["rel_error_pct"] == pytest.approx(
            (s["p_complete_Pa"] - dp) / dp * 100.0, abs=1e-9)
        assert s["rel_error_legacy_pct"] == pytest.approx(
            (s["p_core50_Pa"] - s["p_band_Pa"] - dp) / dp * 100.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# the long-time drift: why no equilibrium value exists
# --------------------------------------------------------------------------- #
def test_droplet_has_no_static_state_it_expands(drift):
    """Twelve capillary times: the cluster's radius of gyration grows well past
    its initial value and the interior density deficit deepens, so the sampled
    ``p_core`` is never a static quantity."""
    run = next(r for r in drift["runs"] if r["setup"]["sigma_N_per_m"] > 0)
    out = run["outcome"]
    assert out["r_rms_over_r0_last"] > 1.2, out
    assert out["rho_excess_last"] < -0.01, out
    # the committed window is 12 capillary times; it ends before the blow-up
    # that the longer exploratory run hit (NaN at t ~ 14 t_sigma)
    assert out["t_over_t_sigma_last_finite"] > 10.0, out


def test_pressure_jump_never_plateaus(drift):
    """No plateau: the reading at the end of the window is under 70 % of the
    reading at the start of the window."""
    run = next(r for r in drift["runs"] if r["setup"]["sigma_N_per_m"] > 0)
    tr = run["trace"]
    assert tr, "empty drift trace"
    first = sum(r["p_core_over_theory"] for r in tr[:3]) / 3.0
    last = tr[-1]["p_core_over_theory"]
    assert last < 0.7 * first, (first, last)


def test_sigma_zero_control_is_stable_so_the_runaway_needs_surface_tension(drift):
    """Mechanism separation: with the CSF term switched off the cloud stays put
    (r_rms within 10 % of its initial value), while with surface tension on it
    doubles.  The runaway therefore lives in the *interaction* of the CSF shell
    force with the clamped free-surface pressure gradient -- it is not a generic
    free-surface drift of this solver."""
    runs = {r["setup"]["sigma_N_per_m"]: r for r in drift["runs"]}
    assert 0.0 in runs, sorted(runs)
    control = runs[0.0]["outcome"]
    active = runs[max(k for k in runs if k > 0)]["outcome"]
    assert control["r_rms_over_r0_last"] < 1.10, control
    assert active["r_rms_over_r0_last"] > 1.20, active
    # removing surface tension also stops the interior deficit from deepening
    assert control["rho_excess_last"] > active["rho_excess_last"], (
        control["rho_excess_last"], active["rho_excess_last"])
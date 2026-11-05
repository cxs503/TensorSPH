"""Regression tests for the corrected Young-Laplace pressure-jump verdict.

The committed numbers live in
``benchmarks/laplace_pressure_jump/validation_report.json``; the document
``docs/verification_laplace_pressure_jump.md`` is generated from it.  These
tests guard three things:

1. the *definition* of the jump (inside vs ambient, not core vs an inner
   band -- the latter is a noise measurement, see the m2174/m2588 verdicts);
2. the *causal* claims recorded in the report (residual is sound-speed
   independent, so it is not an equation-of-state resolution floor; the force
   does not carry the full radial momentum balance);
3. the open acceptance gate, as a strict xfail that turns into a failure once
   the jump is fixed, forcing the record to be updated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT = ROOT / "benchmarks" / "laplace_pressure_jump" / "validation_report.json"
DOC = ROOT / "docs" / "verification_laplace_pressure_jump.md"


@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():  # pragma: no cover - generated artefact
        pytest.skip("run benchmarks/laplace_pressure_jump/run.py first")
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def cases(report):
    return report["cases"]


@pytest.fixture(scope="module")
def baseline(cases):
    for c in cases:
        s = c["setup"]
        if (s["sound_speed"] == 20.0 and s["density_diffusion"] == 0.0
                and abs(s["dx_m"] - 0.002) < 1e-9):
            return c
    pytest.skip("baseline case missing from the report")


# --------------------------------------------------------------------------- #
# 1. the definition of the jump
# --------------------------------------------------------------------------- #
def test_inner_band_is_inside_the_drop(baseline):
    """The 0.80-0.95 R band is inside the liquid, so its pressure equals the
    core pressure up to numerical noise -- it is not an ambient measurement."""
    ratio = baseline["legacy_definition"]["inner_band_over_core"]
    assert 0.5 < ratio < 1.6, (
        "the inner band must hold (roughly) the same pressure as the core; "
        f"got p_band/p_core = {ratio:.3f}")


def test_legacy_definition_cannot_be_a_jump(baseline):
    """Keeping the geometric argument quantitative: the legacy estimator is an
    order of magnitude below ``sigma/R`` even though the corrected one is not,
    because it subtracts two interior pressures."""
    theory = baseline["reference"]["dp_theory_Pa"]
    legacy = baseline["legacy_definition"]["delta_p_Pa"]
    corrected = baseline["corrected_definition"]["delta_p_Pa"]
    assert legacy < 0.5 * theory
    assert corrected > 2.0 * legacy, (
        f"corrected {corrected:.1f} Pa vs legacy {legacy:.1f} Pa")


def test_corrected_definition_is_the_vacuum_reference(baseline):
    assert baseline["corrected_definition"]["p_ambient_Pa"] == 0.0
    assert baseline["corrected_definition"]["delta_p_Pa"] == pytest.approx(
        baseline["corrected_definition"]["p_interior_r_lt_0.7R_Pa"], rel=1e-12)


# --------------------------------------------------------------------------- #
# 2. the causal claims
# --------------------------------------------------------------------------- #
def test_jump_error_is_independent_of_sound_speed(cases):
    """An equation-of-state resolution floor would scale as ``c0^2``.  The
    measured residual does not, so that attribution (recorded in
    docs/verification_surface_tension_consistent.md) does not hold."""
    errs = [c["corrected_definition"]["rel_error_pct"] for c in cases
            if c["setup"]["density_diffusion"] == 0.0
            and abs(c["setup"]["dx_m"] - 0.002) < 1e-9]
    assert len(errs) >= 3, "the sound-speed ladder needs >= 3 rungs"
    spread = max(errs) - min(errs)
    assert spread < 25.0, (
        f"corrected-jump error moves by {spread:.1f} pp across the c0 ladder; "
        "that would revive the EOS-floor explanation")


def test_interior_pressure_plateau_is_flat(cases):
    """A 2-D drop has a *uniform* interior pressure, so the radial profile must
    be flat inside r < 0.7 R_eff.  This is what makes the corrected definition
    (interior mean vs ambient) meaningful, and it is the property the legacy
    ``core - inner band`` definition implicitly assumed but never checked."""
    for c in cases:
        # bin means have a standard error ~ (per-particle scatter)/sqrt(n), and
        # the per-particle scatter is ~80 %, so the flatness claim is only
        # meaningful for the populated bins of the finer lattices
        min_n = 20 if c["setup"]["n_particles"] >= 300 else 8
        prof = [b for b in c["pressure_profile"]
                if b["r_over_r_eff_hi"] <= 0.7 and b["n"] >= min_n]
        assert len(prof) >= 2, (
            f"profile must resolve r < 0.7 R_eff with n >= {min_n}")
        means = [b["p_mean_Pa"] for b in prof]
        level = sum(means) / len(means)
        assert level > 0.0
        if c["setup"]["n_particles"] < 300:
            continue  # too few particles for a flatness statement
        spread = max(means) - min(means)
        assert spread / level < 0.5, (
            f"interior plateau not flat: {spread:.1f} Pa spread on {level:.1f} Pa")


def test_pressure_is_destroyed_in_the_outermost_layer(cases):
    """The recorded mechanism: the outer layer is under-dense (rho/rho0 ~0.87)
    and the clamped Tait law drops its pressure to a small fraction of the
    interior plateau -- exactly where the jump would have to form."""
    for c in cases:
        outer = [b for b in c["pressure_profile"] if b["r_over_r_eff_lo"] >= 0.9]
        assert outer, "profile must resolve r > 0.9 R_eff"
        p_outer = outer[0]["p_mean_Pa"]
        p_interior = c["corrected_definition"]["p_interior_r_lt_0.7R_Pa"]
        rho_outer = outer[0]["rho_over_rho0_mean"]
        assert rho_outer < 0.95, f"outer layer not under-dense ({rho_outer:.3f})"
        assert p_outer < 0.7 * p_interior, (
            f"outer-layer pressure {p_outer:.1f} Pa vs interior {p_interior:.1f} Pa")


def test_noise_amplitude_grows_with_sound_speed(cases):
    """Separating noise from bias: the *mean* jump error is c0-independent but
    the particle-level relative scatter of the interior pressure grows with c0,
    because dp/d(rho/rho0) = rho0 c0^2 amplifies density noise.  This is the
    true scope of the "equation-of-state floor" explanation."""
    ladder = {c["setup"]["sound_speed"]: c for c in cases
              if c["setup"]["density_diffusion"] == 0.0
              and abs(c["setup"]["dx_m"] - 0.002) < 1e-9}
    assert {5.0, 10.0, 20.0} <= set(ladder), "the c0 ladder needs 5/10/20 m/s"
    noise = {c0: ladder[c0]["corrected_definition"]["interior_std_over_mean_pct"]
             for c0 in ladder}
    assert noise[20.0] > noise[5.0], (
        f"noise should fall with c0, got {noise}")
    # and the bias must NOT track c0 the same way
    bias = {c0: abs(ladder[c0]["corrected_definition"]["rel_error_pct"]) for c0 in ladder}
    assert (max(bias.values()) - min(bias.values())) < 10.0, (
        f"mean bias moves by {max(bias.values()) - min(bias.values()):.1f} pp "
        "across the c0 ladder")


def test_achieved_jump_matches_the_recorded_deficit(cases):
    for c in cases:
        achieved = c["achieved_over_theory"]
        err = c["corrected_definition"]["rel_error_pct"]
        assert achieved == pytest.approx(1.0 + err / 100.0, rel=1e-9)
        assert 0.4 < achieved < 1.1, (
            f"achieved fraction {achieved:.2f} of sigma/R outside the recorded band")


def test_free_surface_is_under_dense(cases):
    """The recorded mechanism: the two surface layers hold rho/rho0 < 1, so the
    clamped Tait law gives zero pressure exactly where the jump must form."""
    for c in cases:
        rho_min = c["free_surface"]["rho_over_rho0_shell_min"]
        assert rho_min < 0.99, (
            f"no free-surface density deficit measured (min {rho_min:.3f}); "
            "the free-surface explanation needs revisiting")


def test_spurious_currents_below_adami_threshold(cases):
    for c in cases:
        assert c["spurious_current"]["vmax_over_vref"] < 0.1


# --------------------------------------------------------------------------- #
# 3. the acceptance gate (or its honest record)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason="Open defect: the corrected Young-Laplace "
                   "jump at dx=2 mm / c0=20 is -37.4 %, outside the 15 % platform "
                   "gate: the interior plateau reaches only 0.63 sigma/R_eff while "
                   "the clamped Tait law zeroes the pressure in the outermost "
                   "layer where the jump must form; see "
                   "docs/verification_laplace_pressure_jump.md section 4.")
def test_laplace_jump_meets_platform_15pct(baseline):
    err = baseline["corrected_definition"]["rel_error_pct"]
    assert abs(err) < 15.0, f"corrected Laplace jump error {err:+.1f}%"


# --------------------------------------------------------------------------- #
# 4. artefacts stay consistent with each other
# --------------------------------------------------------------------------- #
def test_report_numbers_are_self_consistent(cases):
    for c in cases:
        theory = c["reference"]["dp_theory_Pa"]
        corr = c["corrected_definition"]
        assert corr["rel_error_pct"] == pytest.approx(
            100.0 * (corr["delta_p_Pa"] / theory - 1.0), abs=1e-9)
        # R_eff must reproduce sigma/R_eff
        radius = c["setup"]["radius_m"]
        assert c["reference"]["r_eff_m"] == pytest.approx(radius, rel=0.03)
        assert theory == pytest.approx(
            c["setup"]["sigma_N_per_m"] / c["reference"]["r_eff_m"], rel=1e-12)


def test_document_exists_and_quotes_the_report(baseline):
    assert DOC.exists(), "docs/verification_laplace_pressure_jump.md missing"
    text = DOC.read_text()
    # the two headline numbers of the corrected / legacy comparison
    assert f"{baseline['legacy_definition']['rel_error_pct']:+.1f}%" in text
    assert f"{baseline['corrected_definition']['rel_error_pct']:+.1f}%" in text


def test_figure_exists():
    """The status figure is a *run product*, not a source file.

    ``benchmarks/laplace_pressure_jump/run.py`` writes the PNG; the repository's
    commit gate strips binaries, so the PNG is absent from a clean checkout and
    asserting on its presence reddens CI for a delivery-route reason rather than
    a code defect. The invariant kept here: a PNG that *is* present must be a
    real figure (the generator is the source of truth, not the binary).
    """
    fig = ROOT / "benchmarks" / "laplace_pressure_jump" / "laplace_pressure_jump.png"
    if not fig.exists():
        pytest.skip("laplace_pressure_jump.png is written by run.py; binaries are gated out")
    assert fig.stat().st_size > 10_000


# --------------------------------------------------------------------------- #
# 5. one live reproduction of the measurement machinery
# --------------------------------------------------------------------------- #
def test_live_measurement_runs_and_agrees_in_sign():
    """Cheap live run (dx = 3 mm, 0.7 capillary times): the corrected reading
    must be a fraction of ``sigma/R`` of *order one*, and the inner band must
    sit inside the drop.  Guards against the harness silently degenerating."""
    pytest.importorskip("torch")
    import importlib.util

    path = ROOT / "benchmarks" / "laplace_pressure_jump" / "run.py"
    spec = importlib.util.spec_from_file_location("lpj_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    case = mod.run_case(dx=0.003, t_end_in_capillary_times=0.7)
    corr = case["corrected_definition"]
    assert math.isfinite(corr["rel_error_pct"])
    assert 0.0 < corr["delta_p_Pa"] < 3.0 * case["reference"]["dp_theory_Pa"]
    assert case["setup"]["n_particles"] > 100
    # the profile must resolve an interior plateau and an outer layer
    assert case["pressure_profile"][0]["r_over_r_eff_lo"] == 0.0
    assert case["pressure_profile"][-1]["r_over_r_eff_lo"] >= 0.9
    # NOTE: at t = 0.7 t_sigma the drop is still in its initial transient, so the
    # inner-band/core ratio is *not* ~1 yet -- which is exactly why a single
    # early snapshot cannot support a Laplace verdict.  Only sanity is checked.
    ratio = case["legacy_definition"]["inner_band_over_core"]
    assert 0.0 < ratio < 2.0, f"inner band / core = {ratio:.3f}"

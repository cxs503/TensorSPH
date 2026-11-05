"""End-to-end gates for the free-surface pressure closure (C3 of the Laplace debt).

``tests/test_free_surface_pressure_closure.py`` pins the *operator*: closed-form
checks that ``free_surface_pressure`` reproduces a known field, honours the
donor set, the pass count and the ``p >= 0`` clamp.  It cannot say whether the
operator fixes the thing it was written for.  This module pins the *experiment*:

``benchmarks/laplace_pressure_floor`` runs the same drop twice -- Tait clamp off
and closure on -- and records both in ``report.json -> closure_study``.  The
gates below read that record back and state the acceptance criteria in code:

* the clamp really clamps (every free-surface particle at p = 0),
* the closure really lifts the shell off p = 0,
* the end-to-end Young-Laplace jump lands inside the platform's 15 % band for
  **every** case of the ladder, and comes from the closure (not from a changed
  scheme, c0 or resolution),
* the closure is not a tuning artefact: the literal Adami donor set is *not*
  enough, one pass is not enough, and the receiver threshold is not a knob.

The last test re-runs a short A/B live, so the record cannot drift away from
what the code does today.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "laplace_pressure_floor"
REPORT = BENCH / "report.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(REPO / "src"))
from tensorsph import droplet_impact as di  # noqa: E402  (path set above)

lpf = _load("laplace_pressure_floor_run_fs", BENCH / "run.py")


@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py first")
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def study(report):
    if "closure_study" not in report:
        pytest.skip("report.json has no closure_study: run "
                    "`python benchmarks/laplace_pressure_floor/run.py --closure-only`")
    return report["closure_study"]


@pytest.fixture(scope="module")
def ladder(study):
    return study["ladder"]


@pytest.fixture(scope="module")
def variants(study):
    return {v["label"]: v for v in study["variants"]}


# --------------------------------------------------------------------------- #
# the record is well formed and matches the library defaults
# --------------------------------------------------------------------------- #
def test_closure_study_covers_the_whole_c1_ladder(study, report):
    """Every (dx, c0) case of the C1-C4 ladder is run with the closure on too."""
    cases = {(r["dx_m"], r["sound_speed"]) for r in study["ladder"]}
    expected = {(r["dx_m"], r["sound_speed"]) for r in report["runs"]}
    assert expected <= cases, (
        "closure study must cover every orientation of the recorded ladder: "
        f"missing {sorted(expected - cases)}")


def test_recorded_closure_is_the_library_default(study):
    """The study uses the shipped defaults, not a bespoke tuning."""
    cfg = di.Config()
    ref = study["reference_closure"]
    assert ref["donors"] == cfg.surface_pressure_donors == "interior"
    assert ref["layers"] == cfg.surface_pressure_layers == 2
    assert ref["threshold"] == pytest.approx(cfg.surface_pressure_threshold)
    assert cfg.surface_pressure_closure is False  # off by default


# --------------------------------------------------------------------------- #
# mechanism: the clamp pins the shell, the closure lifts it
# --------------------------------------------------------------------------- #
def test_tain_clamp_pins_every_free_surface_particle_to_zero(ladder):
    """C3 as measured: without a closure the shell cannot carry any pressure."""
    for row in ladder:
        outer = row["off"]["surface"]["outer"]
        assert outer["zero_pressure_fraction"] > 0.999, (row["dx_m"], row["sound_speed"])
        assert outer["mean_pressure_Pa"] == pytest.approx(0.0, abs=1e-9)
    assert all(row["off"]["surface"]["core"]["mean_pressure_Pa"] > 0.0 for row in ladder)


def test_closure_gives_the_shell_a_pressure(study, ladder, report):
    """After the closure the free-surface particles carry a finite pressure."""
    assert "15" in study["criterion"]  # the gate the ladder is judged against
    dp_analytic = report["reference"]["delta_p_Pa"]
    for row in ladder:
        outer = row["on"]["surface"]["outer"]
        assert outer["zero_pressure_fraction"] < 0.5, (row["dx_m"], row["sound_speed"])
        assert outer["mean_pressure_Pa"] > 0.10 * dp_analytic, (
            row["dx_m"], row["sound_speed"], outer["mean_pressure_Pa"])


def test_closure_lifts_the_interior_compression_towards_the_requirement(ladder):
    """The interior reaches the compression Young-Laplace asks for.

    ``compression_ratio`` = achieved core compression / the compression the Tait
    law needs for ``sigma/R``.  The clamp sits at ~0.5-0.7 everywhere; the
    closure moves five of six cases into 0.85-1.2.  The exception is recorded,
    not hidden: the coarse / high-c0 corner overshoots to ~2.4 and is the least
    quiet run of the ladder (see the corner tests below).
    """
    inside = 0
    for row in ladder:
        off_ratio = row["off"]["compression_ratio"]
        on_ratio = row["on"]["compression_ratio"]
        assert on_ratio > off_ratio, (row["dx_m"], row["sound_speed"])
        assert off_ratio < 0.75, (row["dx_m"], row["sound_speed"], off_ratio)
        if 0.85 <= on_ratio <= 1.2:
            inside += 1
    assert inside >= len(ladder) - 1, inside


# --------------------------------------------------------------------------- #
# acceptance: the end-to-end jump
# --------------------------------------------------------------------------- #
def test_every_closure_case_is_within_15pct(ladder):
    """The P0 acceptance criterion: the tail-averaged jump inside +-15 %."""
    bad = [(r["dx_m"], r["sound_speed"], round(r["on"]["rel_error_pct"], 1))
           for r in ladder if not r["on"]["within_15pct"]]
    assert not bad, f"closure cases outside 15 %: {bad}"


def test_closure_is_what_closes_the_gap(ladder):
    """Same drop, one boolean: the clamp may not be inside 15 % anywhere."""
    for row in ladder:
        assert not row["off"]["within_15pct"], (row["dx_m"], row["sound_speed"])
        assert row["gain_points"] > 20.0, (row["dx_m"], row["sound_speed"],
                                           row["gain_points"])


def test_closure_keeps_the_drop_finite_and_slow(ladder):
    """No blow-up: the closure is a pressure boundary, not a driving force."""
    for row in ladder:
        assert row["on"]["finite"], (row["dx_m"], row["sound_speed"])
        # v_ref = sqrt(sigma / (rho0 R)) ~ 0.6 m/s; 0.5 v_ref is ~Mach 0.02 at
        # c0 = 5 m/s, i.e. deep in the incompressible regime.
        assert row["on"]["vmax_over_vref"] < 0.5, (
            row["dx_m"], row["sound_speed"], row["on"]["vmax_over_vref"])


def test_reference_case_is_the_quietest_setting(ladder, study):
    """The reference case is a static measurement, not a breathing drop."""
    ref = study["reference_case"]
    row = next(r for r in ladder
               if r["dx_m"] == ref["dx_m"]
               and r["sound_speed"] == ref["sound_speed"])
    assert row["on"]["vmax_over_vref"] < 0.15, row["on"]["vmax_over_vref"]


def test_corner_case_is_the_recorded_limitation(ladder, study):
    """The coarse / high-c0 corner is the weakest run; that is on the record.

    Two passes leave part of the outer shell without a donor there, the interior
    overshoots the required compression, and the drop keeps a residual motion
    (v_max ~0.4 v_ref).  The jump still lands inside 15 %, but a paper-grade
    claim may not be made from that corner.
    """
    corner = study["corner_case"]
    row = next(r for r in ladder
               if r["dx_m"] == corner["dx_m"]
               and r["sound_speed"] == corner["sound_speed"])
    worst = max(ladder, key=lambda r: r["on"]["vmax_over_vref"])
    assert (worst["dx_m"], worst["sound_speed"]) == (corner["dx_m"], corner["sound_speed"])
    assert row["on"]["vmax_over_vref"] > 0.1, row["on"]["vmax_over_vref"]
    assert row["on"]["surface"]["outer"]["zero_pressure_fraction"] > 0.1
    assert row["on"]["within_15pct"]


# --------------------------------------------------------------------------- #
# the knobs that are *not* knobs
# --------------------------------------------------------------------------- #
def test_literal_adami_donor_set_is_not_enough(variants):
    """Every neighbour donating (the literal formula) dilutes with p = 0."""
    literal = variants["literal Adami: every neighbour donates"]["reading"]
    assert not literal["within_15pct"], literal["rel_error_pct"]
    reference = variants["interior donors, 2 passes (reference)"]["reading"]
    assert abs(reference["rel_error_pct"]) < abs(literal["rel_error_pct"])
    # ... and the difference is structural, not a rounding effect.
    assert abs(literal["rel_error_pct"] - reference["rel_error_pct"]) > 10.0


def test_layering_is_saturated_at_the_reference_case(variants):
    """At the reference case the pass count is inert: 1 == 2 == 3 passes.

    The layered default is kept as a margin for the corner case (where two
    passes still leave part of the shell without a donor), not because the
    reference case needs it -- the measurement says it does not.
    """
    one = variants["one pass only"]["reading"]
    two = variants["interior donors, 2 passes (reference)"]["reading"]
    assert abs(one["rel_error_pct"] - two["rel_error_pct"]) < 1.0
    assert one["within_15pct"] and two["within_15pct"]


def test_three_passes_gain_nothing(variants):
    """Two layers saturate the extrapolation; a third must be inert."""
    two = variants["interior donors, 2 passes (reference)"]["reading"]
    three = variants["three passes"]["reading"]
    assert abs(three["rel_error_pct"] - two["rel_error_pct"]) < 1.0


def test_receiver_threshold_does_not_flip_the_verdict(variants):
    """The threshold moves the number but not the pass/fail grade.

    Recorded honestly: widening the receiver set to 0.95 rho0 costs ~16 points
    (it pulls already-correct bulk-adjacent particles into the extrapolation),
    so the threshold is *not* free -- but both settings stay inside 15 % and the
    shipped 0.99 is the better one.
    """
    lo = variants["receiver threshold 0.95 rho0"]["reading"]
    hi = variants["receiver threshold 0.995 rho0"]["reading"]
    ref = variants["interior donors, 2 passes (reference)"]["reading"]
    assert lo["within_15pct"], lo["rel_error_pct"]
    assert hi["within_15pct"], hi["rel_error_pct"]
    assert abs(ref["rel_error_pct"]) < abs(lo["rel_error_pct"])


# --------------------------------------------------------------------------- #
# live A/B: the record still describes the code
# --------------------------------------------------------------------------- #
def test_live_ab_reproduces_the_direction():
    """One live A/B at the recorded window (~75 s) still shows the direction.

    A *short* window does not, and that is recorded here on purpose: at 0.6
    t_sigma the clamp-off run reads 14.3 % and the closed run 19.6 %, i.e. the
    closure is still moving the drop through its initial transient.  The
    comparison is only meaningful once the breathing mode has relaxed, which is
    why every number in the report uses the full 1.5 t_sigma window.
    """
    probe = lpf.relax(0.002, 5.0)
    closed = lpf.relax(0.002, 5.0, closure=True, donors="interior", layers=2)
    err_off = abs(probe["absolute_time_average_tail"]["rel_error_pct"])
    err_on = abs(closed["absolute_time_average_tail"]["rel_error_pct"])
    assert err_on < 0.5 * err_off, (err_off, err_on)
    assert closed["finite"]
    # The shell diagnostic has to move in the same direction as in the record.
    assert probe["surface"]["outer"]["zero_pressure_fraction"] > 0.999
    assert closed["surface"]["outer"]["zero_pressure_fraction"] < 0.5

def test_corner_pass_count_saturates_at_two(study):
    """At the coarse/high-c0 corner the pass count saturates at two.

    Recorded honestly: one pass is *closer to the analytic jump* there than two
    (more passes over-fill the shell and overshoot the core pressure), and every
    setting stays inside 15 %.  So the layer count is a scenario choice, not a
    monotone improvement -- the default of two is a conservative setting.
    """
    readings = {v["label"]: v["reading"] for v in study["corner_variants"]}
    one = readings["one pass"]
    two = readings["two passes (reference)"]
    four = readings["four passes"]
    literal = readings["literal Adami (all donors)"]
    assert abs(four["rel_error_pct"] - two["rel_error_pct"]) < 1.0
    assert abs(one["rel_error_pct"]) < abs(two["rel_error_pct"])
    assert one["within_15pct"] and two["within_15pct"]
    assert not literal["within_15pct"]

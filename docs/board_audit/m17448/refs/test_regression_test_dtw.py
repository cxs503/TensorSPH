"""Tests for the DTW regression-baseline protocol (reference practice port).

The reference is ``Xiangyu-Hu/SPHinXsys`` ``src/shared/regression_test/``.  What
is asserted here:

* the DP kernel agrees with an independently coded shortest-path solver;
* analytic alignments reproduce their closed-form distance;
* the two documented quirks of the reference recurrence are reproduced on
  purpose (so that distances stay comparable with the reference) and flagged;
* the protocol's bookkeeping (``runtimes.dat``), convergence rule (four
  successive revisions within the threshold), acceptance band (``1.01 x``) and
  XML round trip behave as the sources describe;
* a real case generation only needs to be cheap, so the end-to-end test runs the
  dam-break case at a coarse resolution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorsph.regression_test import (
    DTWLengthError,
    DTWRegressionFailure,
    DTWRegressionTest,
    ProtocolConfig,
    band_edge_shortcuts,
    dtw_distance,
    dtw_distance_shortest_path,
    dtw_distance_strict,
    run_generation,
)


# ---------------------------------------------------------------------------
# distance kernel
# ---------------------------------------------------------------------------
def test_identical_series_have_zero_distance():
    series = np.sin(np.linspace(0, 3, 25))
    assert dtw_distance(series, series) == 0.0
    assert dtw_distance_strict(series, series) == 0.0


def test_constant_offset_reproduces_the_closed_form_path_cost():
    """Two constant series differ by ``c``; every alignment costs ``c`` per pair."""
    series_a = np.full(24, 1.0)
    series_b = np.full(24, 1.5)
    assert dtw_distance_strict(series_a, series_b, window=None) == pytest.approx(
        0.5 * series_a.size, rel=1e-12)


def test_vector_series_use_the_euclidean_norm():
    step = np.array([0.3, -0.4])  # norm 0.5
    series_a = np.tile([1.0, 2.0], (12, 1))
    series_b = series_a + step
    assert dtw_distance_strict(series_a, series_b, window=None) == pytest.approx(
        0.5 * 12, rel=1e-12)


def test_dp_agrees_with_the_independent_shortest_path_solver():
    rng = np.random.default_rng(7)
    for _ in range(5):
        series_a = rng.standard_normal(24)
        series_b = rng.standard_normal(24)
        for window in (5, 10):
            dp = dtw_distance_strict(series_a, series_b, window=window)
            graph = dtw_distance_shortest_path(series_a, series_b, window=window)
            assert graph == pytest.approx(dp, rel=1e-12, abs=1e-12)


def test_length_guard_rejects_a_large_snapshot_count_change():
    series = np.arange(20.0)
    with pytest.raises(DTWLengthError):
        dtw_distance(series, np.arange(30.0))
    # 10% is inside the reference +-20% guard and compares
    assert math.isfinite(dtw_distance(series, np.arange(22.0)))


def test_faithful_port_reproduces_the_reference_zero_initialised_band():
    """Pinned quirks: zero-initialised band truncates the alignment path.

    Both effects are reproduced on purpose so that TensorSPH distances are
    comparable with the reference implementation; :func:`dtw_distance_strict`
    is the correct constrained optimum and the protocol can be switched to it.
    """
    ramp = np.arange(40.0)

    # (a) length mismatch wider than the window: the terminal cell of the DP
    # table is never written, so the reference returns 0 for differing series.
    assert dtw_distance(ramp, np.concatenate([ramp, ramp[-8:]])) == 0.0

    # (b) an alignment that leaves the band: the reference picks a zero-valued
    # out-of-band neighbour, so the distance is a strict underestimate.
    shifted = np.roll(np.sin(np.linspace(0, 6, 40)), 8)
    reference = np.sin(np.linspace(0, 6, 40))
    diagnostic = band_edge_shortcuts(reference, shifted, window=5)
    assert diagnostic["distance"] < diagnostic["strict"]
    assert diagnostic["shortcut_cells"] > 0
    assert diagnostic["relative_underestimate"] > 0.05


def test_diagonal_alignment_is_unaffected_by_band_edge_shortcuts():
    """Band-edge cells do take zeros, but a pure diagonal alignment does not
    route through them, so the distance equals the constrained optimum."""
    ramp = np.linspace(0.0, 1.0, 30)
    diagnostic = band_edge_shortcuts(ramp, ramp + 0.01, window=5)
    assert diagnostic["distance"] == pytest.approx(diagnostic["strict"], rel=1e-9)


# ---------------------------------------------------------------------------
# protocol bookkeeping
# ---------------------------------------------------------------------------
def _protocol(tmp_path, **kwargs) -> DTWRegressionTest:
    config = ProtocolConfig(
        workdir=tmp_path, dynamics_name="Dyn", quantity_names=("front", "top"),
        window=5, **kwargs)
    return DTWRegressionTest(config)


def _generate(tmp_path, series, threshold=0.01, **kwargs):
    return run_generation(tmp_path, series, threshold=threshold, generate=True,
                          dynamics_name="Dyn", quantity_names=("front", "top"),
                          window=5, **kwargs)


def _series(scale: float, n: int = 20, noise: float = 0.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.1, 2.0, n)
    base = np.column_stack([t ** 0.5, np.sin(t)])
    return (base * scale + noise * rng.standard_normal(base.shape)).tolist()


def test_runtimes_file_holds_the_three_reference_fields(tmp_path):
    report = _generate(tmp_path, _series(1.0))
    lines = (tmp_path / "Dyn_front_top_runtimes.dat").read_text().split()
    assert len(lines) == 3
    converged, number_of_run, label = lines
    assert converged in {"true", "false"}
    assert int(number_of_run) == 2          # first generation stored, one run consumed
    assert int(label) == 0
    assert report["state_after"]["number_of_run"] == 2


def test_generation_files_and_accepted_distances_round_trip(tmp_path):
    _generate(tmp_path, _series(1.0))
    _generate(tmp_path, _series(1.0, noise=1e-3))
    assert (tmp_path / "Dyn_front_top_Run_0_result.xml").exists()
    assert (tmp_path / "Dyn_front_top_Run_1_result.xml").exists()

    test = _protocol(tmp_path)
    stored = test.read_result_from_xml(0)
    assert stored.shape == (2, 20)
    accepted = test.read_dtw_distance_from_xml()
    assert accepted.shape == (2,)
    assert np.all(accepted >= 0.0)


def test_accepted_distance_is_a_running_maximum(tmp_path):
    _generate(tmp_path, _series(1.0))
    first = _generate(tmp_path, _series(1.0, noise=1e-4))["accepted_new"]
    second = _generate(tmp_path, _series(1.0, noise=1e-3))["accepted_new"]
    assert all(a <= b + 1e-15 for a, b in zip(first, second)), (
        "the accepted baseline must never be loosened between generations")


def test_convergence_needs_four_successive_quiet_generations(tmp_path):
    _generate(tmp_path, _series(1.0))
    states = []
    for _ in range(6):
        report = _generate(tmp_path, _series(1.0, noise=1e-6))
        states.append((report["converged"], report.get("label_for_repeat")))
    assert states[0][0] == "false"
    assert any(state[0] == "true" for state in states), (
        f"the database never converged: {states}")
    converged_index = next(i for i, state in enumerate(states) if state[0] == "true")
    assert converged_index >= 4, (
        "converged before four successive revisions were observed")
    # once converged, further generations are skipped (reference behaviour)
    skipped = _generate(tmp_path, _series(1.0, noise=1e-6))
    assert skipped["skipped"] is True


def test_a_drifting_generation_resets_the_convergence_counter(tmp_path):
    _generate(tmp_path, _series(1.0))
    _generate(tmp_path, _series(1.0, noise=1e-6))
    report = _generate(tmp_path, _series(1.02))          # far outside 1% revision
    assert report["label_for_repeat"] == 0


def test_test_result_accepts_a_similar_generation(tmp_path):
    _generate(tmp_path, _series(1.0))
    _generate(tmp_path, _series(1.0, noise=1e-5))
    report = run_generation(tmp_path, _series(1.0, noise=5e-6), threshold=0.01,
                            generate=False, dynamics_name="Dyn",
                            quantity_names=("front", "top"), window=5)
    assert report["passed"] is True
    assert report["failures"] == []


def test_test_result_rejects_a_generation_outside_the_acceptance_band(tmp_path):
    _generate(tmp_path, _series(1.0))
    _generate(tmp_path, _series(1.0, noise=1e-5))
    with pytest.raises(DTWRegressionFailure) as error:
        run_generation(tmp_path, _series(1.3), threshold=0.01, generate=False,
                       dynamics_name="Dyn", quantity_names=("front", "top"),
                       window=5)
    payload = error.value.args[0]
    assert "current" in payload and "accepted" in payload


def test_zero_distance_on_differing_series_is_flagged(tmp_path):
    """The reference quirk must not pass silently through the protocol."""
    # 40 -> 48 snapshots is inside the +-20% length guard; the 8 extra
    # snapshots exceed the band width, so the reference DP never writes the
    # terminal cell and both quantities come back with a distance of exactly 0.
    base = np.arange(40.0)
    stretched = np.concatenate([base, base[-8:]])
    series_a = np.column_stack([base, 2.0 * base])
    series_b = np.column_stack([stretched, 2.0 * stretched])
    _generate(tmp_path, series_a.tolist())
    report = _generate(tmp_path, series_b.tolist())
    flagged = report["degenerate_zero_distances"]
    assert flagged, "a zero distance on differing series must be reported"
    assert {entry["quantity"] for entry in flagged} == {"front", "top"}


def test_length_mismatch_inside_the_guard_trims_the_stored_series(tmp_path):
    _generate(tmp_path, _series(1.0, n=20))
    report = _generate(tmp_path, _series(1.0, n=18, noise=1e-6))
    assert report["snapshots"] == 18


# ---------------------------------------------------------------------------
# end to end on a case (coarse, so it stays cheap)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_dam_break_generations_enter_the_database_and_gate_a_regression(tmp_path):
    from tensorsph.dam_break import DamBreakConfig
    from tensorsph.regression_test.observe import QUANTITIES, dam_break_observations

    def observe(dt_scale: float):
        cfg = DamBreakConfig(dx=0.06)
        cfg.dt = 1.0e-4 * dt_scale
        cfg.steps = int(round(0.1 / cfg.dt))
        cfg.output_every = 10
        times = np.linspace(0.01, 0.098, 10)
        return dam_break_observations(cfg, times)

    workdir = tmp_path / "db"
    for dt_scale in (1.0, 1.005):
        observed = observe(dt_scale)
        assert observed.values.shape == (10, len(QUANTITIES))
        run_generation(workdir, observed.as_rows(), threshold=0.01,
                       generate=True, dynamics_name="DamBreak",
                       quantity_names=QUANTITIES, window=5)

    accepted = DTWRegressionTest(ProtocolConfig(
        workdir=workdir, dynamics_name="DamBreak", quantity_names=QUANTITIES,
        window=5)).read_dtw_distance_from_xml()
    assert np.all(accepted >= 0.0) and np.any(accepted > 0.0), (
        "two generations with different time steps must differ measurably")

    legit = run_generation(workdir, observe(1.0).as_rows(), threshold=0.01,
                           generate=False, dynamics_name="DamBreak",
                           quantity_names=QUANTITIES, window=5)
    assert legit["passed"]

    # negative control: a physically different run must trip the gate
    cfg = DamBreakConfig(dx=0.06, viscosity=50.0 * DamBreakConfig(dx=0.06).viscosity)
    cfg.steps = int(round(0.1 / cfg.dt))
    cfg.output_every = 10
    different = dam_break_observations(cfg, np.linspace(0.01, 0.098, 10))
    with pytest.raises(DTWRegressionFailure):
        run_generation(workdir, different.as_rows(), threshold=0.01, generate=False,
                       dynamics_name="DamBreak", quantity_names=QUANTITIES, window=5)
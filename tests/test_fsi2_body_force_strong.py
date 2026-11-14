"""The body force-from-fluid arms on the strong-coupling FSI2 path.

Regression guard for the defect found while retrieving the cluster transient
``cj3112``: the shipped ``run_body_force_transient.py`` records
``cyl_drag``/``body_drag_fit``/``drag_viscous`` from the row returned by
``step_fsi2_strong``, but that row simply did not contain those keys, so the
driver's ``row.get(k, nan)`` wrote **NaN** for every step of the only path that
survives ``rho_s/rho_f = 1``.  The force-from-fluid load was therefore absent --
silently -- exactly where the Turek-Hron coefficients have to be formed.

The fix routes both drivers through ``fsi2_turek_hron.body_force_from_fluid``;
the tests below pin (a) the keys exist and are finite, (b) the two drivers
produce the *same* load for the same fluid snapshot (one coupling iterate ==
the loose step), which is what makes the shared operator claim meaningful, and
(c) the coverage counter, so a silent zero load cannot pass as a small one.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.fsi2_similarity import similarity_corrected_case  # noqa: E402
from tensorsph.fsi2_strong_coupling import step_fsi2_strong  # noqa: E402
from tensorsph.fsi2_turek_hron import initial_state, step_fsi2  # noqa: E402

KEYS = ("cyl_drag", "cyl_lift", "body_drag", "body_lift", "cyl_drag_fit",
        "cyl_lift_fit", "body_drag_fit", "body_lift_fit", "drag_viscous",
        "lift_viscous", "cyl_coverage")


@pytest.fixture(scope="module")
def case():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    c = similarity_corrected_case(U_mean=0.2)
    c.dx = 0.05
    c.n_beam_nodes = max(15, int(c.beam_length / c.dx))
    c.max_rod_substeps = 2000
    c.rod_stability_target = 0.2
    return c


def test_strong_row_carries_finite_body_force(case):
    state = initial_state(case)
    dt = case.dt()
    row, _ = step_fsi2_strong(state, case, dt, max_iter=2)
    for key in KEYS:
        assert key in row, f"strong-coupling row is missing {key}"
        assert math.isfinite(row[key]), f"{key} is not finite: {row[key]}"
    # the cylinder is fully immersed: every surface row must see fluid
    assert row["cyl_coverage"] == pytest.approx(1.0)
    assert row["n_cyl_particles_on_body"] > 0
    # the load is not a placeholder zero
    assert abs(row["body_drag_fit"]) > 0.0


def test_strong_single_iterate_matches_loose_driver(case):
    """One coupling iterate solves the fluid at the step-start geometry, which
    is exactly what the loose driver does -- so the loads must agree."""
    dt = case.dt()
    loose = step_fsi2(initial_state(case), case, dt)
    strong, _ = step_fsi2_strong(initial_state(case), case, dt, max_iter=1)
    for key in ("cyl_drag", "cyl_lift", "cyl_drag_fit", "cyl_lift_fit",
                "body_drag", "body_lift", "drag", "lift", "drag_viscous",
                "lift_viscous", "cyl_coverage"):
        scale = max(1.0, abs(loose[key]))
        assert abs(strong[key] - loose[key]) <= 1.0e-12 * scale, (
            f"{key}: strong={strong[key]!r} loose={loose[key]!r}")


def test_body_force_stays_finite_over_a_few_steps(case):
    state = initial_state(case)
    dt = case.dt()
    for _ in range(3):
        row, _ = step_fsi2_strong(state, case, dt, max_iter=2)
        for key in KEYS:
            assert math.isfinite(row[key]), f"{key} went non-finite"
        assert state.fluid_vel.norm(dim=1).max() < 50.0
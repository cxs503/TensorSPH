"""The leading-edge metric must survive a boundary-clamped particle (m21404).

The legacy ``front_x_m`` column is a bare ``max`` over the fluid x.  A particle
that crossed a wall is clamped *onto* the wall by ``apply_boundary`` and stays
there, so a single clamp event pins the front at the far wall for the rest of
the record.  That happened on the committed long-horizon dx=0.01 run
(``results/peak_horizon/dx0.01_time_series.csv``: 1.215 m at t=0 -> 3.21 m at
t=0.3 s while the surge is still near 1.9 m; ``boundary_escapes = 5``), i.e. the
front-kinematics split of the arrival attribution was reading a saturated
column.  ``front_position(..., exclude=escape_mask)`` removes exactly the
clamped particles and nothing else.
"""
import torch

from tensorsph.kleefsman_obstacle import (
    KleefsmanConfig,
    boundary_crossing_mask,
    front_position,
)


def _cfg(dx=0.01):
    return KleefsmanConfig(dx=dx, device="cpu")


def test_no_mask_is_the_legacy_maximum():
    pos = torch.tensor([[0.5, 0.1], [1.25, 0.05], [2.0, 0.2]])
    assert front_position(pos) == 2.0


def test_single_clamped_particle_saturates_the_legacy_metric():
    body = torch.tensor([[1.2, 0.05], [1.5, 0.1], [1.9, 0.02]])
    clamped = torch.tensor([[3.22, 0.3]])            # pinned on the far wall
    pos = torch.cat((body, clamped), dim=0)
    exclude = torch.tensor([False, False, False, True])
    assert abs(front_position(pos) - 3.22) < 1e-5     # legacy reading: saturated
    assert abs(front_position(pos, exclude=exclude) - 1.9) < 1e-6
    # a non-clamped particle ahead of the body is *not* removed: spray is
    # genuine SPH kinematics, only the clamp artefact is excluded
    spray = torch.tensor([[2.3, 0.6]])
    pos2 = torch.cat((body, spray), dim=0)
    exclude2 = torch.tensor([False, False, False, False])
    assert abs(front_position(pos2, exclude=exclude2) - 2.3) < 1e-5


def test_mask_matches_the_clamp_predicate():
    cfg = _cfg()
    tol = 0.25 * cfg.dx
    pos = torch.tensor([
        [0.5, 0.5],                      # inside
        [3.22 + 0.1, 0.5],               # beyond the far wall
        [0.5, -0.1],                     # below the floor
        [0.5, cfg.tank_height + 1.0],    # above the tank roof
        [3.22 + 0.1 * tol, 0.5],         # excursion inside the tolerance: kept
    ])
    mask = boundary_crossing_mask(pos, cfg, top=cfg.tank_height)
    assert mask.tolist() == [False, True, True, True, False]


def test_resume_round_trip_keeps_the_mask():
    """The mask is part of the restart state, so a chunked run stays exact."""
    from tensorsph.kleefsman_obstacle import run_case

    cfg = _cfg(dx=0.04)
    cfg.t_end = 0.02
    cfg.output_every = 5
    first = run_case(cfg, time_budget=1e-6)          # one chunk, then stop
    assert first.finished is False
    state = first.state
    assert "escape_mask" in state and state["escape_mask"].shape[0] == state["positions"].shape[0]
    second = run_case(cfg, resume=state)
    assert second.finished is True
    assert len(second.front_robust) == len(second.times) == len(second.excluded)

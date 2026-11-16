"""Independent-realization knob of the wedge-entry case (m22826).

Two groups of checks:

* ``wedge_entry`` side -- the discrete-phase offsets are *only* a discretisation
  phase: with ``scale = 0`` the initial state is bit-identical to the committed
  case, with ``scale > 0`` it is an exact rigid translation of the fluid lattice
  (so every particle-particle distance, and with it the calibrated initial
  density, is untouched), and the wall/body lattice is not moved at all.
* ``seed_ensemble`` side -- the predeclared decision rule actually consumes the
  realizations: it must be able to return PASS *and* to return NOT_DECIDABLE on
  synthetic artefacts, otherwise a green run would prove nothing.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
CASE = HERE / "benchmarks" / "case_sph_wedge_entry_2d"
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))
if str(CASE) not in sys.path:
    sys.path.insert(0, str(CASE))

from tensorsph.wedge_entry import (  # noqa: E402
    GAP_JITTER_MAX_DX,
    PHASE_X_MAX_DX,
    WedgeConfig,
    WedgeImpact,
    realisation_phases,
    with_realisation,
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, CASE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seed_mod = _load("seed_ensemble")


def _cfg(**kw) -> WedgeConfig:
    base = dict(beta_deg=30.0, velocity=1.0, dx=4.0e-3, frames=5)
    base.update(kw)
    return WedgeConfig(**base)


# --------------------------------------------------------------------------- #
# the knob is a phase, not a physics change
# --------------------------------------------------------------------------- #

def test_phases_are_seeded_and_bounded():
    a = realisation_phases(7)
    assert a == realisation_phases(7)              # reproducible
    assert a != realisation_phases(8)              # seed matters
    for phase, gap in [realisation_phases(s) for s in range(20)]:
        assert abs(phase) <= PHASE_X_MAX_DX
        assert abs(gap) <= GAP_JITTER_MAX_DX
    assert realisation_phases(3, scale=0.0) == (0.0, 0.0)
    half = realisation_phases(5, scale=0.5)
    full = realisation_phases(5, scale=1.0)
    assert half[0] == pytest.approx(0.5 * full[0])
    assert half[1] == pytest.approx(0.5 * full[1])
    with pytest.raises(ValueError):
        realisation_phases(1, scale=1.5)


def test_scale_zero_is_bit_identical_to_the_committed_case():
    base = _cfg(phase_x_dx=0.0, gap_jitter_dx=0.0)
    zero = with_realisation(base, seed=4, scale=0.0)
    assert zero.phase_x_dx == 0.0 and zero.gap_jitter_dx == 0.0
    a = WedgeImpact(base)
    b = WedgeImpact(zero)
    assert a.n_fluid == b.n_fluid
    import torch
    assert torch.equal(a.fluid_pos, b.fluid_pos)
    assert torch.equal(a.wall_pos, b.wall_pos)
    assert a.gap == b.gap


def test_lateral_phase_is_a_rigid_translation_of_the_fluid_lattice():
    import torch
    base = _cfg()
    shifted = _cfg(phase_x_dx=0.21)
    a, b = WedgeImpact(base), WedgeImpact(shifted)
    assert a.n_fluid == b.n_fluid
    dx = torch.diff(b.fluid_pos[:, 0]).abs()
    assert torch.equal(b.fluid_pos[:, 1], a.fluid_pos[:, 1])   # y untouched
    delta = b.fluid_pos[:, 0] - a.fluid_pos[:, 0]
    assert float(delta.max() - delta.min()) == pytest.approx(0.0, abs=1e-15)  # one shift
    assert float(delta[0]) == pytest.approx(0.21 * 4.0e-3, rel=0, abs=1e-15)
    # ... and the lattice spacing is therefore unchanged, which is what keeps the
    # calibrated initial density (mass_unit) exact.
    assert dx.numel() == a.n_fluid - 1


def test_lateral_phase_leaves_the_body_and_tank_alone():
    import torch
    a = WedgeImpact(_cfg())
    b = WedgeImpact(_cfg(phase_x_dx=-0.25))
    assert torch.equal(a.wedge_local, b.wedge_local)
    assert torch.equal(a.tank_local, b.tank_local)
    assert torch.equal(a.wall_pos, b.wall_pos)


def test_gap_jitter_shifts_contact_time_only():
    base = _cfg()
    jit = _cfg(gap_jitter_dx=0.2)
    a, b = WedgeImpact(base), WedgeImpact(jit)
    dV = 0.2 * base.dx / base.velocity
    assert b.t_contact == pytest.approx(a.t_contact + dV, rel=1e-12)
    assert b.time == pytest.approx(-b.t_contact, rel=1e-12)
    # the observation window is set by c_end and is unchanged
    assert b.cfg.resolved_t_end() == pytest.approx(a.cfg.resolved_t_end(), rel=0)


@pytest.mark.parametrize("kw", [
    {"phase_x_dx": PHASE_X_MAX_DX + 1e-6},
    {"gap_jitter_dx": -(GAP_JITTER_MAX_DX + 1e-6)},
    {"initial_gap_dx": 1.0, "gap_jitter_dx": -0.2},
])
def test_out_of_bounds_offsets_are_rejected(kw):
    with pytest.raises(ValueError):
        _cfg(**kw).validate()


def test_realization_changes_the_trajectory():
    """A seed must actually decorrelate the trajectory, not just the array."""
    a = WedgeImpact(_cfg())
    b = WedgeImpact(with_realisation(_cfg(), seed=1))
    for sim in (a, b):
        while sim.time < 0.30 * sim.cfg.resolved_t_end():
            sim.advance(sim.dt())
    da, db = a.diagnostics(), b.diagnostics()
    assert da["c"] > 0.0 and db["c"] > 0.0
    # the seed must move the *dynamics*, not only the initial array: the impact
    # phase differs by a fraction of a cell, so at a fixed observation time the
    # wetted width and the centre pressure cannot coincide
    assert (da["c"], da["p_centre"]) != (db["c"], db["p_centre"])


# --------------------------------------------------------------------------- #
# the rule consumes the realizations (and can say no)
# --------------------------------------------------------------------------- #

def _fake_artefact(out: Path, dx: float, seed: int, obs: float,
                   spread: float, scale: float = 1.0) -> None:
    payload = {
        "config": {"dx": dx},
        "realisation": {"seed": seed, "scale": scale,
                        "phase_x_dx": 0.1, "gap_jitter_dx": 0.1},
        "by_frames": {"61": {"resolution_band": {"peak_median": obs,
                                                 "spread_rel": spread},
                             "jackknife_rel": spread}},
        "series": {"cp_centre": [1.0, 2.0, 1.5, 2.5, 2.0],
                   "max_rho_ratio": [1.0, 1.01, 1.02, 1.01, 1.0]},
    }
    tag = f"real_dx{dx*1e3:g}mm_c0v20_seed{seed}"
    if scale != 1.0:
        tag += f"_scale{scale:g}"
    (out / f"{tag}.json").write_text(json.dumps(payload))


BIAS = HERE / "docs" / "artifacts" / "case_sph_wedge_entry_2d_bias" / "bias_report.json"


def _truth(bias: dict, tag: str) -> float:
    return bias["grids"][tag]["by_frames"]["61"]["synth_smoothed"]["peak_median"]


def test_rule_passes_when_realizations_agree(tmp_path):
    bias = json.loads(BIAS.read_text())
    for tag, dx in (("dx2mm", 2.0e-3), ("dx1mm", 1.0e-3)):
        truth = _truth(bias, tag)
        for i, eps in enumerate((1.01, 0.99, 1.005, 0.995)):
            _fake_artefact(tmp_path, dx, i + 1, truth * eps, 0.02)
    agg = seed_mod.aggregate(tmp_path, BIAS)
    assert all(g["n_realisations"] == 4 for g in agg["grids"])
    assert all(g["judgeable"] for g in agg["grids"])
    assert agg["verdict"]["verdict"] == "PASS"


def test_rule_refuses_when_realizations_disagree(tmp_path):
    """Negative control: same machinery must fail on a sign flip / big spread."""
    bias = json.loads(BIAS.read_text())
    t2, t1 = _truth(bias, "dx2mm"), _truth(bias, "dx1mm")
    for i, eps in enumerate((1.30, 0.70, 1.25, 0.75)):
        _fake_artefact(tmp_path, 2.0e-3, i + 1, t2 * eps, 0.30)
        _fake_artefact(tmp_path, 1.0e-3, i + 1, t1 * (2.0 - eps), 0.30)
    agg = seed_mod.aggregate(tmp_path, BIAS)
    assert not any(g["judgeable"] for g in agg["grids"])
    assert agg["verdict"]["verdict"] == "NOT_DECIDABLE"


def test_scale_check_keeps_scale_variants_out_of_the_ensemble(tmp_path):
    bias = json.loads(BIAS.read_text())
    truth = _truth(bias, "dx2mm")
    _fake_artefact(tmp_path, 2.0e-3, 1, truth * 1.00, 0.02)
    _fake_artefact(tmp_path, 2.0e-3, 1, truth * 1.01, 0.02, scale=0.25)
    agg = seed_mod.aggregate(tmp_path, BIAS)
    grid = agg["grids"][0]
    assert grid["n_realisations"] == 1        # the scale variant is not a member
    assert "0.25" in grid["scale_check"] and "1" in grid["scale_check"]
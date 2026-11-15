"""Checkpoint/resume for the 3-D Kleefsman case.

The production rungs of this case are killed intermittently by a glibc heap
abort (``free(): corrupted unsorted chunks``) mid-run.  A crash that discards
the integration state turns a 10-minute cluster job into zero data, so the
solver can now persist its state and continue from it.

These tests assert the *invariants* the feature has to have; finding a file on
disk after running the code is not enough:

* a resumed run is **bit-identical** to the uninterrupted run (same positions,
  velocities, gauge series) -- a resume that merely "looks right" would still
  invalidate every number the case publishes;
* a checkpoint written for a different configuration is **refused**, not
  silently used (that would mix two grids into one series);
* a checkpoint taken before ``t_end`` is resumable with a **longer** horizon,
  which is the whole point (the crash hits before ``t_end``);
* a completed run leaves **no** checkpoint behind, so the retry wrapper cannot
  replay a finished grid;
* the file on disk survives a torn write: the writer goes through
  ``os.replace``, so ``.tmp`` leftovers are never read as state.

An "interrupted run" is simulated by raising from the sample callback, which is
how a mid-run death behaves as far as the solver is concerned: the state file
written up to that moment stays on disk and the run never reaches its cleanup.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from tensorsph.kleefsman_obstacle_3d import (
    CHECKPOINT_VERSION,
    Kleefsman3DConfig,
    load_checkpoint,
    run_case,
    validate_case,
)


def _cfg(t_end: float, **kw) -> Kleefsman3DConfig:
    params = {"dx": 0.05, "t_end": t_end, "output_every": 5, "sound_speed": 20.0,
              "wall_layers": 2}
    params.update(kw)
    cfg = Kleefsman3DConfig(**params)
    validate_case(cfg)
    cfg.dt = 0.9 * cfg.dt_from_courant()
    return cfg


def _ckpt_path() -> Path:
    return Path(tempfile.mkdtemp(prefix="kleefsman3d_ckpt_")) / "state.pt"


def _crash(cfg, ckpt: Path, *, after_samples: int = 3, every: int = 2) -> None:
    """Run until ``after_samples`` samples exist, then die mid-run."""
    seen = {"n": 0}

    def on_sample(_i, _row):
        seen["n"] += 1
        if seen["n"] >= after_samples:
            raise RuntimeError("simulated mid-run death")

    with pytest.raises(RuntimeError, match="simulated mid-run death"):
        run_case(cfg, checkpoint_path=ckpt, checkpoint_every=every,
                 on_sample=on_sample)
    assert ckpt.exists(), "a crash after a checkpoint must leave the state file"


def test_resume_is_bit_identical_to_an_uninterrupted_run():
    straight = run_case(_cfg(0.02))
    assert straight.steps > 10
    assert straight.resumed is False

    ckpt = _ckpt_path()
    _crash(_cfg(0.02), ckpt)
    resumed = run_case(_cfg(0.02), checkpoint_path=ckpt, checkpoint_every=2)

    assert resumed.resumed is True
    assert resumed.steps == straight.steps
    assert resumed.simulated_time == straight.simulated_time
    for name in ("times", "slab", "height", "front"):
        got, want = getattr(resumed, name), getattr(straight, name)
        assert len(got) == len(want), f"{name}: {len(got)} vs {len(want)} samples"
        assert got == want, f"resumed {name} differs from the uninterrupted run"


def test_a_crashed_run_keeps_the_state_it_reached():
    ckpt = _ckpt_path()
    _crash(_cfg(0.02), ckpt)
    state = load_checkpoint(ckpt, _cfg(0.02))
    assert state["step"] >= 1
    assert 0.0 < state["elapsed"] < 0.02, "the crash cannot be past t_end"
    assert state["positions"].ndim == 2 and state["positions"].shape[1] == 3
    assert state["velocity"].shape == state["positions"].shape
    assert state["times"], "the gauge series has to survive the crash"
    assert not (ckpt.parent / (ckpt.name + ".tmp")).exists()


def test_resume_extends_the_horizon():
    ckpt = _ckpt_path()
    _crash(_cfg(0.01), ckpt)
    first = load_checkpoint(ckpt, _cfg(0.02))
    resumed = run_case(_cfg(0.02), checkpoint_path=ckpt, checkpoint_every=2)
    assert resumed.simulated_time > first["elapsed"]
    assert resumed.steps > first["step"]


def test_checkpoint_from_another_configuration_is_refused():
    ckpt = _ckpt_path()
    _crash(_cfg(0.02), ckpt)
    other = _cfg(0.02, wall_layers=3)
    with pytest.raises(ValueError, match="different configuration"):
        load_checkpoint(ckpt, other)
    with pytest.raises(ValueError, match="different configuration"):
        run_case(other, checkpoint_path=ckpt, checkpoint_every=2)


def test_a_completed_run_leaves_no_checkpoint():
    ckpt = _ckpt_path()
    run = run_case(_cfg(0.02), checkpoint_path=ckpt, checkpoint_every=2)
    assert run.resumed is False
    assert not ckpt.exists(), "a finished grid must not be replayed"


def test_checkpoint_carries_its_version():
    ckpt = _ckpt_path()
    _crash(_cfg(0.02), ckpt)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["version"] == CHECKPOINT_VERSION
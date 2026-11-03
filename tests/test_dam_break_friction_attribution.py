"""Guards on the B2 dam-break bed-resistance attribution.

``benchmarks/dam_break/friction_sweep.json`` is produced by a sweep that costs
~10 minutes, so it is archived and pinned here rather than re-simulated by the
suite (same contract as ``tests/test_dam_break_resolution_ladder.py``).

What is pinned:

1. the metric arithmetic: a stored mean absolute error must be reproducible
   from the stored trajectory, so the document cannot drift from the data;
2. the *finding* that no wall-friction blend closes the 0.15 gate -- if a future
   change closes it, this test fails and forces the verification record to be
   rewritten instead of leaving a stale "fails" claim in the docs;
3. that bedding friction actually resists the front (a short live run), so the
   sweep's direction of change is not an artefact of the analysis code;
4. that the timing-shift diagnostic recovers a shift it is given.
"""
from __future__ import annotations

import bisect
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "dam_break" / "friction_sweep.json"
SCRIPT = ROOT / "benchmarks" / "dam_break" / "friction_sweep.py"
DOCUMENT = ROOT / "docs" / "benchmarks" / "dam-break-friction-attribution.md"


def _load_script():
    spec = importlib.util.spec_from_file_location("friction_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _interp(xs, ys, xq):
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_left(xs, xq)
    w = (xq - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + w * (ys[i] - ys[i - 1])


@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT.name} not archived in this checkout")
    return json.loads(ARTIFACT.read_text())


def test_stored_mae_reproduces_from_the_stored_trajectory(artifact):
    exp = artifact["experimental_reference"]["data"]
    for r in artifact["results"]:
        tau, za = r["sim_tau"], r["sim_za"]
        errs = [abs(_interp(tau, za, e["tau"]) - e["Z_over_a"])
                for e in exp if tau[0] <= e["tau"] <= tau[-1]]
        assert errs, "no overlap between the stored run and the reference"
        stored = r["experimental_error"]["mean_absolute_error"]
        assert stored == pytest.approx(sum(errs) / len(errs), abs=1e-9)
        assert r["experimental_error"]["max_absolute_error"] == pytest.approx(max(errs), abs=1e-9)


def test_free_slip_matches_the_resolution_ladder(artifact):
    """The sweep's mu_w = 0 rung must reproduce the ladder's coarse rung."""
    ladder = json.loads((ROOT / "benchmarks" / "dam_break"
                         / "resolution_ladder.json").read_text())
    ladder_mae = {r["dx"]: r["experimental_error"]["mean_absolute_error"]
                  for r in ladder["results"]}
    for r in artifact["results"]:
        if r["wall_friction"] != 0.0 or r["dx"] not in ladder_mae:
            continue
        assert r["experimental_error"]["mean_absolute_error"] == pytest.approx(
            ladder_mae[r["dx"]], rel=0.02)


def test_no_wall_setting_closes_the_experimental_gate(artifact):
    """The honest verdict: friction does not rescue the 0.15 limit."""
    best = min(r["experimental_error"]["mean_absolute_error"]
               for r in artifact["results"])
    assert best > 0.15, (
        "a wall-friction setting now passes the experimental gate; the "
        "attribution document and docs/benchmarks/dam-break-experimental.md "
        "must be rewritten rather than left claiming failure")


def test_timing_shift_cannot_explain_the_whole_error(artifact):
    """After the fitted shift a one-signed residual must remain."""
    for r in artifact["results"]:
        m = r["metrics"]
        assert m["mae_after_shift"] is not None
        assert m["mae_after_shift"] <= r["experimental_error"]["mean_absolute_error"] + 1e-9
        assert m["mae_after_shift"] > 0.10, (
            f"dx={r['dx']} mu_w={r['wall_friction']}: the residual after a "
            "timing shift is small enough that the timing explanation dominates; "
            "re-examine the attribution")


def test_bed_friction_slows_the_front():
    """Live short run: a no-slip bed must advance the front less far."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from tensorsph.dam_break import DamBreakConfig, run_case

    def end_front(mu: float) -> float:
        cfg = DamBreakConfig(dx=0.05, pool_width=2.0, column_width=0.2,
                             sound_speed=25.0, device="cpu", wall_friction=mu,
                             density_diffusion=0.0, adaptive_dt=True)
        cfg.column_height = 0.4
        cfg.dt = cfg.dt_from_courant()
        cfg.steps = int(round(0.8 / cfg.dt))
        cfg.output_every = max(1, int(round(0.02 / cfg.dt)))
        run = run_case(cfg)
        return (run.front[-1] - cfg.column_width) / cfg.column_width

    free = end_front(0.0)
    no_slip = end_front(1.0)
    assert free > 0.0 and no_slip > 0.0
    assert no_slip < 0.8 * free, (
        f"no-slip bed reached Z/a={no_slip:.3f} against free slip {free:.3f}; "
        "the wall-friction blend is not resisting the front")


def test_shift_diagnostic_recovers_a_known_shift():
    module = _load_script()
    # A synthetic reference and a front that lags it by exactly 0.4 in tau.
    exp = [{"tau": 0.2 * i, "Z_over_a": 0.9 * 0.2 * i, "uncertainty": 0.05}
           for i in range(1, 20)]
    tau = [0.01 * i for i in range(0, 601)]
    za = [0.9 * (t - 0.4) if t > 0.4 else -0.9 * 0.4 * 0.0 for t in tau]
    m = module.derive_metrics(tau, za, exp)
    assert m["best_shift"] == pytest.approx(-0.4, abs=0.02)
    assert m["tau_start"] == pytest.approx(0.4 + 0.05 / 0.9, abs=0.03)
    assert m["mae_after_shift"] < 0.02


def test_document_is_generated_from_the_artifact(artifact):
    if not DOCUMENT.exists():
        pytest.skip("attribution document not present")
    text = DOCUMENT.read_text()
    assert "Generated file" in text
    for r in artifact["results"]:
        mae = r["experimental_error"]["mean_absolute_error"]
        assert f"{mae:.4f}" in text, (
            f"MAE {mae:.4f} for dx={r['dx']} mu_w={r['wall_friction']} is "
            "missing from the document: it was hand-edited after generation")
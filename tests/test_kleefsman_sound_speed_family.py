"""Regression guard for the hard-coded-sound-speed dt defect family (Kleefsman case).

Background: ``benchmarks/kleefsman_obstacle/run.py`` used to derive the time step
as ``dt = min(2e-4, 0.3 * 1.3 * dx / 20.0)`` -- the artificial sound speed was a
literal, so a run with c0 > ~78 m/s silently stepped past the acoustic Courant
limit (found and fixed in m16758).  Two sibling drivers, ``chunked.py`` and
``convergence.py``, kept the same literal.  These tests pin the fix down at the
configuration level (not by reading the source text), so the defect cannot come
back through a refactor.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
BENCH = HERE / "benchmarks" / "kleefsman_obstacle"
GEOM = json.loads((BENCH / "reference.json").read_text())["geometry_m"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"_kl_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chunked = _load("chunked", BENCH / "chunked.py")
convergence = _load("convergence", BENCH / "convergence.py")
ladder = _load("grid_ladder", BENCH / "grid_ladder_c080.py")
from tensorsph.kleefsman_obstacle import validate_case as _validate_case  # noqa: E402


@pytest.mark.parametrize("module", [chunked, convergence])
@pytest.mark.parametrize("c0", [20.0, 40.0, 80.0, 150.0])
@pytest.mark.parametrize("dx", [0.04, 0.02, 0.01])
def test_driver_dt_obeys_acoustic_courant(module, c0, dx):
    cfg = module.driver_config(dx, c0, 1.2, GEOM)
    assert cfg.sound_speed == c0
    assert cfg.dt == pytest.approx(min(2.0e-4, cfg.dt_acoustic()))
    assert cfg.dt <= cfg.dt_acoustic() * (1.0 + 1e-12)
    # the configuration must also survive the solver's own validator
    _validate_case(cfg)


@pytest.mark.parametrize("module", [chunked, convergence])
def test_default_sound_speed_keeps_legacy_dt(module):
    """Backwards compatibility: c0 = 20 m/s still gives the historical 2e-4 s."""
    cfg = module.driver_config(0.04, 20.0, 1.2, GEOM)
    assert cfg.dt == pytest.approx(2.0e-4)


@pytest.mark.parametrize("module", [chunked, convergence])
def test_legacy_literal_dt_would_violate_the_cfl(module):
    """The defect the fix removes, demonstrated on the real numbers."""
    cfg = module.driver_config(0.04, 80.0, 1.2, GEOM)
    legacy_dt = min(2.0e-4, 0.3 * 1.3 * 0.04 / 20.0)      # the old expression
    assert cfg.dt_acoustic() < legacy_dt                  # 1.95e-4 < 2e-4
    broken = module.driver_config(0.04, 80.0, 1.2, GEOM)
    broken.dt = legacy_dt
    with pytest.raises(Exception):                        # validate_case rejects it
        _validate_case(broken)


def _divisions_by_literal_20(path: Path):
    """AST hits of ``<something> / 20.0`` (string literals are invisible here)."""
    tree = ast.parse(path.read_text(), str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            rhs = node.right
            if isinstance(rhs, ast.Constant) and rhs.value == 20.0:
                hits.append(node.lineno)
    return hits


def test_no_hardcoded_sound_speed_divisor_left():
    """No benchmark may divide a dt-like expression by a literal 20.0 any more."""
    offenders = {p.name: _divisions_by_literal_20(p) for p in sorted(BENCH.glob("*.py"))}
    offenders = {k: v for k, v in offenders.items() if v}
    assert offenders == {}, f"hard-coded 20.0 divisor found in {offenders}"


# --------------------------------------------------------------------------- #
# Ladder analysis (pure arithmetic, no solver)
# --------------------------------------------------------------------------- #
def test_ladder_order_is_exact_for_a_second_order_sequence():
    f = lambda dx: 0.40 + 5.0 * dx ** 2          # noqa: E731
    res = ladder.ladder_metrics((0.04, 0.02, 0.01), tuple(f(dx) for dx in (0.04, 0.02, 0.01)))
    assert res["verdict"] == "monotone_converging"
    assert res["observed_order"] == pytest.approx(2.0, rel=1e-9)
    assert res["gci_fine"] > 0.0


def test_ladder_flags_an_oscillating_sequence():
    res = ladder.ladder_metrics((0.04, 0.02, 0.01), (0.513, 0.455, 0.470))
    assert res["verdict"] == "oscillatory"
    assert res["observed_order"] is None and res["gci_fine"] is None


def test_ladder_flags_divergence():
    res = ladder.ladder_metrics((0.04, 0.02, 0.01), (0.40, 0.45, 0.52))
    assert res["verdict"] == "monotone_diverging"


def test_ladder_rejects_wrong_length():
    with pytest.raises(ValueError):
        ladder.ladder_metrics((0.04, 0.02), (0.4, 0.45))


def test_committed_c0_scan_rows_are_internally_consistent():
    """The two committed levels the ladder reuses must still parse and agree."""
    outdir = BENCH / "results" / "c0_scan"
    for dx in (0.04, 0.02):
        row = json.loads((outdir / f"dx{dx:g}_c080.json").read_text())
        assert row["sound_speed"] == 80.0
        assert row["dt_nominal"] <= row["dt_acoustic_limit"] * (1.0 + 1e-12)
        assert math.isfinite(row["peak_height_m"]) and row["peak_height_m"] > 0.0
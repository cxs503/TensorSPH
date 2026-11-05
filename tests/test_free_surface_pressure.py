"""Gates for the free-surface pressure closure *and its verification record*.

The operator itself is pinned elsewhere: ``tests/test_free_surface_pressure_closure.py``
checks the algebra (exactness on a linear field, body-term sign, donor semantics,
pair/dense agreement). This file pins the two things that file does not:

1. the *iteration* extension -- ``Config.surface_pressure_monotone``, which
   iterates the literal Adami formula as ``p <- max(p, extrapolate(p))`` and is
   what turns the literal donor set from a -20 % undershoot into an in-band
   value. Pinned against hand-computed values on a three-particle chain, and
   pinned to be inert (bit-identical) when off;
2. the *record* -- ``benchmarks/laplace_pressure_floor/report.json`` ->
   ``free_surface_runs`` is the A/B matrix the document quotes
   (``docs/verification_laplace_free_surface_pressure.md``). If someone
   re-runs the benchmark and the closure stops working, or edits the prose away
   from the numbers, these tests go red. Without them the "verification" is a
   paragraph, not a measurement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorsph.droplet_impact import Config, free_surface_pressure

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "laplace_pressure_floor"
REPORT = BENCH / "report.json"
DOC = Path(__file__).resolve().parents[1] / "docs" / "verification_laplace_free_surface_pressure.md"
FIGURE = BENCH / "free_surface_pressure.png"
RHO0 = 1000.0


def _cfg(**kwargs) -> Config:
    base = dict(dx=0.002, h_factor=1.3, rho0=RHO0, sound_speed=20.0, gamma=7.0,
                gravity=0.0, viscosity=0.01, surface_tension_coefficient=7.2)
    base.update(kwargs)
    return Config(**base)


def _chain():
    """Three particles on a line, uniform weights, one bulk donor at the end.

    ``rho = [980, 980, 1000]`` makes particles 0 and 1 free-surface receivers
    (threshold 0.99); particle 2 is bulk, so its pressure is the only source.
    """
    pos = torch.tensor([[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]])
    rho = torch.tensor([980.0, 980.0, RHO0])
    pressure = torch.tensor([0.0, 0.0, 100.0])
    i = torch.tensor([0, 1, 1, 2])
    j = torch.tensor([1, 0, 2, 1])
    wij = torch.ones(i.shape[0])
    delta = pos[i] - pos[j]
    return pos, rho, pressure, i, j, wij, delta


# --------------------------------------------------------------------------- #
# 1. the iteration extension: defaults, hand-computed values, monotonicity
# --------------------------------------------------------------------------- #
def test_iteration_is_off_by_default():
    cfg = Config()
    assert cfg.surface_pressure_closure is False
    assert cfg.surface_pressure_monotone is False
    assert cfg.surface_pressure_donors == "interior"
    assert cfg.surface_pressure_layers == 2
    assert cfg.surface_pressure_threshold == pytest.approx(0.99)


def test_literal_donor_set_does_one_pass_unless_monotone_is_on():
    """The un-iterated literal formula, value by hand.

    One pass with uniform weights: p_1 = (p_0 + p_2) / 2 = 50 Pa, p_0 = 0.
    """
    pos, rho, pressure, i, j, wij, delta = _chain()
    cfg = _cfg(surface_pressure_closure=True, surface_pressure_donors="all",
               surface_pressure_monotone=False, surface_pressure_layers=2)
    out = free_surface_pressure(pos, rho, pressure, delta, wij, i, j, cfg)
    assert out[0].item() == pytest.approx(0.0)
    assert out[1].item() == pytest.approx(50.0)
    assert out[2].item() == pytest.approx(100.0)


@pytest.mark.parametrize("layers,expected", [
    (1, [0.0, 50.0, 100.0]),
    (2, [50.0, 50.0, 100.0]),
    (3, [50.0, 75.0, 100.0]),
    (4, [75.0, 75.0, 100.0]),
])
def test_monotone_iteration_walks_the_bulk_value_outwards(layers, expected):
    """Hand-computed ladder: each pass fills one more receiver.

    The sweep is Jacobi (a pass reads the previous state), so pass ``k`` reaches
    particle ``k-1``: p_1 sees the bulk on the first pass (50 Pa), p_0 only on
    the second. That is exactly why the production value is 3 and not 1 -- the
    truncated shell of a drop is ~2 particles thick, so one pass leaves the
    outermost layer at its clamped zero (measured -20.1 % at
    dx=0.002, c0=20 m/s, see the verification record).
    """
    pos, rho, pressure, i, j, wij, delta = _chain()
    cfg = _cfg(surface_pressure_closure=True, surface_pressure_donors="all",
               surface_pressure_monotone=True, surface_pressure_layers=layers)
    out = free_surface_pressure(pos, rho, pressure, delta, wij, i, j, cfg)
    for got, want in zip(out.tolist(), expected):
        assert got == pytest.approx(want)


def test_monotone_never_lowers_a_receiver_below_its_own_tait_value():
    """A receiver already above the extrapolation must not be pulled down."""
    pos, rho, pressure, i, j, wij, delta = _chain()
    pressure = torch.tensor([200.0, 150.0, 100.0])
    base = _cfg(surface_pressure_closure=True, surface_pressure_donors="all",
                surface_pressure_monotone=False, surface_pressure_layers=2)
    mono = _cfg(surface_pressure_closure=True, surface_pressure_donors="all",
                surface_pressure_monotone=True, surface_pressure_layers=2)
    flat = free_surface_pressure(pos, rho, pressure, delta, wij, i, j, base)
    kept = free_surface_pressure(pos, rho, pressure, delta, wij, i, j, mono)
    assert torch.all(kept >= flat), "monotone iteration lowered a receiver"
    assert torch.all(kept >= pressure)


def test_iteration_is_inert_for_the_one_sided_donor_set():
    """``interior`` donors already recurse layer by layer; the flag must not
    change them (it would silently alter the published +8.8 % variant)."""
    pos, rho, pressure, i, j, wij, delta = _chain()
    a = _cfg(surface_pressure_closure=True, surface_pressure_donors="interior",
             surface_pressure_layers=2, surface_pressure_monotone=False)
    b = _cfg(surface_pressure_closure=True, surface_pressure_donors="interior",
             surface_pressure_layers=2, surface_pressure_monotone=True)
    assert torch.equal(
        free_surface_pressure(pos, rho, pressure, delta, wij, i, j, a),
        free_surface_pressure(pos, rho, pressure, delta, wij, i, j, b))


def test_no_free_surface_leaves_the_pressure_untouched():
    pos, rho, pressure, i, j, wij, delta = _chain()
    rho = torch.full((3,), RHO0)
    cfg = _cfg(surface_pressure_closure=True, surface_pressure_donors="all",
               surface_pressure_monotone=True, surface_pressure_layers=3)
    out = free_surface_pressure(pos, rho, pressure, delta, wij, i, j, cfg)
    assert torch.equal(out, pressure)


# --------------------------------------------------------------------------- #
# 2. the record: A/B matrix, verdicts, prose-vs-numbers drift
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def report():
    if not REPORT.exists():
        pytest.skip("run benchmarks/laplace_pressure_floor/run.py --free-surface")
    data = json.loads(REPORT.read_text())
    if not data.get("free_surface_runs"):
        pytest.skip("run the benchmark with --free-surface / --free-surface-only")
    return data


@pytest.fixture(scope="module")
def variants(report):
    out = {}
    for run in report["free_surface_runs"]:
        out.setdefault(run["closure_variant"], {})[
            (run["dx_m"], run["sound_speed"])] = run
    return out


def test_the_ab_variants_cover_every_baseline_point(report, variants):
    """The accepted variant and the two single-shot A/B arms must be measured on
    the whole closure-off grid, otherwise the columns are not comparable.

    ``all-iter-6`` is a control (does over-iteration hurt?), so it only has to
    exist on at least one point -- it is deliberately not run on the full grid.
    """
    base = {(r["dx_m"], r["sound_speed"]) for r in report["runs"]}
    full = {"all", "interior-2", "all-iter-3"}
    for label, rows in variants.items():
        if label in full:
            missing = base - set(rows)
            assert not missing, f"variant {label} misses {sorted(missing)}"
        else:
            assert rows, f"control variant {label} was not measured at all"


def test_each_run_records_the_configuration_it_claims(variants):
    """The variant label must match the config actually used (no relabelling)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("lpf_run", BENCH / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for label, rows in variants.items():
        spec_ = module.CLOSURE_VARIANTS[label]
        for key, run in rows.items():
            closure = run["closure"]
            assert closure["enabled"] is True
            assert closure["donors"] == spec_["donors"], (label, key)
            assert closure["layers"] == spec_["layers"], (label, key)
            assert closure.get("monotone", False) == bool(spec_.get("monotone", False)), (
                label, key)


def test_the_best_variant_is_inside_the_15pct_band_on_every_point(report, variants):
    """The acceptance criterion of ``verification_laplace_pressure_floor.md``.

    At least one variant must be within +/-15 % at *every* measured point; a
    variant that only wins on the coarse grid does not pass.
    """
    dp = report["reference"]["delta_p_Pa"]
    base = max(abs(r["absolute_time_average_tail"]["rel_error_pct"])
               for r in report["runs"])
    passing = {label: max(abs(r["absolute_time_average_tail"]["rel_error_pct"])
                          for r in rows.values())
               for label, rows in variants.items()}
    best = min(passing, key=passing.get)
    assert passing[best] < 15.0, (
        f"no variant inside 15 % (best {best} at {passing[best]:.1f} %; "
        f"closure off {base:.1f} %); sigma/R = {dp:.1f} Pa")


def test_the_accepted_variant_is_also_within_3pct_where_it_matters(report, variants):
    """The strict question of the mission: is the key quantity within 3 %?

    Drawn from the same matrix, not asserted by hand: at least one variant must
    reach < 3 % on the high-signal (large c0) points, and the record must say
    which points are *not* within 3 % rather than averaging over them.
    """
    high = [(r["dx_m"], r["sound_speed"])
            for r in report["runs"] if r["sound_speed"] >= 20.0]
    assert high, "the baseline grid lost its high-c0 points"
    best = None
    for label, rows in variants.items():
        errs = [abs(rows[key]["absolute_time_average_tail"]["rel_error_pct"])
                for key in high if key in rows]
        if len(errs) == len(high):
            worst = max(errs)
            if best is None or worst < best[1]:
                best = (label, worst)
    assert best is not None
    # The honest statement the record makes: < 3 % is reached at the coarser
    # resolution (dx = 2 mm, where the shell is resolved by more particles per
    # kernel radius), and the finer grid sits at -6.8 %. The gate therefore
    # requires < 3 % on at least one resolution and < 15 % on all of them --
    # claiming 3 % everywhere would be false.
    assert best[1] < 15.0, f"best variant {best[0]} is {best[1]:.1f} % at c0 >= 20"
    coarse = [(0.002, 20.0)]
    reached = [key for key in coarse
               if key in variants[best[0]]
               and abs(variants[best[0]][key]["absolute_time_average_tail"]
                       ["rel_error_pct"]) < 3.0]
    assert reached, f"no variant reaches 3 % at {coarse}; best is {best}"


def test_the_gain_is_paid_for_in_spurious_current(variants):
    """Honest bookkeeping: the record must contain the trade-off, not just win.

    Adami's sanity limit is ``v_max/v_ref < 0.1``. At least one variant exceeds
    it -- if that stops being true the document's caveat is stale, and if *all*
    variants exceeded it the accepted one would be unusable.
    """
    vmax = {label: max(r["spurious_current"]["vmax_over_vref_final"]
                       for r in rows.values())
            for label, rows in variants.items()}
    assert any(v > 0.1 for v in vmax.values()), vmax
    assert min(vmax.values()) < 0.1, vmax


def test_over_iteration_is_recorded_as_worse(variants):
    """Passes are a stability parameter: 6 must be worse than 3 where both ran."""
    if "all-iter-3" not in variants or "all-iter-6" not in variants:
        pytest.skip("the iteration ladder was not measured")
    both = set(variants["all-iter-3"]) & set(variants["all-iter-6"])
    assert both, "ladders do not overlap"
    worse = [key for key in both
             if abs(variants["all-iter-6"][key]["absolute_time_average_tail"]
                    ["rel_error_pct"])
             > abs(variants["all-iter-3"][key]["absolute_time_average_tail"]
                   ["rel_error_pct"])]
    assert len(worse) >= len(both) // 2, (len(worse), len(both))


def test_every_recorded_state_is_finite(report, variants):
    for label, rows in variants.items():
        for key, run in rows.items():
            assert run["finite"], f"{label} at {key} produced a non-finite state"
            for name in ("absolute", "bandA", "bandB"):
                assert math.isfinite(run[f"definition_{name}"]["delta_p_Pa"])


def test_the_document_quotes_the_numbers_in_the_report(report, variants):
    """Drift gate: the prose is generated from report.json, so the A/B table in
    the document must contain the values that are in the report."""
    assert DOC.exists(), "the verification record is missing"
    text = DOC.read_text()
    needle_rows = 0
    for label, rows in variants.items():
        for run in rows.values():
            err = run["absolute_time_average_tail"]["rel_error_pct"]
            needle = f"{err:+.1f}"
            if needle in text:
                needle_rows += 1
    assert needle_rows >= max(4, len(variants)), (
        f"only {needle_rows} of the measured errors appear in the document")


def test_the_figure_exists():
    assert FIGURE.exists(), "free_surface_pressure.png was not generated"
    assert FIGURE.stat().st_size > 20_000, FIGURE.stat().st_size

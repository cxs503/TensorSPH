"""Tests for ``CASE-HYDROSTATIC`` (still-water pressure profile).

Two layers:

* **Unit / live checks** (fast, no committed artifacts): the analytic reference,
  the gate arithmetic, the free-surface overlap rule, one live ``rhs``
  evaluation of the lattice (interior complete-support column is exact, the
  wall columns are not), and a coarse smoke run of the case.
* **Artifact checks** (read ``report.json`` + the run files next to it): the
  shipped gate value is the one in the run file, it is the pre-declared config,
  and it is inside the 1 % limit.  These assert an *invariant of the shipped
  evidence* (a report that disagrees with its own run files fails here), not
  that a simulation was executed; the simulation is reproducible with
  ``python benchmarks/case_hydrostatic/run.py --config window_8s``.
* **Shipped-mirror checks** (``outputs/case_hydrostatic``): the delivery copy of
  the bundle is complete (report + every run file + the figures) and matches its
  own sha256 manifest, so a half-copied or hand-edited bundle fails here.  The
  mirror exists because ``runs/`` is outside the delivery whitelist
  (``docs/ src/ tests/ examples/ outputs/``): evidence kept only in ``runs/``
  never reaches a fresh checkout.
* **Harness checks**: ``--config all --static`` must schedule *both* layers of
  the declared grid, and ``--aggregate`` must refuse to emit a report with no
  gate record (it used to die with a ``TypeError`` on ``None``).
* **Reproducibility layer** (fast, always runs): the time-stepped gate needs
  ~11 min, so the only re-run check used to skip on a fresh checkout -- a check
  that never runs where the evidence is read.  The deterministic ``t = 0``
  lattice record does not: every shipped ``runs/static_*.json`` is re-derived
  from source here (~2 s for all seven) and diffed band by band, so knob drift,
  a changed band partition or a moved truncation ladder fails CI.  The gate run
  itself stays reproducible with
  ``python benchmarks/case_hydrostatic/run.py --config window_8s``.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.case_hydrostatic import (CASE_ID, DEPTH, G, RHO0,  # noqa: E402
                                       TOLERANCE_PCT, analytic_pressure,
                                       evaluate, run_case, run_initial_state,
                                       support_width)

#: Evidence root, most-authoritative first: a locally regenerated ``runs/``
#: bundle wins over ``outputs/case_hydrostatic`` (the full mirror with figures),
#: which wins over the *shippable* text channel ``tests/data/case_hydrostatic``.
#:
#: Why a third root: the full mirror is artifact-class in the delivery rule
#: (``outputs/`` + ``*.png``, merge-governor artifact cap), so a PR carrying it
#: is held and never lands -- on a fresh checkout of ``main`` the mirror is
#: absent and this whole layer used to ERROR out, i.e. the 1 % acceptance of the
#: case was not decidable where it is actually read.  The channel holds the same
#: numbers as text (``report.json`` + every ``runs/*.json`` + ``MANIFEST.sha256``
#: + ``channel.json``), written by ``benchmarks/case_hydrostatic/freeze.py``;
#: the figures are re-creatable from those numbers and are the only part the
#: delivery rule drops.
EVIDENCE_CANDIDATES = (REPO / "runs" / "case_hydrostatic",
                       REPO / "outputs" / "case_hydrostatic",
                       REPO / "tests" / "data" / "case_hydrostatic")
CHANNEL = REPO / "tests" / "data" / "case_hydrostatic"
MANIFEST = "MANIFEST.sha256"
CHANNEL_INDEX = "channel.json"


def shipped_base() -> Path:
    """The shipped evidence root: full mirror if it landed, else the channel."""
    if (REPO / "outputs" / "case_hydrostatic" / "report.json").exists():
        return REPO / "outputs" / "case_hydrostatic"
    assert (CHANNEL / "report.json").exists(), (
        f"no shipped CASE-HYDROSTATIC evidence under {CHANNEL}; freeze one with "
        "`python benchmarks/case_hydrostatic/run.py --config all --static` + "
        "`--aggregate --figure` then `python benchmarks/case_hydrostatic/freeze.py`")
    return CHANNEL


def evidence_base() -> Path:
    for base in EVIDENCE_CANDIDATES:
        if (base / "report.json").exists():
            return base
    raise AssertionError(
        "no CASE-HYDROSTATIC evidence found; expected one of "
        f"{[str(b) for b in EVIDENCE_CANDIDATES]}. Reproduce with "
        "`python benchmarks/case_hydrostatic/run.py --config all --static` then "
        "`--aggregate --figure` (mirror) and `python "
        "benchmarks/case_hydrostatic/freeze.py` (shippable text channel).")


def _harness():
    """Import the benchmark harness script (``benchmarks/`` is not a package)."""
    import importlib.util

    path = REPO / "benchmarks" / "case_hydrostatic" / "run.py"
    spec = importlib.util.spec_from_file_location("case_hydrostatic_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Analytic reference and gate arithmetic
# --------------------------------------------------------------------------- #

def test_analytic_reference_is_hydrostatic_equilibrium():
    y = torch.tensor([0.0, 0.1, 0.2, 0.3, DEPTH])
    p = analytic_pressure(y)
    assert torch.allclose(p, RHO0 * G * (DEPTH - y), rtol=0, atol=1e-9)
    assert float(p[0]) == pytest.approx(RHO0 * G * DEPTH)
    assert float(p[-1]) == 0.0
    # dp/dy = -rho g: finite-difference check of the closed form itself
    depths = torch.linspace(0.05, 0.35, 7, dtype=torch.float64)
    dp = (analytic_pressure(depths + 1e-4) - analytic_pressure(depths - 1e-4)) / 2e-4
    assert torch.allclose(dp, torch.full_like(dp, -RHO0 * G), rtol=1e-6, atol=1e-6)


def _row(band, y, p, p_ref):
    return {"band": band, "y": y, "y_low": y - 0.05, "y_high": y + 0.05,
            "particles": 10, "pressure": p, "pressure_analytic": p_ref,
            "rel_error_pct": (p - p_ref) / p_ref * 100.0, "free_surface": False}


def test_gate_takes_the_worst_interior_band_only():
    rows = [_row(0, 0.05, 3900.0, RHO0 * G * (DEPTH - 0.05)),
            _row(1, 0.15, 2400.0, RHO0 * G * (DEPTH - 0.15)),
            _row(2, 0.25, 1400.0, RHO0 * G * (DEPTH - 0.25))]
    gate = evaluate(rows)
    assert gate["limit_pct"] == TOLERANCE_PCT
    assert gate["max_band_error_pct"] == pytest.approx(
        max(abs(r["rel_error_pct"]) for r in rows))
    assert gate["worst_band"] == 0
    assert gate["pass"] is False          # the 0.5 %-low interior band is off
    rows[0]["pressure"] = rows[0]["pressure_analytic"] * 1.002   # +0.2 %
    rows[1]["pressure"] = rows[1]["pressure_analytic"] * 1.003
    rows[2]["pressure"] = rows[2]["pressure_analytic"] * 0.999
    gate = evaluate(rows)
    assert gate["pass"] is True and gate["max_band_error_pct"] < TOLERANCE_PCT


def test_free_surface_overlap_rule_excludes_the_shelf_band():
    """A band whose *centre* is deeper than 2h but whose *upper edge* is inside
    the kernel support of the surface must be excluded from the gate."""
    dx = 0.02
    rows = run_initial_state(dx=dx)["profile"]
    support = support_width(dx)
    shelf = [r for r in rows if r["y_high"] > DEPTH - support]
    assert shelf and all(r["free_surface"] and not r["gated"] for r in shelf)
    assert rows[-1]["free_surface"] is True              # topmost band always out
    assert all(not r["free_surface"] for r in rows if r["y_high"] <= DEPTH - support)
    # the exclusion band is 2h = 2.6 dx wide: at 8 bands over a 20-cell column
    # (2.5 cm bands) exactly the top two bands fall inside it
    assert len(shelf) == 2


# --------------------------------------------------------------------------- #
# Live solver checks
# --------------------------------------------------------------------------- #

def test_static_lattice_is_exact_away_from_the_walls():
    """One ``rhs`` evaluation: complete-support interior is < 0.1 %, the wall
    columns are an order of magnitude worse (that is the residual the tank mean
    carries)."""
    record = run_initial_state(dx=0.02)
    rows = [r for r in record["profile"] if r["gated"]]
    assert rows
    # interior: away from floor row, away from the side walls
    support = support_width(0.02)
    interior = rows[1:] if len(rows) > 1 else rows
    assert max(abs(r["rel_error_pct"]) for r in interior) < 5.0

    # complete-support columns only (the lateral truncation is the mechanism)
    from tensorsph.case_hydrostatic import _pool
    from tensorsph.droplet_impact import PairListCache, rhs, wall_lattice
    from tensorsph.vv.hydrostatic import HydrostaticConfig, band_table

    dx, width = 0.02, 1.0
    cfg = HydrostaticConfig(dx=dx, tank_width=width, depth=DEPTH)
    pos, mass, solver = _pool(cfg)
    wall = wall_lattice(solver, pos.device, top=DEPTH)
    h = solver.h_factor * dx
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=pos.shape[0])
    pairs, _ = cache.get(torch.cat((pos, wall), dim=0))
    _, pressure, _ = rhs(pos, torch.zeros_like(pos), mass, solver, pairs=pairs,
                         wall=wall, dt=0.3 * h / 20.0)
    support = support_width(dx)
    keep = (pos[:, 0] >= support) & (pos[:, 0] <= width - support)
    rows = band_table(pos[keep], pressure[keep], mass[keep], cfg)
    inner = [abs(r["rel_error_pct"]) for r in rows[1:] if not r["free_surface"]]
    assert inner and max(inner) < 0.1

    outer = (pos[:, 0] < 1.5 * dx) | (pos[:, 0] > width - 1.5 * dx)
    mid = pos[:, 1] < DEPTH - support
    ref_all = analytic_pressure(pos[outer & mid, 1])
    wall_err = ((pressure[outer & mid] - ref_all) / ref_all).abs().median()
    assert float(wall_err) * 100.0 > 10.0 * max(inner)


@pytest.mark.slow
def test_negative_control_measures_gravity():
    """``g = 0`` in the solver, reference unchanged -> the case must fail hard."""
    record = run_case(dx=0.05, tank_width=0.5, seconds=1.0, bands=5,
                      sample_every=10, gravity_override=0.0)
    assert abs(record["gate"]["mean_band_error_pct"]) > 80.0
    assert record["gate"]["pass"] is False


@pytest.mark.slow
def test_smoke_case_produces_a_benchmark_table():
    record = run_case(dx=0.05, tank_width=0.5, seconds=1.0, bands=5, sample_every=10)
    assert record["case_id"] == CASE_ID
    assert record["reference"]["formula"] == "p(y) = rho0 * g * (H - y)"
    assert record["config"]["particle_requirement_met"] is False  # smoke size
    assert record["benchmark"], "no gated band -> nothing was measured"
    for row in record["benchmark"]:
        # every benchmark row carries ref and got and the error between them
        assert row["ref_Pa"] > 0 and row["got_Pa"] > 0
        assert row["rel_error_pct"] == pytest.approx(
            (row["got_Pa"] - row["ref_Pa"]) / row["ref_Pa"] * 100.0)
    assert record["gate"]["max_band_error_pct"] == pytest.approx(
        max(abs(r["rel_error_pct"]) for r in record["benchmark"]))


# --------------------------------------------------------------------------- #
# Shipped evidence
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def evidence():
    return evidence_base()


@pytest.fixture(scope="module")
def report(evidence):
    return json.loads((evidence / "report.json").read_text())


def test_shipped_gate_is_the_predeclared_config_and_within_limit(report, evidence):
    verdict = report["verdict"]
    assert verdict["case_id"] == CASE_ID
    assert verdict["gate_limit_pct"] == TOLERANCE_PCT
    assert verdict["gate_config"] == "window_8s"
    run = json.loads((evidence / "runs" / f"{verdict['gate_config']}.json").read_text())
    assert verdict["gate_value_pct"] == run["gate"]["max_band_error_pct"]
    assert verdict["gate_pass"] is bool(verdict["gate_value_pct"] <= TOLERANCE_PCT)
    assert verdict["gate_pass"] is True


def test_every_gated_benchmark_row_has_ref_and_got(report, evidence):
    run = json.loads((evidence / "runs" / f"{report['verdict']['gate_config']}.json").read_text())
    assert len(run["benchmark"]) >= 4
    for row in run["benchmark"]:
        assert row["ref_Pa"] > 0
        assert row["within_tolerance"] is True
    assert run["config"]["fluid_particles"] >= run["config"]["min_particles"]


def test_report_quotes_the_run_files(report, evidence):
    for run in report["runs"]:
        stored = json.loads((evidence / "runs" / f"{run['config_name']}.json").read_text())
        assert run == stored
    for static in report["static_runs"]:
        path = evidence / "runs" / f"static_{static['config_name']}.json"
        stored = json.loads(path.read_text())
        assert static == stored


def test_repro_command_is_recorded(report):
    repro = report["repro"]
    assert "python benchmarks/case_hydrostatic/run.py --config window_8s" == repro["gate"]
    assert "runs/case_hydrostatic" in repro["one_shot"] + repro["grid"]

def _manifest(root: Path) -> dict[str, tuple[str, int]]:
    entries = {}
    for line in (root / MANIFEST).read_text().splitlines():
        if not line.strip():
            continue
        digest, size, rel = line.split("  ", 2)
        entries[rel] = (digest, int(size))
    return entries


def test_shipped_mirror_is_manifest_complete(report):
    """The shipped copy must carry the report, every run file and a manifest
    that covers exactly those files with the right digests.

    Full mirror (``outputs/``) -> the figures must be there too.  Text channel
    (``tests/data/``) -> no artifact-class file may be relied on, and
    ``channel.json`` must record exactly the same file list plus what the
    delivery rule made it drop (figures only, and they are re-creatable from
    the shipped numbers).
    """
    import hashlib

    shipped = shipped_base()
    assert (shipped / "report.json").exists(), (
        f"shipped evidence root {shipped} is missing report.json")
    entries = _manifest(shipped)
    actual = {p.relative_to(shipped).as_posix()
              for p in shipped.rglob("*") if p.is_file()} - {MANIFEST, CHANNEL_INDEX}
    assert set(entries) == actual, (
        f"manifest/files disagree: unlisted={sorted(actual - set(entries))[:5]}, "
        f"missing={sorted(set(entries) - actual)[:5]}")
    for rel, (digest, size) in entries.items():
        data = (shipped / rel).read_bytes()
        assert len(data) == size, f"{rel}: size {len(data)} != manifest {size}"
        assert hashlib.sha256(data).hexdigest() == digest, f"{rel}: sha256 drift"
    # the shipped bundle is the *whole* case, not just the verdict
    assert "report.json" in entries
    for run in report["runs"]:
        assert f"runs/{run['config_name']}.json" in entries
    for static in report["static_runs"]:
        assert f"runs/static_{static['config_name']}.json" in entries
    index_path = shipped / CHANNEL_INDEX
    if index_path.exists():
        index = json.loads(index_path.read_text())
        assert set(index["files"]) == set(entries)
        assert index["gate_value_pct"] == report["verdict"]["gate_value_pct"]
        assert index["gate_limit_pct"] == report["verdict"]["gate_limit_pct"]
        assert all(not rel.endswith((".png", ".npz", ".pt")) for rel in entries), (
            "the shipped text channel must not depend on artifact-class files")
        assert index["dropped_not_shippable"], (
            "channel.json must state which bundle files were dropped and why")
    else:
        assert any(rel.startswith("figures/") for rel in entries), (
            "the full mirror must ship the figures")


def test_gate_value_is_recomputable_from_the_shipped_band_rows(report):
    """Anti-hand-copy invariant: the shipped gate number, every band error and
    the reference column must all be re-derivable from the shipped raw band
    rows, against the *analytic* ``rho0 g (H - y)`` evaluated here.

    A hand-edited verdict, a transcribed reference table that drifted from the
    closed form, or a band error that does not follow from ``got_Pa``/``ref_Pa``
    fails this test even though the manifest would still be self-consistent.
    """
    shipped = shipped_base()
    gate_config = report["verdict"]["gate_config"]
    run = json.loads((shipped / "runs" / f"{gate_config}.json").read_text())
    ref = run["reference"]
    assert ref["formula"] == "p(y) = rho0 * g * (H - y)"
    assert ref["kind"] == "analytic"
    rows = run["benchmark"]
    assert len(rows) >= 4, "no gated band rows in the shipped gate run"
    for row in rows:
        p_ref = ref["rho0"] * ref["g"] * (ref["H"] - row["y_m"])  # closed form, here
        assert row["ref_Pa"] == pytest.approx(p_ref, rel=1e-9), (
            f"band {row['band']}: shipped ref {row['ref_Pa']} != analytic {p_ref}")
        recomputed = (row["got_Pa"] - p_ref) / p_ref * 100.0
        assert row["rel_error_pct"] == pytest.approx(recomputed, rel=1e-6, abs=1e-9)
        assert row["within_tolerance"] is (abs(recomputed) <= TOLERANCE_PCT)
    worst = max(abs(r["rel_error_pct"]) for r in rows)
    assert run["gate"]["max_band_error_pct"] == pytest.approx(worst, rel=1e-9)
    assert report["verdict"]["gate_value_pct"] == run["gate"]["max_band_error_pct"]
    assert report["verdict"]["gate_value_pct"] <= TOLERANCE_PCT
    assert report["verdict"]["gate_pass"] is True


def test_shipped_evidence_agrees_with_a_regenerated_bundle():
    """If this checkout has a fresh ``runs/`` bundle, it must agree with what
    ships.  A byte-identical copy is the strongest form (same machine, same
    build); across machines/architectures the gate number and the pass verdict
    must still agree, otherwise the frozen evidence is not reproducible."""
    fresh = REPO / "runs" / "case_hydrostatic"
    if not (fresh / "report.json").exists():
        pytest.skip("no regenerated runs/ bundle in this checkout; the deterministic"
                    " t = 0 layer is re-derived by "
                    "test_shipped_static_layer_reproduces_on_a_fresh_checkout")
    shipped = shipped_base()
    if (fresh / "report.json").read_bytes() == (shipped / "report.json").read_bytes():
        return
    a = json.loads((shipped / "report.json").read_text())
    b = json.loads((fresh / "report.json").read_text())
    assert a["verdict"]["gate_config"] == b["verdict"]["gate_config"]
    assert a["verdict"]["gate_limit_pct"] == b["verdict"]["gate_limit_pct"]
    assert a["verdict"]["gate_pass"] == b["verdict"]["gate_pass"], (
        f"gate verdict flips on re-run: shipped "
        f"{a['verdict']['gate_value_pct']:.4f} % vs fresh "
        f"{b['verdict']['gate_value_pct']:.4f} %")
    assert abs(a["verdict"]["gate_value_pct"]
               - b["verdict"]["gate_value_pct"]) < 0.05, "gate value not reproducible"
    # same config grid, same gate metric on every shared config
    by_b = {r["config_name"]: r for r in b["runs"]}
    for row in a["runs"]:
        other = by_b.get(row["config_name"])
        if other is None:
            continue
        assert abs(row["gate"]["max_band_error_pct"]
                   - other["gate"]["max_band_error_pct"]) < 0.05, (
            f"{row['config_name']}: gate metric not reproducible")


def test_config_all_with_static_schedules_the_whole_grid(monkeypatch):
    """Regression: ``--config all --static`` used to run the t = 0 layer only,
    so the documented one-shot grid left the gate record absent."""
    harness = _harness()
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(harness, "run_one",
                        lambda name, static=False: calls.append((name, static)) or {})
    monkeypatch.setattr(harness, "aggregate",
                        lambda: {"verdict": {"gate_value_pct": 0.5, "gate_pass": True}})
    monkeypatch.setattr(harness, "ship",
                        lambda: {"dest": "x", "manifest": "y", "files": 0})
    assert harness.main(["--config", "all", "--static"]) == 0
    stepped = [name for name, static in calls if not static]
    static_layer = [name for name, static in calls if static]
    assert stepped == list(harness.CONFIGS), "time-stepped ladder missing"
    assert static_layer == list(harness.CONFIGS), "t = 0 layer missing"
    # the harness schedules every config; ``run_one`` drops the g = 0 static
    # record itself (a single RHS evaluation carries no gravity balance)
    assert harness.run_one("control_g0", static=True) == {}


def test_named_config_with_static_stays_static_only(monkeypatch):
    harness = _harness()
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(harness, "run_one",
                        lambda name, static=False: calls.append((name, static)) or {})
    harness.main(["--config", "window_8s", "--static"])
    assert calls == [("window_8s", True)]


def test_aggregate_refuses_a_missing_gate_record(monkeypatch, tmp_path):
    harness = _harness()
    monkeypatch.setattr(harness, "RUNS", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        harness.aggregate()
    assert "gate config" in str(excinfo.value)


@pytest.mark.slow
def test_package_entry_dispatches_to_the_case(tmp_path):
    """``python -m tensorsph --case hydrostatic`` is a real entry, not a stub."""
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "OMP_NUM_THREADS": "1"}
    out = tmp_path / "record.json"
    proc = subprocess.run(
        [sys.executable, "-m", "tensorsph", "--case", "hydrostatic", "--quick",
         "--json-out", str(out)],
        capture_output=True, text=True, timeout=600, env=env, cwd=str(REPO))
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    record = json.loads(out.read_text())
    assert record["case_id"] == CASE_ID
    assert record["benchmark"], "the CLI record carries no ref-vs-got rows"
    assert "max interior-band error" in proc.stdout


# --------------------------------------------------------------------------- #
# Reproducibility of the shipped evidence (deterministic layer -- always runs)
#
# The time-stepped gate needs the whole declared grid (~11 min), so on a fresh
# checkout the only re-run check used to skip: the case's 1 % acceptance was
# decidable there, but *reproducibility* was not.  The t = 0 lattice record is
# deterministic and costs ~0.3 s per configuration, so every shipped
# ``runs/static_*.json`` is re-derived from source below (~2 s for all seven).
# --------------------------------------------------------------------------- #

#: Float tolerance for the static layer.  The shipped numbers were written by
#: this code; the only legitimate drift is summation order in the density loop
#: (arch / BLAS), which stays far below 1e-9 relative.  A real change -- a band
#: value moving in the 6th digit, a different band partition, a moved ladder --
#: is orders of magnitude larger and fails here.
STATIC_REL = 1e-9


def _static_records() -> list[Path]:
    return sorted((CHANNEL / "runs").glob("static_*.json"))


def _static_kwargs(record: dict) -> dict:
    """The t = 0 knobs, read off the shipped record itself (no hand copy)."""
    cfg = record["config"]
    return dict(dx=cfg["dx"], tank_width=cfg["tank_width"], depth=cfg["depth"],
                sound_speed=cfg["sound_speed"], bands=cfg["bands"])


def _same_gate(shipped: dict, fresh: dict, rel: float = STATIC_REL) -> None:
    """Every gate number/partition the two records share must agree."""
    shared = set(shipped) & set(fresh)
    assert {"max_band_error_pct", "mean_band_error_pct"} <= shared
    for key in sorted(shared):
        a, b = shipped[key], fresh[key]
        if isinstance(a, float) and isinstance(b, float):
            assert b == pytest.approx(a, rel=rel, abs=1e-12), f"gate.{key}: {a} -> {b}"
        else:
            assert b == a, f"gate.{key}: {a!r} -> {b!r}"


def test_shipped_static_layer_reproduces_on_a_fresh_checkout():
    """Re-derive every shipped t = 0 record and diff it band by band.

    This is the check that keeps the frozen evidence honest *without* the
    expensive gate run: a solver change, a moved geometry/band rule or a hand
    edit of the shipped numbers all show up here.
    """
    paths = _static_records()
    assert paths, f"no shipped static records under {CHANNEL}"
    for path in paths:
        shipped = json.loads(path.read_text())
        assert shipped["case_id"] == CASE_ID
        assert shipped["kind"] == "static_initial_state"
        fresh = run_initial_state(**_static_kwargs(shipped))

        assert fresh["config"] == shipped["config"], f"{path.name}: config drift"

        old_rows, new_rows = shipped["benchmark"], fresh["benchmark"]
        assert len(new_rows) == len(old_rows), f"{path.name}: band count"
        for a, b in zip(old_rows, new_rows):
            assert a["band"] == b["band"]
            for field in ("y_m", "ref_Pa", "got_Pa", "rel_error_pct"):
                assert b[field] == pytest.approx(a[field], rel=STATIC_REL, abs=1e-12), (
                    f"{path.name} band {a['band']} {field}: {a[field]} -> {b[field]}")
        _same_gate(shipped["gate"], fresh["gate"])
        assert fresh["reference"]["formula"] == shipped["reference"]["formula"]


def test_static_truncation_ladder_is_monotone_and_wall_owned():
    """The t = 0 error is a truncation diagnostic, so it must *shrink* as dx
    shrinks and its worst band must sit on the floor/wall edge.

    Read on freshly computed records (not on the shipped table): this is the
    physical invariant behind the mechanism claim in ``docs/``, and it is what
    makes the static layer useful as a diagnostic even though it is not the gate.
    """
    ladder = []
    for dx in (0.04, 0.025, 0.02):
        rec = run_initial_state(dx=dx, tank_width=1.0)
        ladder.append((dx, rec["gate"]))
    errors = [gate["max_band_error_pct"] for _, gate in ladder]
    assert errors == sorted(errors, reverse=True), f"ladder not monotone: {errors}"
    assert len(set(errors)) == len(errors), f"ladder not strictly decreasing: {errors}"
    for dx, gate in ladder:
        # the coarse lattice truncates the summation density at the floor row;
        # the interior complete-support columns are exact (see the live rhs test)
        assert gate["worst_band"] == 0, f"dx={dx}: worst band {gate['worst_band']}"
        assert gate["worst_band_y"] < 2.0 * dx, "worst band is not the floor band"
    # the coarse static layer is a diagnostic, never the acceptance number
    assert errors[0] > TOLERANCE_PCT


def test_repro_check_detects_a_tampered_static_record(tmp_path):
    """The re-derivation above must be able to fail.

    ``benchmarks/case_hydrostatic/reproduce_static.py`` is the tool version of
    the same check; feed it a shipped record with one band moved by 0.1 % and it
    has to report the record as not reproduced (otherwise the tool -- and the
    evidence it guards -- would be vacuous).
    """
    import importlib.util

    tool_path = REPO / "benchmarks" / "case_hydrostatic" / "reproduce_static.py"
    spec = importlib.util.spec_from_file_location("case_hydrostatic_repro", tool_path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    clean = _static_records()[0]
    record = json.loads(clean.read_text())
    assert tool.repro_one(clean)["reproduced"] is True

    record["benchmark"][1]["got_Pa"] *= 1.001
    tampered = tmp_path / clean.name
    tampered.write_text(json.dumps(record))
    result = tool.repro_one(tampered)
    assert result["reproduced"] is False
    assert any("got_Pa" in d for d in result["diffs"]), result["diffs"]

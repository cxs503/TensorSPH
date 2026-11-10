"""Tests for the free-surface verification face (campaign P482-N04).

The tests are the *fast* versions of the benchmark: smaller spacings and shorter
runs, but the same code paths, the same gates and the same negative controls.
Every gate is exercised twice -- once on a configuration that must pass and once
on its negative control, which must fail -- because a gate that cannot fail is
not a gate.

Nothing here asserts a physical number: the physical results live in
``benchmarks/verified/`` and their values change when the solver changes.  What
is pinned is the *mechanism*: measurement definitions, gate arithmetic, the
negative controls, and the SQL provenance of every reported figure.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tensorsph.verification import (conservation, convergence, dam_break_2d as m1,
                                    hydrostatic as m2, report)
from tensorsph.verification.evidence import EvidenceStore


# --------------------------------------------------------------------------- #
# M1 -- dam break
# --------------------------------------------------------------------------- #

def test_experimental_table_carries_its_audit_grade():
    table = m1.load_experimental()
    assert table["_grade"] == "WITHHELD"
    assert table["data"], "the bundled table must have points"
    # The grade is the repository's own verdict, not something this module made
    # up: it must come from the table's integrity_audit block.
    assert table["integrity_audit"]["grade"] == "WITHHELD"


def test_front_errors_reject_a_shifted_curve():
    """The front gate must fail a curve that is uniformly 30 % short."""
    table = m1.load_experimental()
    a = 0.2
    rows = [{"tau": p["tau"],
             "front_coherent": a + 0.7 * a * p["Z_over_a"],
             "front_raw": a + 0.7 * a * p["Z_over_a"]}
            for p in table["data"]]
    run = {"config": {"column_width": a}, "rows": rows}
    err = m1.front_errors(run, table, min_tau=0.5)
    assert err["max_rel_error"] == pytest.approx(0.3, abs=1e-9)
    assert err["rms_rel_error"] == pytest.approx(0.3, abs=1e-9)


def test_front_errors_pass_an_exact_curve():
    table = m1.load_experimental()
    a = 0.2
    rows = [{"tau": p["tau"], "front_coherent": a + a * p["Z_over_a"],
             "front_raw": a + a * p["Z_over_a"]} for p in table["data"]]
    run = {"config": {"column_width": a}, "rows": rows}
    err = m1.front_errors(run, table, min_tau=0.5)
    assert err["max_rel_error"] < 1e-12


def test_height_anchors_are_exact_for_the_unperturbed_column():
    """A column that does not move must satisfy both exact height anchors."""
    cfg = m1.DamBreak2DConfig(dx=0.04, seconds=0.1)
    a, H, g = cfg.column_width, cfg.column_height, cfg.gravity
    c0 = cfg.c0_analytic
    rows = []
    for k in range(1, 21):
        t = 0.005 * k
        rows.append({"t": t, "tau": 0.0, "h_back": H,
                     "h_dam": H - 0.5 * g * t ** 2})
    run = {"config": cfg.as_dict(), "rows": rows}
    err = m1.height_anchor_errors(run)
    assert err["residual_max_rel_error"] < 1e-12
    assert err["ballistic_max_rel_error"] < 1e-12
    assert err["t_back"] == pytest.approx(0.5 * a / c0)


def test_height_anchor_detects_a_sagging_column():
    cfg = m1.DamBreak2DConfig(dx=0.04, seconds=0.1)
    rows = [{"t": 0.005 * k, "tau": 0.0, "h_back": 0.8 * cfg.column_height,
             "h_dam": 0.8 * cfg.column_height} for k in range(1, 21)]
    err = m1.height_anchor_errors({"config": cfg.as_dict(), "rows": rows})
    assert err["residual_max_rel_error"] == pytest.approx(0.2, abs=1e-9)
    assert err["ballistic_max_rel_error"] > 0.1


@pytest.mark.parametrize("seconds", [0.1])
def test_gravity_off_control_keeps_the_column(seconds):
    """Negative control M1: without gravity the column must not collapse."""
    cfg = m1.DamBreak2DConfig(dx=0.04, seconds=seconds)
    out = m1.negative_control_gravity_off(cfg, seconds=seconds)
    assert out["front_drift_over_dx"] <= out["gate"]["front_drift_over_dx"], out
    assert out["height_drift_rel"] <= out["gate"]["height_drift_rel"], out


def test_gravity_on_does_collapse():
    """The control above must be able to fail: with gravity the front moves."""
    torch.set_num_threads(4)
    cfg = m1.DamBreak2DConfig(dx=0.04, seconds=0.1, sample_interval=0.05)
    result = m1.run(cfg)
    moved = (result["rows"][-1]["front_coherent"] - result["rows"][0]["front_raw"]) / cfg.dx
    assert moved > 0.5, "the collapsing column must travel at least half a spacing"


# --------------------------------------------------------------------------- #
# M2 -- hydrostatic
# --------------------------------------------------------------------------- #

def test_zero_density_control():
    out = m2.zero_density_control()
    assert out["eos_at_zero_density"] == 0.0
    assert out["eos_finite"] and out["eos_non_negative"]
    assert out["empty_domain_ok"]
    assert out["pass"] is True


def test_hydrostatic_gate_rejects_a_wrong_profile():
    """The 2 % gate must fail a pressure field that is 10 % off."""
    fake = {"measured": {"max_band_error_pct": 10.0, "n_particles": 1200,
                         "particle_requirement_met": True},
            "config": {"min_particles": 1000}}
    assert m2.gate(fake, limit_pct=2.0)["pass"] is False
    fake["measured"]["max_band_error_pct"] = 1.0
    assert m2.gate(fake, limit_pct=2.0)["pass"] is True
    # and the particle-count requirement is part of the gate
    fake["measured"]["n_particles"] = 400
    gate = m2.gate(fake, limit_pct=2.0)
    assert gate["pass"] is False and gate["particles_ok"] is False


def test_hydrostatic_case_counts_particles_from_the_lattice():
    case = m2.HydrostaticCase(dx=0.02, tank_width=1.0, depth=0.4)
    assert case.particle_count() == 1000


# --------------------------------------------------------------------------- #
# M3 -- mass conservation
# --------------------------------------------------------------------------- #

def test_closed_tank_conserves_mass():
    torch.set_num_threads(4)
    case = conservation.ConservationCase(dx=0.04, seconds=0.10, open_x=None,
                                         horizon_steps=1000, sample_every=50)
    out = conservation.run(case)
    gate = conservation.gate(out, min_steps=100)
    assert out["measured"]["relative_loss_accounted"] < 1e-12
    assert gate["pass"] is True


def test_open_tank_loses_mass_when_outflow_is_forgotten():
    torch.set_num_threads(4)
    case = conservation.ConservationCase(dx=0.04, seconds=0.20, open_x=0.30,
                                         sample_every=25)
    out = conservation.run(case)
    gate = conservation.gate_overflow(out)
    assert out["measured"]["particles_removed"] > 0, "water must reach the opening"
    # the accounting is float32-limited (sums of particle masses), so the exact
    # value is not zero -- it is 1e-8, i.e. seven orders below the 0.1 % gate
    assert out["measured"]["relative_loss_accounted"] < 1e-6
    assert out["measured"]["relative_loss_unaccounted"] > 0.01
    assert gate["pass"] is True


def test_mass_gate_rejects_a_leak():
    fake = {"measured": {"mass_initial": 1.0, "relative_loss_accounted": 0.02,
                         "steps_run": 1500},
            "history": [{"mass_accounted": 0.98}]}
    gate = conservation.gate(fake, min_steps=1000)
    assert gate["pass"] is False
    # too few steps also fails, even with perfect mass
    fake["measured"]["relative_loss_accounted"] = 0.0
    fake["measured"]["steps_run"] = 10
    assert conservation.gate(fake, min_steps=1000)["pass"] is False


# --------------------------------------------------------------------------- #
# M4 -- convergence
# --------------------------------------------------------------------------- #

def test_interpolation_converges_and_truncated_support_breaks_the_sum():
    torch.set_num_threads(4)
    case = convergence.ConvergenceCase(dxx=(0.02, 0.01, 0.005))
    result = convergence.run(case)
    orders = result["orders"]["normalized"]
    assert orders["min_order"] >= 1.0, orders
    defect = convergence.kernel_sum_defect(dx=0.02)
    assert defect["support_2h"]["mean_defect"] < 1e-3
    assert defect["defect_ratio"] > 50.0, defect


def test_truncated_support_control_breaks_the_hydrostatic_balance():
    torch.set_num_threads(4)
    control = convergence.truncated_support_hydrostatic(support_factor=1.5,
                                                        dx=0.02, seconds=0.4)
    baseline = m2.run_hydrostatic_case(m2.HydrostaticCase(
        dx=0.02, tank_width=0.4, depth=0.4, seconds=0.4, min_particles=1))
    assert control["diverged"] or (control["max_band_error_pct"]
                                   > 2.0 * baseline["measured"]["max_band_error_pct"])


def test_convergence_gate_arithmetic():
    result = {"orders": {"normalized": {"min_order": 0.4}},
              "support_defect": {"defect_ratio": 2.0}}
    gate = convergence.gate(result)
    assert gate["pass"] is False
    result["orders"]["normalized"]["min_order"] = 1.5
    result["support_defect"]["defect_ratio"] = 100.0
    assert convergence.gate(result)["pass"] is True


# --------------------------------------------------------------------------- #
# evidence store and report provenance
# --------------------------------------------------------------------------- #

def test_evidence_store_roundtrip_and_digest(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    run_id = store.add_run("unit_case", {"dx": 0.02}, notes="unit test")
    store.add_metric(run_id, "error", 0.5, unit="1", gate_limit=0.1,
                     verdict="FAIL", provenance="unit test")
    store.add_series(run_id, "curve", [1, 2, 3])
    assert store.metric(run_id, "error").value == 0.5
    digest = store.digest(run_id, "error")
    assert digest["value"] == 0.5 and "SELECT" in digest["sql"]
    assert store.gate_table("unit_case")[0]["metric"] == "error"
    assert store.run_count("unit_case") == 1
    with pytest.raises(KeyError):
        store.metric(run_id, "missing")
    store.close()


def test_metrics_are_write_once(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    run_id = store.add_run("unit_case", {})
    store.add_metric(run_id, "x", 1.0)
    with pytest.raises(Exception):
        store.add_metric(run_id, "x", 2.0)
    store.close()


def test_reference_digest_is_pinned(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "dam_break" / \
        "martin_moyce_1952.json"
    digest = store.add_reference("martin_moyce_1952", path, "WITHHELD", "audit")
    assert len(digest) == 64
    store.close()


def test_report_provenance_covers_every_reported_number(tmp_path):
    """Every number quoted by the report must come back with its SQL."""
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    for case, names in report.REPORTED_METRICS.items():
        run_id = store.add_run(case, {})
        for name in names:
            store.add_metric(run_id, name, 1.0, gate_limit=1.0, verdict="PASS")
    collected = report.collect(store)
    quoted = {p["metric"] for p in collected["provenance"]}
    for names in report.REPORTED_METRICS.values():
        for name in names:
            assert name in quoted, f"{name} is reported without provenance"
    assert all("SELECT" in p["sql"] for p in collected["provenance"])
    store.close()


def test_report_tables_render():
    collected = {"cases": {"c": {"run_id": 1, "metrics": {"m": 0.5}}},
                 "provenance": [{"case": "c", "metric": "m", "run_id": 1,
                                 "value": 0.5, "sql": "SELECT 1"}],
                 "gates": {"c": {"m": {"value": 0.5, "limit": 0.1,
                                       "verdict": "FAIL"}}}}
    text = report.summary_table(collected)
    assert "| c | m | 0.5 | 0.1 | FAIL |" in text
    assert "SELECT 1" in report.provenance_block(collected)

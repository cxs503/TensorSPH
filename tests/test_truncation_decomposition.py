"""Tests for the CASE-HYDROSTATIC static truncation decomposition.

The benchmark under test (``benchmarks/case_hydrostatic/truncation_decomposition
.py``) is a *measurement* module: it decomposes the deterministic ``t = 0`` field
of CASE-HYDROSTATIC by band class and fits the observed order of each class.  The
tests therefore pin three things that the numbers depend on and that a refactor
could silently break:

1. the banding is the canonical one (``vv.hydrostatic.band_table``), not a
   second, slightly different definition of "band";
2. the ``t = 0`` field is the case's own field (``case_hydrostatic.
   run_initial_state``), so the decomposition cannot drift into measuring a
   different state;
3. the class labels and the order fit follow their declared rules on synthetic
   inputs whose answer is known by construction.

The physics-level assertions use a deliberately tiny tank (40 particles) so the
suite stays fast; the production ladder is the benchmark CLI, not a unit test.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tensorsph.case_hydrostatic import (RHO0, HydrostaticGeometryError,
                                       pressure_profile, run_initial_state,
                                       support_width)
from tensorsph.vv.hydrostatic import HydrostaticConfig, band_table

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "case_hydrostatic" / "truncation_decomposition.py"
#: The shipped record of the production ladder (six resolutions + c0 sweep).
SHIPPED = REPO_ROOT / "docs" / "case_hydrostatic_truncation_decomposition.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("truncation_decomposition", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


td = _load_module()

#: Tiny but declared-geometry-legal tank: 8 columns x 7 rows at dx = 0.05 m.
#: The depth must leave one band fully outside both support shells
#: (``2h = 0.13 m`` at this dx) or the interior class is empty.
SMALL = dict(dx=0.05, tank_width=0.4, depth=0.35, bands=5)
#: The same tank restricted to the arguments of ``static_field`` (no banding).
FIELD = dict(dx=SMALL["dx"], tank_width=SMALL["tank_width"], depth=SMALL["depth"])


def test_band_mean_reproduces_the_canonical_band_table():
    pos, _, rho, pressure, solver = td.static_field(**FIELD)
    unit_mass = torch.ones_like(pressure)
    cfg = HydrostaticConfig(dx=SMALL["dx"], tank_width=SMALL["tank_width"],
                            depth=SMALL["depth"], bands=SMALL["bands"])
    canonical = band_table(pos, pressure, unit_mass, cfg)
    mine = td.band_mean(pos, pressure, unit_mass, SMALL["depth"], SMALL["bands"])
    assert [r["band"] for r in mine] == [r["band"] for r in canonical]
    for a, b in zip(mine, canonical):
        assert a["y"] == pytest.approx(b["y"], rel=1e-12, abs=1e-12)
        assert a["value"] == pytest.approx(b["pressure"], rel=1e-12, abs=1e-9)
        assert a["particles"] == b["particles"]


def test_static_field_is_the_case_field_not_a_lookalike():
    """The decomposition must measure ``run_initial_state``, not a variant."""
    pos, mass, rho, pressure, solver = td.static_field(**FIELD)
    case = run_initial_state(dx=SMALL["dx"], tank_width=SMALL["tank_width"],
                             depth=SMALL["depth"], bands=SMALL["bands"])
    mine = td.band_mean(pos, pressure, mass, SMALL["depth"], SMALL["bands"])
    for row, case_row in zip(mine, case["profile"]):
        assert row["band"] == case_row["band"]
        assert row["value"] == pytest.approx(case_row["pressure"], rel=1e-10)
    assert [r["y"] for r in mine] == pytest.approx(
        [r["y"] for r in case["profile"]], rel=1e-12)


def test_class_labels_follow_their_declared_rules():
    support = support_width(0.02)          # 0.052 m
    depth = 0.4
    # floor: the band's lower edge sits inside the wall support shell
    assert td.classify(0.0, 0.05, depth, support) == "floor"
    assert td.classify(0.06, 0.11, depth, support) == "interior"
    # free surface: the upper edge enters the surface support shell
    assert td.classify(0.30, 0.35, depth, support) == "free_surface"
    # a band thick enough to touch both shells is labelled floor-first: the
    # wall verdict is the more conservative one and is reported first.
    assert td.classify(0.0, 0.39, depth, support) == "floor"


def test_classes_partition_the_bands_without_overlap():
    report = td.diagnose(**SMALL)
    labelled = [b["band"] for b in report["bands"]]
    for name in td.CLASSES:
        assert set(report["summary"][name]["bands"]) <= set(labelled)
    union = []
    for name in td.CLASSES:
        union += report["summary"][name]["bands"]
    assert sorted(union) == sorted(labelled)
    assert len(union) == len(set(union))


def test_diagnose_records_both_fields_and_the_tait_mapping():
    report = td.diagnose(**SMALL)
    assert report["config"]["fluid_particles"] == 56
    for row in report["bands"]:
        assert row["rho"] > 0.0
        assert row["rho_ref"] > 0.0
        assert row["pressure_ref"] > 0.0
        assert row["pressure"] == pytest.approx(
            row["pressure_ref"]
            * (1.0 + row["pressure_error_pct"] / 100.0), rel=1e-12)
        # The exact Tait mapping of the measured density ratio must reproduce
        # the recorded amplification, i.e. the two recorded fields are the same
        # state seen through the equation of state.
        assert row["pressure_error_from_density_pct"] == pytest.approx(
            row["tait_amplification"] * row["density_deficit_pct"], rel=1e-9)
    interior = report["summary"]["interior"]
    assert interior["bands"], "the tiny tank must keep at least one interior band"
    assert interior["max_abs_pressure_error_pct"] > 0.0


def test_post_processing_residual_is_small_away_from_the_free_surface():
    """Interior pressure is Tait(density); only the surface band is re-worked."""
    report = td.diagnose(**SMALL)
    for row in report["bands"]:
        if row["class"] != "free_surface":
            assert abs(row["post_processing_pct"]) < 0.05 * abs(row["pressure_error_pct"]) + 0.1


def test_geometry_gate_still_applies_to_the_decomposition():
    with pytest.raises(HydrostaticGeometryError):
        td.diagnose(dx=0.03, tank_width=0.4, depth=SMALL["depth"],
                    bands=SMALL["bands"])


def test_order_fit_recovers_a_known_power_law():
    points = [(0.04, 2.0), (0.02, 1.0), (0.01, 0.5)]
    fit = td.fit_order(points)
    assert fit["order"] == pytest.approx(1.0, abs=1e-12)
    assert fit["coefficient"] == pytest.approx(50.0, rel=1e-12)
    quad = [(0.04, 4.0), (0.02, 1.0), (0.01, 0.25)]
    assert td.fit_order(quad)["order"] == pytest.approx(2.0, abs=1e-12)


def test_order_fit_refuses_to_invent_an_order():
    assert td.fit_order([(0.02, 1.0)])["order"] is None
    assert td.fit_order([(0.02, 0.0), (0.01, 1.0)])["order"] is None


def test_interior_error_is_first_order_on_the_small_tank():
    """A two-point ladder: interior decays ~dx, the floor band does not decay
    with the same coefficient (it is the wall shell, not the quadrature)."""
    report = td.decompose(dxs=(0.05, 0.025), **{k: v for k, v in SMALL.items()
                                                if k != "dx"})
    interior = report["ladder"]["interior"]
    assert interior["order"]["order"] == pytest.approx(1.0, abs=0.25)
    floor = report["ladder"]["floor"]["order"]["order"]
    assert floor is None or floor > 0.0
    # the floor class is the worse one at every recorded resolution
    for rec in report["records"]:
        assert (rec["summary"]["floor"]["max_abs_pressure_error_pct"]
                > rec["summary"]["interior"]["max_abs_pressure_error_pct"])


def test_sound_speed_sweep_moves_deficit_and_amplification_oppositely():
    report = td.decompose(dxs=(0.05,), sound_speed=20.0, sound_speeds=(10.0,),
                          **{k: v for k, v in SMALL.items() if k != "dx"})
    sweep = report["sound_speed_sweep"]
    assert [row["sound_speed"] for row in sweep] == [20.0, 10.0]
    slow, base = sweep[1], sweep[0]
    # softer EOS (smaller B) => larger density deficit, smaller amplification
    assert abs(slow["interior_density_deficit_pct"]) > abs(
        base["interior_density_deficit_pct"])
    assert slow["interior_mean_tait_amplification"] < base["interior_mean_tait_amplification"]
    # ... while the *response* error is what survives the change of EOS: it is
    # the quantity the pressure gate actually sees, so it must not move like the
    # deficit does (the tiny tank is coarse, hence the factor-3 tolerance).
    ratio_deficit = (abs(slow["interior_density_deficit_pct"])
                     / abs(base["interior_density_deficit_pct"]))
    ratio_response = (abs(slow["interior_density_response_error_pct"])
                      / abs(base["interior_density_response_error_pct"]))
    assert ratio_deficit > 2.0
    assert ratio_response < 3.0


def test_density_response_error_is_recorded_per_band():
    report = td.diagnose(**SMALL)
    interior = [b for b in report["bands"] if b["class"] == "interior"]
    assert interior
    for row in interior:
        # the response is the error of (rho - rho0), not of rho itself
        expected = (row["rho"] - RHO0) / (row["rho_ref"] - RHO0) - 1.0
        assert row["density_response_error_pct"] == pytest.approx(expected * 100.0, rel=1e-9)


def test_markdown_and_json_are_written_from_the_record(tmp_path):
    report = td.decompose(dxs=(0.05, 0.025), sound_speed=20.0, sound_speeds=(10.0,),
                          **{k: v for k, v in SMALL.items() if k != "dx"})
    md = td.markdown(report)
    # every recorded resolution appears verbatim in the table
    for rec in report["records"]:
        assert f"| {rec['config']['dx']:g} |" in md
    assert "声速扫描" in md
    assert report["records"][0]["config"]["bulk_modulus"] > 0.0

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    rc = td.main(["--dxs", "0.05", "--tank-width", str(SMALL["tank_width"]),
                  "--depth", str(SMALL["depth"]), "--bands", str(SMALL["bands"]),
                  "--json-out", str(json_path), "--md-out", str(md_path)])
    assert rc == 0
    written = json.loads(json_path.read_text())
    assert written["case_id"] == "CASE-HYDROSTATIC"
    assert md_path.read_text().startswith("# CASE-HYDROSTATIC")


def test_cli_rejects_an_unrepresentable_tank(tmp_path, capsys):
    rc = td.main(["--dxs", "0.03", "--depth", str(SMALL["depth"]),
                  "--tank-width", str(SMALL["tank_width"]),
                  "--bands", str(SMALL["bands"])])
    assert rc == 2
    assert "geometry rejected" in capsys.readouterr().err


def test_decomposition_gate_agrees_with_the_case_gate_on_the_tiny_tank():
    """Cross-check: the case's own gate reads the same pressure band means."""
    pos, mass, _, pressure, _ = td.static_field(**FIELD)
    rows = pressure_profile(pos, pressure, mass, depth=SMALL["depth"],
                            dx=SMALL["dx"], bands=SMALL["bands"])
    gated = [r for r in rows if r["gated"]]
    assert gated
    report = td.diagnose(**SMALL)
    by_band = {b["band"]: b for b in report["bands"]}
    for row in gated:
        assert by_band[row["band"]]["pressure"] == pytest.approx(row["pressure"], rel=1e-10)

def test_shipped_document_is_the_rendering_of_the_shipped_record():
    """The committed .md is generated from the committed .json, not edited."""
    if not SHIPPED.is_file():       # pragma: no cover - shallow checkouts
        pytest.skip("shipped truncation decomposition record is not in the tree")
    report = json.loads(SHIPPED.read_text())
    document = SHIPPED.with_suffix(".md").read_text()
    assert document == td.markdown(report)
    assert SHIPPED.with_suffix(".png").is_file()
    assert len(report["records"]) >= 3
    assert report["dynamic_gate"]["source"].endswith("window_8s.json")
    # the ladder really is first order on the interior class: the finding the
    # document states must come from this record, not from prose.
    assert report["ladder"]["interior"]["order"]["order"] == pytest.approx(1.0, abs=0.1)
    assert report["ladder"]["interior_density_response"]["order"]["order"] == pytest.approx(
        1.0, abs=0.1)

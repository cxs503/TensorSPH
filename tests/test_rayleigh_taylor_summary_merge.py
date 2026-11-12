"""Guards for the Rayleigh–Taylor campaign summary (audit m16318, findings 5–6).

The runner used to rewrite ``runs/summary.json`` from the current invocation
only, so a sharded campaign left one row of eight, and it stored a failed fit as
``NaN`` — a token that is not legal JSON.  These tests pin the fixed behaviour:

* merge by ``tag``: a second invocation cannot delete a first one's rows;
* no usable fit -> ``status="no_fit"`` and ``null`` numbers, never ``NaN``;
* the summary on disk parses under a strict parser and covers every case file;
* aggregate/reference tables in ``runs/`` are never mistaken for case records.

They run on synthetic records in ``tmp_path`` and finish in milliseconds; the
one test that touches the repository artifact only reads it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "rayleigh_taylor"
sys.path.insert(0, str(BENCH))

import summary_io  # noqa: E402

RUNS = BENCH / "runs"


def _record(tag: str, sigma: float, n_fit: int = 100, **extra) -> dict:
    record = {
        "tag": tag,
        "fit": {"sigma": sigma, "r2": 0.999, "n_fit": n_fit,
                "window": [0.02, 0.1]},
        "reference": {"sigma_ideal": 5.551488},
        "sigma_over_ideal": sigma / 5.551488,
        "sigma_over_finite_depth": sigma / 5.551488,
        "sigma_over_viscous": sigma / 5.551488,
        "max_amplitude": 0.04,
        "derived": {"note": "synthetic"},
        "wall_seconds": 1.0,
    }
    record.update(extra)
    return record


def _strict(text: str):
    def boom(token):  # pragma: no cover - only fires on the bug we are pinning
        raise ValueError(f"non-finite JSON token: {token}")

    return json.loads(text, parse_constant=boom)


# --------------------------------------------------------------------------- #
# merge semantics
# --------------------------------------------------------------------------- #
def test_second_invocation_does_not_drop_first_rows(tmp_path):
    """The defect: shard 2 used to overwrite shard 1's summary."""
    summary_io.write_summary(tmp_path, new_rows=[_record("dxA", 4.0)])
    summary_io.write_summary(tmp_path, new_rows=[_record("dxB", 3.0)])

    rows = _strict((tmp_path / "summary.json").read_text())
    assert [row["tag"] for row in rows] == ["dxA", "dxB"]
    assert rows[0]["status"] == rows[1]["status"] == "ok"


def test_rerun_same_tag_replaces_instead_of_duplicating(tmp_path):
    summary_io.write_summary(tmp_path, new_rows=[_record("dxA", 4.0)])
    summary_io.write_summary(tmp_path, new_rows=[_record("dxA", 4.2)])

    rows = _strict((tmp_path / "summary.json").read_text())
    assert len(rows) == 1
    assert rows[0]["sigma_over_ideal"] == pytest.approx(4.2 / 5.551488)


def test_rows_are_built_from_case_files_on_disk(tmp_path):
    """``--aggregate-only`` path: no ``new_rows``, everything from artifacts."""
    (tmp_path / "dxA.json").write_text(json.dumps(_record("dxA", 4.0)))
    (tmp_path / "dxB.json").write_text(json.dumps(_record("dxB", 3.0)))

    rows = summary_io.write_summary(tmp_path)
    assert [row["tag"] for row in rows] == ["dxA", "dxB"]
    assert all(row["status"] == "ok" for row in rows)


def test_summary_is_stable_under_repeated_aggregation(tmp_path):
    (tmp_path / "dxA.json").write_text(json.dumps(_record("dxA", 4.0)))
    first = summary_io.write_summary(tmp_path)
    text = (tmp_path / "summary.json").read_text()
    summary_io.write_summary(tmp_path)
    assert (tmp_path / "summary.json").read_text() == text
    assert summary_io.write_summary(tmp_path) == first


# --------------------------------------------------------------------------- #
# NaN handling
# --------------------------------------------------------------------------- #
def test_failed_fit_is_recorded_as_no_fit_not_nan(tmp_path):
    bad = _record("dxCoarse", float("nan"), n_fit=0,
                  sigma_over_ideal=float("nan"))
    summary_io.write_summary(tmp_path, new_rows=[bad])

    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    row = _strict(text)[0]
    assert row["status"] == "no_fit"
    assert row["sigma_over_ideal"] is None
    assert row["fit"]["sigma"] is None
    assert row["fit"]["n_fit"] == 0


def test_nested_non_finite_values_are_nulled():
    record = _record("dxA", 4.0)
    record["derived"] = {"ratio": [1.0, float("inf"), {"x": float("-inf")}]}
    row = summary_io.sanitize_row(record)

    assert row["derived"]["ratio"][0] == 1.0
    assert row["derived"]["ratio"][1] is None
    assert row["derived"]["ratio"][2]["x"] is None
    assert "Infinity" not in summary_io.dumps_strict(row)


def test_dumps_strict_refuses_non_finite():
    with pytest.raises(ValueError):
        summary_io.dumps_strict({"sigma": math.nan})


# --------------------------------------------------------------------------- #
# aggregate tables are not case records
# --------------------------------------------------------------------------- #
def test_aggregate_tables_are_not_treated_as_case_files(tmp_path):
    """``table/reference/damping`` carry no case record; ``summary`` is a base."""
    (tmp_path / "dxA.json").write_text(json.dumps(_record("dxA", 4.0)))
    (tmp_path / "table.json").write_text(json.dumps(
        {"tag": "not-a-case", "fit": {"sigma": 1.0}}))
    (tmp_path / "reference.json").write_text(json.dumps({"sigma": 5.551488}))
    (tmp_path / "damping.json").write_text(json.dumps({"tag": "x"}))

    rows = summary_io.load_case_rows(tmp_path)
    assert list(rows) == ["dxA"]


def test_previous_summary_is_a_base_but_case_files_win(tmp_path):
    """A row whose case file exists is rebuilt from the file, not the summary."""
    (tmp_path / "summary.json").write_text(json.dumps(
        [_record("dxA", 1.0), _record("archived", 2.0)]))
    (tmp_path / "dxA.json").write_text(json.dumps(_record("dxA", 4.0)))

    rows = summary_io.write_summary(tmp_path)
    by_tag = {row["tag"]: row for row in rows}
    # kept alive although its case file was archived
    assert by_tag["archived"]["sigma_over_ideal"] == pytest.approx(2.0 / 5.551488)
    # rebuilt from the file, not from the stale summary
    assert by_tag["dxA"]["sigma_over_ideal"] == pytest.approx(4.0 / 5.551488)


def test_current_invocation_wins_over_both(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps([_record("dxA", 1.0)]))
    (tmp_path / "dxA.json").write_text(json.dumps(_record("dxA", 2.0)))

    rows = summary_io.write_summary(tmp_path, new_rows=[_record("dxA", 4.0)])
    assert len(rows) == 1
    assert rows[0]["sigma_over_ideal"] == pytest.approx(4.0 / 5.551488)


def test_non_case_payloads_are_skipped(tmp_path):
    (tmp_path / "dxA.json").write_text(json.dumps(_record("dxA", 4.0)))
    (tmp_path / "notes.json").write_text(json.dumps({"tag": "no fit key"}))
    (tmp_path / "broken.json").write_text("{not json")

    assert [row["tag"] for row in summary_io.write_summary(tmp_path)] == ["dxA"]


# --------------------------------------------------------------------------- #
# repository artifact
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (RUNS / "summary.json").exists(),
                    reason="campaign summary not present in this checkout")
def test_repository_summary_is_strict_json_and_covers_all_case_files():
    rows = _strict((RUNS / "summary.json").read_text())
    on_disk = summary_io.load_case_rows(RUNS)

    assert len(rows) == len(on_disk) >= 8
    assert set(row["tag"] for row in rows) == set(on_disk)
    for row in rows:
        assert row["status"] in ("ok", "no_fit")
        if row["status"] == "no_fit":
            assert row["sigma_over_ideal"] is None
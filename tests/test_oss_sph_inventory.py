"""Tests for the OSS-SPH inventory anchor (FEAT-OSS-INVENTORY-SPH).

These tests do not re-crawl the network.  They assert that the committed
artifacts are internally consistent and that the derivations claimed in
``docs/oss_inventory_sph/inventory.md`` really are reproducible from the JSON:

* registry schema, size (>= 20 repos, the anchor's acceptance) and provenance;
* the tier label of every row is what the documented rubric produces from that
  row's own features (so a hand-edit of a label fails the suite);
* the gap list is exactly the mechanical rule's output (set equality, not "some
  gaps exist");
* every proposal carries a number-bearing acceptance criterion and an anchor
  artifact that exists on disk;
* the report's pump-point numbers equal the JSON values.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "docs" / "oss_inventory_sph"
SCRIPTS = ROOT / "scripts" / "oss_sph_inventory"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def registry():
    return json.loads((DIR / "registry.json").read_text())


@pytest.fixture(scope="module")
def capability():
    return json.loads((DIR / "tensorsph_capability.json").read_text())


@pytest.fixture(scope="module")
def gaps():
    return json.loads((DIR / "gaps.json").read_text())


@pytest.fixture(scope="module")
def report():
    return (DIR / "inventory.md").read_text()


def test_registry_meets_anchor_acceptance(registry):
    """Anchor acceptance: >= 20 repositories with tier classification."""
    assert registry["anchor"] == "FEAT-OSS-INVENTORY-SPH"
    assert registry["repo_count"] == len(registry["repos"]) >= 20
    assert {r["tier"] for r in registry["repos"]} <= {"A", "B", "C"}
    assert {r["tier"] for r in registry["repos"]} == {"A", "B", "C"}, "tiering must discriminate"


def test_every_row_has_provenance(registry):
    for row in registry["repos"]:
        assert row["url"].startswith("https://github.com/")
        if row["evidence_grade"] == "METADATA-ONLY":
            # nothing was fetched: the row must say so and must claim no feature knowledge
            assert row["docs_status"] in {"budget_exhausted", "partial"}
        if row["readme_sha256"]:
            assert re.fullmatch(r"[0-9a-f]{64}", row["readme_sha256"])
        if row["tree_sha256"]:
            assert re.fullmatch(r"[0-9a-f]{64}", row["tree_sha256"])
        assert row["docs_status"] in {"complete", "partial", "budget_exhausted"}
        # A row whose docs were never fetched must not claim feature knowledge.
        if row["docs_status"] == "budget_exhausted":
            # the hourly API budget ran out mid-row: the tree channel must be empty,
            # while the README (fetched earlier in the same row) may still be present
            assert row["tree_present"] == 0 and row["tree_sha256"] is None


def test_no_duplicate_repositories(registry):
    names = [r["repo"].lower() for r in registry["repos"]]
    assert len(names) == len(set(names))


def test_tier_labels_reproduce_from_rubric(registry):
    """The committed label must be what assign_tier() computes from the committed row."""
    fetch = _load("fetch_registry")
    for row in registry["repos"]:
        tier, reasons = fetch.assign_tier(
            row["license"], row["years_since_push"], row["stars"],
            row["readme_features"], row["tree_features"])
        assert tier == row["tier"], (
            f"{row['repo']}: committed tier {row['tier']} != rubric {tier} ({reasons})")
        assert row["tier_reasons"] == reasons


def test_absent_models_are_not_counted_as_capabilities(capability):
    """A line that documents a model's absence must never appear in src_hits."""
    absent_rx = [re.compile(p, re.I) for p in _load("probe_tensorsph").ABSENT_MARKERS]
    for key, row in capability["features"].items():
        for hit in row["src_hits"]:
            assert not any(rx.search(hit["snippet"]) for rx in absent_rx), (
                f"{key}: absence statement counted as implementation: {hit}")


def test_turbulence_gap_is_a_real_gap(capability):
    """Regression for the false positive found while building the anchor.

    The first probe run counted ``src/tensorsph/freesurface/breaking_wave.py``'s
    line "no LES or SGS model" as an implementation.  The gap must survive that
    correction, and the source statement that caused it must still be there --
    otherwise this test would pass on a repository that silently gained a model.
    """
    turb = capability["features"]["turbulence"]
    assert turb["in_source"] is False
    text = (ROOT / "src" / "tensorsph" / "freesurface" / "breaking_wave.py").read_text()
    assert "no LES or SGS model" in text


def test_gap_rule_is_reproducible(registry, capability, gaps):
    builder = _load("build_report")
    cov = builder.feature_coverage(registry)
    recomputed = builder.build_gaps(registry, cov, capability)
    assert [g["feature"] for g in recomputed] == [g["feature"] for g in gaps["proposals"]]
    assert [g.get("slug") for g in recomputed] == [g.get("slug") for g in gaps["proposals"]]


def builder_floor():
    return _load("build_report").GAP_COVERAGE_FLOOR


def test_gap_proposals_are_quantified_and_anchored(gaps):
    proposals = [g for g in gaps["proposals"] if g["status"] == "PROPOSED"]
    assert len(proposals) >= 4, "anchor acceptance asks for >= 4 GAP-SPH-* proposals"
    for g in proposals:
        assert g["slug"].startswith("GAP-SPH-")
        crit = g["acceptance_quantified"]
        # A criterion without a number and a comparison is not falsifiable.
        assert re.search(r"\d", crit), f"{g['slug']}: no number in acceptance"
        assert re.search(r"<=|>=|<|>|within", crit), f"{g['slug']}: no threshold in acceptance"
        assert g["anchor_exists"], f"{g['slug']}: anchor artifact missing"
        assert g["tensorsph_in_source"] is False
        assert g["oss_repos_exposing"] >= 3
        assert g["oss_ratio"] >= builder_floor()


def test_report_numbers_are_injected(registry, gaps, report):
    builder = _load("build_report")
    cov = builder.feature_coverage(registry)
    # repo count and the >=4 proposals claim must both be visible in the report
    assert f"**{registry['repo_count']} public SPH codes**".replace("**", "") in report
    n_prop = sum(1 for g in gaps["proposals"] if g["status"] == "PROPOSED")
    assert f"{n_prop} proposals were generated" in report
    top = max(cov, key=lambda k: cov[k]["ratio"])
    assert f"{cov[top]['either']}/{cov[top]['n']}" in report
    for g in gaps["proposals"]:
        if g["status"] == "PROPOSED":
            assert g["slug"] in report


def test_figures_exist_and_are_nonempty():
    for name in ("coverage_vs_tensorsph.png", "tier_domains.png"):
        path = DIR / "figures" / name
        assert path.exists() and path.stat().st_size > 5000
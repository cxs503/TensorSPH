"""Tests for the artefact digest / restore path (m21953).

The compute gate's ``fetch`` returns the *staged* tree, not the tree the job
wrote (m21005 §③), so a grid produced on the cluster reaches the writing side as
text on the job's standard output.  This module is that text path, and the point
of the tests is that the path is **lossless for the ledger**: a restored
artefact must yield bit-identical rows out of
``footprint_ledger.grid_record``, otherwise the numbers transcribed from a run
could differ from the numbers the solver computed without anyone noticing.

Tests:

* round-trip fidelity against the committed artefacts (both grids present);
* the restored file still carries what the ledger reads, and nothing it needs is
  dropped silently (negative control: strip the footprint block -> ledger
  refuses, so a truncated digest cannot be laundered into a "pass");
* the provenance hash really is the hash of the source file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "case_sph_wedge_entry_2d"
FP = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_footprint"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

import artifact_digest as ad  # noqa: E402
import footprint_ledger as fl  # noqa: E402

ARTEFACTS = sorted(FP.glob("ensemble_dx*mm_c0v20.json"))


def test_committed_artefacts_present():
    assert ARTEFACTS, "no committed footprint artefacts to test the digest against"


@pytest.mark.parametrize("src", ARTEFACTS, ids=[p.stem for p in ARTEFACTS])
def test_roundtrip_preserves_every_ledger_field(src: Path, tmp_path: Path):
    """Digest -> restore -> ledger row must equal the row from the original."""
    art = json.loads(src.read_text())
    dig = ad.digest(art, src)
    dst = tmp_path / src.name
    ad.restore(dig, dst)

    point = fl.eb.wagner_peak_over_q(0.5235987755982988)   # beta = 30 deg
    tag = src.stem.split("_")[1]                            # e.g. dx2mm
    want = fl.grid_record(tag, src.parent, point)
    got = fl.grid_record(tag, tmp_path, point)

    for key in ("dx_m", "phi_m", "footprint_cells", "delta_tau",
                "obs_cp", "synth_cp", "uncertainty", "residual", "judgeable"):
        assert got[key] == want[key], f"{tag}: {key} changed through the digest"
    assert got["bootstrap"] == want["bootstrap"]
    assert got["all_footprints"] == want["all_footprints"]


@pytest.mark.parametrize("src", ARTEFACTS, ids=[p.stem for p in ARTEFACTS])
def test_digest_is_small_but_keeps_the_reading(src: Path):
    dig = ad.digest(json.loads(src.read_text()), src)
    row = dig["footprint_peak"]["rows"]
    assert row and all("cp_peak" in r for r in row)
    assert dig["footprint_peak"]["bootstrap"]
    assert dig["config"]["dx"] and dig["t_end"] and dig["C"]
    # the digest is the stdout read-out, so it must be materially smaller than
    # the artefact it describes (otherwise the transport is pointless)
    assert len(json.dumps(dig)) <= len(src.read_text())


def test_provenance_hash_is_the_source_hash(tmp_path: Path):
    src = ARTEFACTS[0]
    dig = ad.digest(json.loads(src.read_text()), src)
    assert dig["provenance"]["digest_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()


def test_restored_artefact_records_its_provenance(tmp_path: Path):
    src = ARTEFACTS[0]
    dst = tmp_path / src.name
    ad.restore(ad.digest(json.loads(src.read_text()), src), dst)
    back = json.loads(dst.read_text())
    assert back["restored_from_digest"]["digest_sha256"]
    assert back["restored_from_digest"]["source"] == str(src)


def test_truncated_digest_is_refused_not_laundered(tmp_path: Path):
    """Negative control: a digest without the footprint block must not score."""
    src = ARTEFACTS[0]
    dig = ad.digest(json.loads(src.read_text()), src)
    del dig["footprint_peak"]
    dst = tmp_path / src.name
    ad.restore(dig, dst)
    point = fl.eb.wagner_peak_over_q(0.5235987755982988)
    with pytest.raises(SystemExit):
        fl.grid_record(src.stem.split("_")[1], tmp_path, point)
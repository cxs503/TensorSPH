"""Contract tests for :mod:`tensorsph.registry` (capability registry, m25063).

Pins the properties the platform dashboard depends on (dual-pilot spec,
m23186/m23215):

* the static AST dump is a valid JSON contract (``source_commit`` +
  ``capabilities``) and needs neither the package import nor torch;
* import-time registration and static scanning agree field for field;
* the maturity discipline is enforced in-process (a false ``mature`` cannot
  even be registered, mirroring the server's 422 gate);
* every ``mature`` row rests on a measured error band (< 3% for the key
  quantity) whose evidence citation resolves inside the repository;
* every ``cli:`` endpoint names a ``--case`` that ``python -m tensorsph``
  actually accepts.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src" / "tensorsph"
sys.path.insert(0, str(ROOT / "src"))

from tensorsph import registry as reg  # noqa: E402

REQUIRED_IDS = (
    "sph.solver.wcsp",
    "sph.case.hydrostatic",
    "sph.case.poiseuille_duct",
    "sph.case.stokes_second",
    "sph.case.dam_break",
    "sph.case.dam_break_3d",
    "sph.case.sloshing",
    "sph.case.wave_flume",
    "sph.case.slamming",
    "sph.case.gresho",
    "sph.case.taylor_green",
    "sph.case.rayleigh_taylor",
    "sph.case.ritter",
    "sph.case.capillary_oscillation",
    "sph.case.lake_at_rest",
    "sph.case.kleefsman_obstacle",
    "sph.fsi.turek_hron",
    "sph.ice.splash",
    "sph.compressible.sod",
)


@pytest.fixture(scope="module")
def static_dump():
    caps, problems = reg.scan_static(PKG_ROOT, repo_root=ROOT)
    return caps, problems


def test_static_scan_is_clean_and_populated(static_dump):
    caps, problems = static_dump
    errors = [p for p in problems if p["severity"] == "error"]
    assert errors == [], errors
    assert len(caps) >= 18, f"only {len(caps)} capabilities registered"
    ids = [c.id for c in caps]
    assert len(ids) == len(set(ids)), "duplicate capability ids in the static scan"
    for required in REQUIRED_IDS:
        assert required in ids, f"missing capability {required}"


def test_static_mode_runs_without_importing_the_package(tmp_path):
    """``python registry.py --static`` must work with torch unavailable."""
    blocker = tmp_path / "torch"
    blocker.mkdir()
    (blocker / "__init__.py").write_text(
        "raise ImportError('torch must not be imported in static mode')\n",
        encoding="utf-8")
    env = {"PYTHONPATH": str(blocker), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, str(PKG_ROOT / "registry.py"), "--static"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload["mode"] == "static"
    assert payload["source_commit"]
    assert payload["count"] == len(payload["capabilities"]) >= 18
    assert "torch" not in proc.stderr


def test_import_mode_matches_static_mode(static_dump):
    caps, _ = static_dump
    runtime, meta = reg.import_registry(pkg_root=PKG_ROOT)
    assert meta["failed"] == [], meta["failed"]
    static = {c.id: c.to_payload() for c in caps}
    live = {c.id: c.to_payload() for c in runtime.capabilities()}
    assert sorted(static) == sorted(live), (
        f"static-only={sorted(set(static) - set(live))} "
        f"runtime-only={sorted(set(live) - set(static))}")
    for cid in static:
        assert static[cid] == live[cid], f"field mismatch for {cid}"


def test_mature_requires_verified_and_wired():
    with pytest.raises(reg.MaturityDisciplineError):
        reg.Registry().register(reg.Capability(
            id="probe.bad", kind="solver", maturity="mature",
            module_path="tensorsph.droplet_impact", entry="run",
            wired=False, verified=True))


def test_planned_requires_feature_ledger_anchor():
    with pytest.raises(reg.MaturityDisciplineError):
        reg.Registry().register(reg.Capability(
            id="probe.roadmap", maturity="planned",
            module_path="tensorsph.droplet_impact",
            evidence={"grade": "NONE"}))
    cap = reg.Registry().register(reg.Capability(
        id="probe.roadmap", maturity="planned",
        module_path="tensorsph.droplet_impact",
        evidence={"grade": "NONE", "feature_ledger": "GAP-PROBE"}))
    assert cap.maturity == "planned"


def test_duplicate_ids_are_rejected():
    table = reg.Registry()
    cap = reg.Capability(id="probe.dup", maturity="developing",
                         module_path="tensorsph.droplet_impact")
    table.register(cap)
    with pytest.raises(reg.DuplicateCapabilityError):
        table.register(cap)


def test_endpoint_must_be_a_wired_callable_contract():
    with pytest.raises(reg.MaturityDisciplineError):
        reg.Registry().register(reg.Capability(
            id="probe.endpoint", maturity="developing",
            module_path="tensorsph.droplet_impact",
            wired=False, endpoint="api:tensorsph.droplet_impact.run"))
    with pytest.raises(reg.MaturityDisciplineError):
        reg.Registry().register(reg.Capability(
            id="probe.endpoint", maturity="developing",
            module_path="tensorsph.droplet_impact",
            wired=True, endpoint="run"))


def test_mature_entries_carry_a_reproducible_error_band(static_dump):
    caps, _ = static_dump
    for cap in caps:
        if cap.maturity != "mature":
            continue
        assert cap.wired and cap.verified and cap.tested, cap.id
        assert cap.evidence.get("grade") in reg.VERIFYING_GRADES, cap.id
        err = float(cap.evidence["max_error_pct"])
        assert 0.0 <= err < 3.0, (
            f"{cap.id}: key-quantity error band {err}% is not < 3%")
        # the evidence note must cite a file that exists in the repository
        cited = re.findall(r"[\w./-]+\.(?:md|json|py)", str(cap.evidence.get("note", "")))
        assert cited, f"{cap.id}: evidence note cites no repository file"
        for c in cited:
            assert (ROOT / c).is_file(), f"{cap.id}: citation {c} missing"


def test_cli_endpoints_match_the_main_entry_choices(static_dump):
    """Every ``cli:`` endpoint must name a --case the CLI actually accepts."""
    main_src = (ROOT / "src" / "tensorsph" / "__main__.py").read_text(encoding="utf-8")
    m = re.search(r'add_argument\("--case".*?choices=\(([^)]*)\)', main_src, flags=re.S)
    assert m, "could not find the --case choices tuple in __main__.py"
    choices = set(re.findall(r'"(\w+)"', m.group(1)))
    caps, _ = static_dump
    for cap in caps:
        if not cap.endpoint or not cap.endpoint.startswith("cli:"):
            continue
        if "--case" not in cap.endpoint:
            continue  # bare module entry point (python -m tensorsph)
        case = cap.endpoint.split("--case", 1)[-1].strip()
        assert case in choices, (
            f"{cap.id}: endpoint {cap.endpoint!r} but --case {case!r} "
            f"not in {sorted(choices)}")

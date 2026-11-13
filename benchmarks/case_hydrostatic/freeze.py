"""Freeze / verify the *shippable* text channel of ``CASE-HYDROSTATIC``.

Why this tool exists
--------------------
The case bundle produced by ``run.py --aggregate --figure`` lives in
``runs/case_hydrostatic`` (not pushed: the delivery rule keeps ``runs/`` out of
git) and is mirrored to ``outputs/case_hydrostatic`` (pushed, but classed as
*artifact* by the merge governor, which caps artifact payloads at 200 KB per
PR -- the mirrored bundle is ~370 KB once the PNGs are counted, so a PR that
carries it is held and never lands on ``main``).  The consequence on ``main``
was: ``git ls-files outputs/`` empty -> the shipped-evidence group of
``tests/test_case_hydrostatic.py`` errors out on *every* fresh checkout, i.e.
the case's 1 % acceptance is not machine-decidable where it matters.

This tool closes that hole without weakening the check:

* it copies only the **text** payload of the bundle (``report.json`` + every
  ``runs/*.json``) into ``tests/data/case_hydrostatic`` -- a path that is
  neither ``runs/`` nor artifact-class, so it survives push and merge;
* the PNG figures are deliberately *not* shipped: they are re-creatable from
  the shipped numbers with ``run.py --aggregate --figure`` and are the only
  part of the bundle that the delivery rule drops;
* it writes ``MANIFEST.sha256`` (``sha256  bytes  relpath``, sorted) so the
  shipped numbers are tamper-evident, and ``channel.json`` recording exactly
  what was shipped, what was not, and why;
* ``--verify`` re-derives the shipped gate value from the local bundle and
  compares it with the shipped one, so a re-run of the case is a *check* on the
  frozen evidence rather than a replacement for it.

Usage
-----
    python benchmarks/case_hydrostatic/freeze.py            # freeze
    python benchmarks/case_hydrostatic/freeze.py --verify   # fresh run vs frozen
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "runs" / "case_hydrostatic"
SHIPPED = REPO / "tests" / "data" / "case_hydrostatic"
MANIFEST = "MANIFEST.sha256"
CHANNEL = "channel.json"
#: Text extensions only: ``.png`` (and friends) are artifact-class in the
#: delivery rule and must not be relied on for decidability.
TEXT_SUFFIXES = (".json",)
#: Files the case's shipped-evidence tests read.
REQUIRED = ("report.json",)


def _posix(path: Path) -> str:
    """Repo-relative when possible (``--from`` may point outside the tree)."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _rel_files(root: Path, suffixes=TEXT_SUFFIXES) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix in suffixes
                  and p.name not in (MANIFEST, CHANNEL))


def _manifest_lines(root: Path) -> list[str]:
    lines = []
    for path in _rel_files(root):
        data = path.read_bytes()
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {len(data)}  "
                     f"{path.relative_to(root).as_posix()}")
    return lines


def freeze(bundle: Path = BUNDLE, dest: Path = SHIPPED) -> dict:
    """Copy the text payload of ``bundle`` into ``dest`` + manifest + channel."""
    if not (bundle / "report.json").exists():
        raise SystemExit(
            f"freeze: no bundle report at {bundle / 'report.json'}; run "
            "`python benchmarks/case_hydrostatic/run.py --config all --static` "
            "then `--aggregate` first")
    report = json.loads((bundle / "report.json").read_text())
    dropped = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
                     if p.is_file() and p.suffix not in TEXT_SUFFIXES)
    dest.mkdir(parents=True, exist_ok=True)
    for stale in _rel_files(dest):
        stale.unlink()
    copied = []
    for path in _rel_files(bundle):
        rel = path.relative_to(bundle)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(rel.as_posix())
    lines = _manifest_lines(dest)
    (dest / MANIFEST).write_text("\n".join(lines) + "\n")
    channel = {
        "case_id": report["verdict"]["case_id"],
        "source_bundle": _posix(bundle),
        "gate_config": report["verdict"]["gate_config"],
        "gate_value_pct": report["verdict"]["gate_value_pct"],
        "gate_limit_pct": report["verdict"]["gate_limit_pct"],
        "gate_pass": report["verdict"]["gate_pass"],
        "files": copied,
        "dropped_not_shippable": dropped,
        "why": ("text payload only: *.png is artifact-class in the delivery "
                "rule (merge governor artifact cap, outputs/ + .png), and the "
                "shipped-evidence tests must be decidable on a fresh checkout"),
        "repro": report["repro"],
        "manifest": MANIFEST,
    }
    (dest / CHANNEL).write_text(json.dumps(channel, indent=1) + "\n")
    return {"dest": _posix(dest), "files": len(copied),
            "dropped": dropped,
            "gate_value_pct": channel["gate_value_pct"]}


def verify(bundle: Path = BUNDLE, shipped: Path = SHIPPED) -> dict:
    """Compare a freshly regenerated ``bundle`` with the frozen channel."""
    rows = []
    for rel in _rel_files(shipped):
        relstr = rel.relative_to(shipped).as_posix()
        fresh = bundle / relstr
        rec = {"file": relstr, "shipped": True, "fresh": fresh.exists()}
        if rec["fresh"]:
            rec["byte_identical"] = fresh.read_bytes() == rel.read_bytes()
        rows.append(rec)
    report = json.loads((shipped / "report.json").read_text())
    gate_config = report["verdict"]["gate_config"]
    fresh_report = bundle / "report.json"
    out = {"shipped_root": _posix(shipped),
           "bundle_root": _posix(bundle),
           "gate_config": gate_config,
           "gate_value_shipped_pct": report["verdict"]["gate_value_pct"],
           "gate_limit_pct": report["verdict"]["gate_limit_pct"],
           "files": rows}
    # Cheapest reproduction is the pre-declared single-config command
    # (`run.py --config window_8s`), which writes only the gate run file.
    fresh_gate_run = bundle / "runs" / f"{gate_config}.json"
    if fresh_report.exists():
        fresh = json.loads(fresh_report.read_text())
        out["gate_value_fresh_pct"] = fresh["verdict"]["gate_value_pct"]
        out["fresh_pass"] = bool(fresh["verdict"]["gate_pass"])
    elif fresh_gate_run.exists():
        fresh = json.loads(fresh_gate_run.read_text())
        out["fresh_source"] = _posix(fresh_gate_run)
        out["gate_value_fresh_pct"] = fresh["gate"]["max_band_error_pct"]
        out["fresh_pass"] = bool(fresh["gate"]["pass"])
    else:
        out["reproduced"] = False
        out["reason"] = "no fresh bundle to compare"
        return out
    out["delta_pp"] = abs(out["gate_value_fresh_pct"]
                          - out["gate_value_shipped_pct"])
    out["reproduced"] = bool(
        out["delta_pp"] < 0.05
        and out["fresh_pass"] == report["verdict"]["gate_pass"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="src", default=str(BUNDLE))
    parser.add_argument("--to", dest="dst", default=str(SHIPPED))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        out = verify(Path(args.src), Path(args.dst))
        print(json.dumps(out, indent=1))
        return 0 if out["reproduced"] else 1
    out = freeze(Path(args.src), Path(args.dst))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
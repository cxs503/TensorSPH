"""Does a refinement-trend claim compare like with like? (admissibility matching)

Why this module exists
----------------------
``benchmarks/method_selection/adjudicate_over_tolerance.py`` (m22787) banks the
record ``x_couette_viscous_time_top_h16`` (moving-wall shear error 17.08 %) as
``genuine_accuracy_failure`` with ``worsens_with_resolution: true``, and that
record is the **exclusion basis** of the published decision for the planar
Couette profile (``cites_this_record: true`` in
``docs/reports/method_selection_strict_bound_m21537.json``).

The trend was computed as "steady end of the sweep" per rung, where the steady
end is *the last admissible point, or the last point if none is admissible*
(``summarise()`` in ``benchmarks/couette/viscous_time_convergence.py``).  At
``H/dx = 8`` that sweep contains **zero** admissible points, so the coarse
reading is a startup-transient value while the fine reading is a steady one.
"Error rises from 14.20 % to 17.08 % under refinement" therefore compares two
different admissibility classes: it is a trend estimate confounded by the very
diagnostic the sweep was built to control.

This module re-adjudicates such claims with the confound made explicit, and
never invents a trend the artefacts cannot support:

``matched_trend``
    Two (or more) rungs each supply an admissible reading at the **same**
    terminal-time tier; the trend is measured on matched readings only.
``mixed_admissibility``
    The compared readings come from different admissibility classes: the
    resolution trend is *not established* by the comparison, whatever its sign.
``trend_untestable``
    Fewer than two rungs supply an admissible reading at all: the resolution
    response cannot be measured from the artefacts the repository has.
``time_insensitive_exceedance``
    Extra, class-independent evidence for the same decision: at the rung that
    *does* have admissible readings, **no** terminal-time tier brings the
    reading inside the owner standard, so the over-tolerance value is not a
    startup artefact of that rung.

Everything is read from committed artefacts (plus optional extension ladders,
``benchmarks/couette/out_design/viscous_time_convergence_*.json``); no number
is typed in by hand.

Usage::

    PYTHONPATH=src python benchmarks/method_selection/admissibility_matched_ladder.py
    PYTHONPATH=src python benchmarks/method_selection/admissibility_matched_ladder.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BASE_SWEEP = ROOT / "benchmarks" / "couette" / "out_design" / "viscous_time_convergence.json"
EXTENSION_GLOB = "viscous_time_convergence_*.json"
SWEEP_DIR = ROOT / "benchmarks" / "couette" / "out_design"
REGISTRY_JSON = ROOT / "docs" / "reports" / "method_selection_registry.json"
STRICT_JSON = ROOT / "docs" / "reports" / "method_selection_strict_bound_m21537.json"

OUT_JSON = ROOT / "docs" / "reports" / "method_selection_admissibility_matched_ladder.json"
OUT_MD = ROOT / "docs" / "reports" / "method_selection_admissibility_matched_ladder.md"
OUT_PNG = ROOT / "docs" / "figures" / "method_selection_admissibility_matched_ladder.png"

#: Owner precision standard [%] -- the same threshold the router uses.
OWNER_TOLERANCE_PCT = 15.0
#: The run's own steadiness gate (``gates.steady_drift`` in the sweep artefact).
DEFAULT_DRIFT_GATE = 0.01

#: Claims to re-adjudicate.  Each is anchored on a registry record, so the
#: banked value is read from the registry and never retyped here.
CLAIMS = (
    {
        "claim_id": "moving_wall_worsens_with_resolution",
        "extractor": "x_couette_viscous_time_top_h16",
        "wall": "top",
        "coarse_rung": 8,
        "fine_rung": 16,
        "asserted_by": "benchmarks/method_selection/adjudicate_over_tolerance.py "
                       "(field rederived.worsens_with_resolution)",
    },
)

WALL_FIELDS = {
    "bottom": "bottom_pct",
    "top": "top_pct",
}


# --------------------------------------------------------------------------- #
# Artefact accessors
# --------------------------------------------------------------------------- #
def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sweep_paths() -> list[Path]:
    """The committed sweep first, then any extension ladders (sorted by name)."""
    paths = [BASE_SWEEP]
    for extra in sorted(SWEEP_DIR.glob(EXTENSION_GLOB)):
        if extra.resolve() != BASE_SWEEP.resolve():
            paths.append(extra)
    return [p for p in paths if p.exists()]


def load_points(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Merge every sweep artefact into one point list, keeping provenance.

    A label that appears twice must carry identical readings: two artefacts that
    disagree on the same run mean one of them is stale, and silently keeping the
    later file would hide that.
    """
    merged: dict[str, dict] = {}
    provenance: list[dict] = []
    drift_gate = DEFAULT_DRIFT_GATE
    for path in paths:
        doc = json.loads(path.read_text())
        drift_gate = float(doc.get("gates", {}).get("steady_drift", drift_gate))
        provenance.append({
            "artifact": _rel(path),
            "sha256": _sha(path),
            "resolutions": doc.get("resolutions"),
            "viscous_times_swept": doc.get("viscous_times_swept"),
            "n_points": len(doc.get("points", [])),
        })
        for point in doc.get("points", []):
            row = {
                "label": str(point["label"]),
                "rung": int(point["resolution_H_over_dx"]),
                "tier": float(point["viscous_times"]),
                "t_end": float(point["t_end"]),
                "bottom_pct": float(point["wall_shear_error_pct_bottom"]),
                "top_pct": float(point["wall_shear_error_pct_top"]),
                "steady_drift": float(point["steady_drift"]),
                "admissible": bool(
                    point.get("admissible",
                               float(point["steady_drift"]) <= drift_gate)),
                "source": _rel(path),
            }
            previous = merged.get(row["label"])
            if previous is not None:
                same = all(
                    abs(previous[key] - row[key]) <= 1e-12
                    for key in ("bottom_pct", "top_pct", "steady_drift")
                )
                if not same:
                    raise SystemExit(
                        f"stale artefacts: {row['label']} differs between "
                        f"{previous['source']} and {row['source']}")
                continue
            merged[row["label"]] = row
    points = sorted(merged.values(), key=lambda r: (r["rung"], r["tier"]))
    return points, provenance


def registry_by_extractor() -> dict[str, dict]:
    doc = json.loads(REGISTRY_JSON.read_text())
    return {record["extractor"]: record for record in doc["records"]}


# --------------------------------------------------------------------------- #
# Ladder construction
# --------------------------------------------------------------------------- #
def ladder(points: list[dict], wall: str) -> dict[str, list[dict]]:
    field = WALL_FIELDS[wall]
    out: dict[str, list[dict]] = {}
    for point in points:
        out.setdefault(str(point["rung"]), []).append({
            "tier": point["tier"],
            "t_end": point["t_end"],
            "pct": point[field],
            "steady_drift": point["steady_drift"],
            "admissible": point["admissible"],
            "label": point["label"],
        })
    for rows in out.values():
        rows.sort(key=lambda r: r["tier"])
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def steady_end(rows: list[dict]) -> dict:
    """The reading the sweep's own ``summarise()`` would bank for this rung."""
    admissible = [r for r in rows if r["admissible"]]
    chosen = (admissible or rows)[-1]
    return {
        "label": chosen["label"],
        "tier": chosen["tier"],
        "pct": chosen["pct"],
        "steady_drift": chosen["steady_drift"],
        "admissible": chosen["admissible"],
        "n_admissible": len(admissible),
        "n_points": len(rows),
        "fallback_to_last_point": not admissible,
    }


def matched_reading(rows: list[dict], tier: float) -> dict | None:
    """The admissible reading at an exact terminal-time tier, if there is one."""
    for row in rows:
        if abs(row["tier"] - tier) <= 1e-9:
            return row if row["admissible"] else None
    return None


# --------------------------------------------------------------------------- #
# Adjudication
# --------------------------------------------------------------------------- #
def adjudicate_claim(claim: dict, points: list[dict],
                     registry: dict[str, dict]) -> dict:
    record = registry.get(claim["extractor"])
    if record is None:
        raise SystemExit(f"registry has no record {claim['extractor']}")
    wall = claim["wall"]
    table = ladder(points, wall)
    coarse_rows = table.get(str(claim["coarse_rung"]), [])
    fine_rows = table.get(str(claim["fine_rung"]), [])
    if not coarse_rows or not fine_rows:
        raise SystemExit(f"{claim['claim_id']}: rung missing from the artefacts")

    coarse = steady_end(coarse_rows)
    fine = steady_end(fine_rows)
    admissible_rungs = [rung for rung, rows in table.items()
                        if any(r["admissible"] for r in rows)]

    problems: list[str] = []
    if coarse["fallback_to_last_point"]:
        problems.append(
            f"H/dx={claim['coarse_rung']} has no admissible point "
            f"({coarse['n_points']} tiers, steady_drift gate "
            f"{min(r['steady_drift'] for r in coarse_rows):.2e}); its reading "
            f"{coarse['pct']:.3f} % at {coarse['tier']:g} H^2/nu is a "
            f"startup-transient value (drift {coarse['steady_drift']:.3e})")
    if fine["fallback_to_last_point"]:
        problems.append(
            f"H/dx={claim['fine_rung']} has no admissible point either")

    # Matched-admissibility trend: same tier, both readings admissible.
    matched = None
    if (coarse["admissible"] and fine["admissible"]
            and abs(coarse["tier"] - fine["tier"]) <= 1e-9):
        matched = {
            "tier": fine["tier"],
            "coarse_pct": coarse["pct"],
            "fine_pct": fine["pct"],
            "worsens": fine["pct"] > coarse["pct"],
        }
    else:
        shared = sorted({r["tier"] for r in coarse_rows} & {r["tier"] for r in fine_rows})
        for tier in shared:
            c = matched_reading(coarse_rows, tier)
            f = matched_reading(fine_rows, tier)
            if c is not None and f is not None:
                matched = {
                    "tier": tier,
                    "coarse_pct": c["pct"],
                    "fine_pct": f["pct"],
                    "worsens": f["pct"] > c["pct"],
                }
                break

    if len(admissible_rungs) < 2:
        cls = "trend_untestable"
    elif matched is None:
        cls = "mixed_admissibility"
    else:
        cls = "matched_trend"

    # Class-independent robustness evidence at every rung that has an
    # admissible point: does any terminal-time tier bring the reading inside
    # the owner standard?
    robust = []
    for rung, rows in table.items():
        if not any(r["admissible"] for r in rows):
            continue
        values = {r["tier"]: r["pct"] for r in rows}
        robust.append({
            "rung": int(rung),
            "min_pct": min(values.values()),
            "max_pct": max(values.values()),
            "min_tier": min(values, key=values.get),
            "all_tiers_over_standard": all(v > OWNER_TOLERANCE_PCT
                                           for v in values.values()),
            "admissible_tiers": [r["tier"] for r in rows if r["admissible"]],
        })
    time_insensitive = bool(robust) and all(r["all_tiers_over_standard"]
                                            for r in robust)

    return {
        "claim_id": claim["claim_id"],
        "asserted_by": claim["asserted_by"],
        "extractor": claim["extractor"],
        "method": record["method"],
        "wall": wall,
        "banked_pct": float(record["measured_pct"]),
        "registry_status": record.get("status"),
        "quantities": record.get("quantity"),
        "coarse_rung": claim["coarse_rung"],
        "fine_rung": claim["fine_rung"],
        "coarse_reading": coarse,
        "fine_reading": fine,
        "admissible_rungs": [int(r) for r in admissible_rungs],
        "matched_trend": matched,
        "class": cls,
        "problems": problems,
        "robustness": robust,
        "time_insensitive_exceedance": time_insensitive,
        "admissible_rung_count": len(admissible_rungs),
        "verdict": _verdict(cls, claim, coarse, fine, matched, robust,
                            len(admissible_rungs)),
    }


def _verdict(cls: str, claim: dict, coarse: dict, fine: dict,
             matched: dict | None, robust: list[dict],
             n_admissible_rungs: int) -> str:
    if cls == "trend_untestable":
        head = (f"{claim['claim_id']}: NOT ESTABLISHED as a resolution trend -- "
                f"only {n_admissible_rungs} rung(s) of the sweep have an "
                f"admissible reading, so a matched-admissibility ladder does "
                f"not exist")
    elif cls == "mixed_admissibility":
        head = (f"{claim['claim_id']}: NOT ESTABLISHED as a resolution trend -- "
                f"the compared readings come from different admissibility "
                f"classes (coarse{'admissible' if coarse['admissible'] else 'NOT admissible'}"
                f", fine{'admissible' if fine['admissible'] else 'NOT admissible'})")
    else:
        head = (f"{claim['claim_id']}: matched-admissibility trend at "
                f"{matched['tier']:g} H^2/nu, {matched['coarse_pct']:.3f} % -> "
                f"{matched['fine_pct']:.3f} % "
                f"({'worsens' if matched['worsens'] else 'improves'})")
    if robust:
        bits = [f"H/dx={r['rung']}: best case over all terminal-time tiers "
                f"{r['min_pct']:.3f} % at {r['min_tier']:g} H^2/nu "
                f"({'still above' if r['all_tiers_over_standard'] else 'inside'} "
                f"the {OWNER_TOLERANCE_PCT:g} % standard)" for r in robust]
        head += "; independent evidence " + "; ".join(bits)
    return head


def design_reading_check(points: list[dict], registry: dict[str, dict]) -> dict | None:
    """Is the *endorsed* design-time reading taken on a steady flow?

    ``x_couette_wall_shear_design`` is the record the router SELECTS the profile
    on (0.478 % at H/dx=16).  The sweep says what that reading's steadiness was
    and what the same rung measures once it is admissible; both numbers matter
    and neither may be dropped.
    """
    record = registry.get("x_couette_wall_shear_design")
    if record is None:
        return None
    doc = json.loads(BASE_SWEEP.read_text())
    design_tier = float(doc["design_viscous_times"])
    finest = max(int(r) for r in ladder(points, "bottom"))
    rows = ladder(points, "bottom")[str(finest)]
    row = next((r for r in rows if abs(r["tier"] - design_tier) <= 1e-9), None)
    if row is None:
        return None
    blocked = [r for r in rows if r["admissible"]]
    admissible_reading = blocked[-1] if blocked else None
    gap = abs(row["pct"] - float(record["measured_pct"]))
    return {
        "extractor": "x_couette_wall_shear_design",
        "rung": finest,
        "design_tier": design_tier,
        "banked_pct": float(record["measured_pct"]),
        "recomputed_pct": row["pct"],
        "recomputed_vs_banked_pct": gap,
        "steady_drift": row["steady_drift"],
        "admissible": row["admissible"],
        "admissible_reading": None if admissible_reading is None else {
            "tier": admissible_reading["tier"],
            "pct": admissible_reading["pct"],
            "steady_drift": admissible_reading["steady_drift"],
        },
        "class": ("admissible_design_reading" if row["admissible"]
                  else "design_reading_not_admissible"),
        "within_owner_standard": row["pct"] <= OWNER_TOLERANCE_PCT,
        "note": (
            "the banked design-time reading is reproduced from the sweep to "
            f"{gap:.2e} % but its run does "
            f"{'not ' if not row['admissible'] else ''}clear the run's own "
            "steady_drift gate; the admissible reading of the same rung is "
            "reported alongside so the accuracy claim can be read on either"),
    }


def decision_linkage(claim: dict, banked_pct: float) -> dict:
    """Is the record the published exclusion basis?  Read, never assumed.

    The strict-bound rows cite an exclusion by (method, worst accuracy value),
    so the record is matched on its banked value, not on its name.
    """
    if not STRICT_JSON.exists():
        return {"strict_artifact": None, "rows": []}
    strict = json.loads(STRICT_JSON.read_text())
    out = {"strict_artifact": str(STRICT_JSON.relative_to(ROOT)), "rows": []}
    for row in strict.get("rows", []):
        for mode in ("strict", "lenient"):
            for decision in (row.get(mode) or {}).get("excluded", []) or []:
                worst = decision.get("worst_accuracy_pct")
                if worst is None:
                    continue
                if abs(float(worst) - banked_pct) <= 1e-6:
                    out["rows"].append({"label": row.get("label"), "mode": mode,
                                        "decision": decision})
    return out


def build() -> dict:
    paths = sweep_paths()
    points, provenance = load_points(paths)
    registry = registry_by_extractor()
    claims = []
    for claim in CLAIMS:
        entry = adjudicate_claim(claim, points, registry)
        entry["decision_linkage"] = decision_linkage(claim, entry["banked_pct"])
        claims.append(entry)
    return {
        "anchor": "XVAL-SPH-METHOD-SELECTION-2026-11-12",
        "generated_by": "benchmarks/method_selection/admissibility_matched_ladder.py",
        "owner_tolerance_pct": OWNER_TOLERANCE_PCT,
        "provenance": provenance,
        "ladder_top": ladder(points, "top"),
        "ladder_bottom": ladder(points, "bottom"),
        "claims": claims,
        "design_reading": design_reading_check(points, registry),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_table(rows: list[dict]) -> list[str]:
    lines = ["| rung H/dx | t_end/(H^2/nu) | error [%] | steady_drift | admissible |",
             "|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row_label(row)} | {row['tier']:g} | {row['pct']:.3f} | "
                     f"{row['steady_drift']:.3e} | {'yes' if row['admissible'] else 'no'} |")
    return lines


def row_label(row: dict) -> str:
    return row["label"].split("-")[0].lstrip("r")


def render_markdown(result: dict) -> str:
    lines = [
        "# Admissibility-matched refinement ladder for the Couette wall-shear records",
        "",
        f"**Anchor**: `{result['anchor']}` ｜ **generated by**: "
        f"`{result['generated_by']}` ｜ owner standard "
        f"{result['owner_tolerance_pct']:g} %",
        "",
        "## Ladder as measured (moving wall)",
        "",
        *_fmt_table([r for rows in result["ladder_top"].values() for r in rows]),
        "",
        "## Ladder as measured (bottom wall)",
        "",
        *_fmt_table([r for rows in result["ladder_bottom"].values() for r in rows]),
        "",
        "## Claims",
        "",
    ]
    for claim in result["claims"]:
        lines += [
            f"### `{claim['claim_id']}` -- class `{claim['class']}`",
            "",
            f"- asserted by: `{claim['asserted_by']}`",
            f"- registry record `{claim['extractor']}` (method `{claim['method']}`), "
            f"banked {claim['banked_pct']:.3f} %",
            f"- coarse rung H/dx={claim['coarse_rung']}: "
            f"{claim['coarse_reading']['pct']:.3f} % at "
            f"{claim['coarse_reading']['tier']:g} H^2/nu, "
            f"admissible={claim['coarse_reading']['admissible']}, "
            f"n_admissible={claim['coarse_reading']['n_admissible']}",
            f"- fine rung H/dx={claim['fine_rung']}: "
            f"{claim['fine_reading']['pct']:.3f} % at "
            f"{claim['fine_reading']['tier']:g} H^2/nu, "
            f"admissible={claim['fine_reading']['admissible']}, "
            f"n_admissible={claim['fine_reading']['n_admissible']}",
            f"- matched-admissibility trend: `{claim['matched_trend']}`",
            f"- verdict: {claim['verdict']}",
        ]
        for problem in claim["problems"]:
            lines.append(f"- problem: {problem}")
        if claim["decision_linkage"].get("rows"):
            cited = ", ".join(row["label"] for row in claim["decision_linkage"]["rows"])
            lines.append(f"- published decision(s) citing this record: {cited}")
        lines.append("")
    if result.get("design_reading"):
        design = result["design_reading"]
        lines += [
            "## Is the endorsed design-time reading steady?",
            "",
            f"- record `{design['extractor']}` banked {design['banked_pct']:.3f} % "
            f"(recomputed from the sweep {design['recomputed_pct']:.3f} %, "
            f"difference {design['recomputed_vs_banked_pct']:.2e} %)",
            f"- that run's steady_drift {design['steady_drift']:.3e} -> "
            f"admissible={design['admissible']} (class `{design['class']}`)",
        ]
        if design["admissible_reading"]:
            adm = design["admissible_reading"]
            lines.append(f"- admissible reading of the same rung: {adm['pct']:.3f} % "
                         f"at {adm['tier']:g} H^2/nu (drift {adm['steady_drift']:.3e})")
        lines += [f"- {design['note']}", ""]
    lines += ["## Provenance", ""]
    for entry in result["provenance"]:
        lines.append(f"- `{entry['artifact']}` sha256 `{entry['sha256'][:16]}...` "
                     f"({entry['n_points']} points, rungs {entry['resolutions']})")
    lines.append("")
    return "\n".join(lines)


def render_figure(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for wall, colour, marker in (("top", "crimson", "o"), ("bottom", "tab:blue", "s")):
        table = result["ladder_top"] if wall == "top" else result["ladder_bottom"]
        for rung, rows in table.items():
            tiers = [r["tier"] for r in rows]
            values = [r["pct"] for r in rows]
            ax.plot(tiers, values, marker=marker, color=colour, ls="-",
                    alpha=0.45 if rung == min(table, key=int) else 0.9,
                    label=f"{wall} wall, H/dx={rung}")
            for row in rows:
                if row["admissible"]:
                    ax.plot([row["tier"]], [row["pct"]], marker="*", ms=13,
                            color="black", zorder=5)
    ax.axhline(result["owner_tolerance_pct"], color="k", ls="--", lw=1.1,
               label=f"owner standard {result['owner_tolerance_pct']:g} %")
    ax.set_xlabel("terminal time  $t_{end}/(H^2/\\nu)$")
    ax.set_ylabel("wall-shear error  $|1-\\tau_{SPH}/\\tau_{exact}|$  [%]")
    ax.set_title("Reading vs run length; ★ = steady-admissible")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels, values, colours = [], [], []
    for claim in result["claims"]:
        for key, tag in (("coarse_reading", "coarse (fallback)"),
                         ("fine_reading", "fine")):
            reading = claim[key]
            labels.append(f"H/dx={claim[key.replace('_reading', '_rung')]}\n{tag}")
            values.append(reading["pct"])
            colours.append("tab:red" if not reading["admissible"] else "tab:green")
    ax.bar(range(len(values)), values, color=colours)
    ax.axhline(result["owner_tolerance_pct"], color="k", ls="--", lw=1.1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("banked moving-wall error [%]")
    ax.set_title("The trend claim's two readings\n(red = non-admissible reading)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed artefacts differ from a fresh build")
    args = parser.parse_args(argv)
    result = build()
    markdown = render_markdown(result)
    if args.check:
        stale = []
        if not OUT_JSON.exists() or json.loads(OUT_JSON.read_text()) != result:
            stale.append(str(OUT_JSON.relative_to(ROOT)))
        if not OUT_MD.exists() or OUT_MD.read_text() != markdown:
            stale.append(str(OUT_MD.relative_to(ROOT)))
        if stale:
            print("stale artefacts: " + ", ".join(stale), flush=True)
            return 1
        print("admissibility-matched ladder artefacts are fresh", flush=True)
        return 0
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    OUT_MD.write_text(markdown)
    try:
        render_figure(result, OUT_PNG)
        print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}, "
              f"{OUT_PNG.relative_to(ROOT)}", flush=True)
    except Exception as exc:  # plotting is reporting, never a precondition
        print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}; "
              f"figure skipped: {exc}", flush=True)
    for claim in result["claims"]:
        print(f"[{claim['class']}] {claim['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
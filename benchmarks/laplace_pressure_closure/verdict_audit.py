"""Machine-checked verdict matrix for the Young-Laplace evidence of this benchmark.

Why this file exists
--------------------
Three verification records (`verification_laplace_pressure_floor.md`,
`verification_laplace_pressure_closure.md`, `verification_laplace_pressure_jump.md`)
report the *same* physical quantity under four different measurement definitions
and three different configurations, and the ``WITHHELD`` grading rests on prose
written by hand.  Prose drifts: in the revision this script was added for, the
document's headline sentence took the "+8.8 %" read-out as "the best closure
variant" although ``tests/test_laplace_pressure_closure.py`` only ranks variants
that satisfy Adami's spurious-current criterion, and then called that number
"not inside the 15 % acceptance band" although ``|8.8| < 15`` by arithmetic.

So the band membership, the admissibility and the independence of every recorded
read-out are now *computed here* from the evidence files, and both the document
generator (`make_doc.py`) and the test suite read this matrix instead of
re-deriving the rule.  Nothing in this file advances the physics, so it costs
seconds and needs no time stepping.

    PYTHONPATH=src python benchmarks/laplace_pressure_closure/verdict_audit.py

Writes ``verdict_matrix.json`` next to itself.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORCE = HERE / "report_force_balance.json"
LADDER = HERE / "raw" / "relax_ladder.json"
FLOOR = HERE.parent / "laplace_pressure_floor" / "report.json"
OUT = HERE / "verdict_matrix.json"

#: Platform acceptance band on a key quantity (owner standard).
GATE_PCT = 15.0
#: Adami, Ihmsen & Adams (2012) spurious-current criterion for a static drop.
SPURIOUS_CURRENT_MAX = 0.1
#: Bit-identity threshold for "these two rows are the same run".
IDENTICAL_TOL = 1e-12
#: Which group answers the capability question (Young-Laplace pressure jump).
#: The static force balance is a consistency identity: `dp/dr = rho a_r` holds by
#: construction, so being inside the band there says nothing about whether the
#: *pressure field* carries the jump.  Counting it as capability evidence is the
#: verdict inflation this file exists to prevent.
CAPABILITY_GROUP = "relaxation"


def _row(**kw) -> dict:
    row = {
        "label": kw["label"],
        "group": kw["group"],
        "source": kw["source"],
        "config": kw["config"],
        "value_Pa": kw["value_Pa"],
        "rel_error_pct": kw["rel_error_pct"],
        "within_15pct": abs(kw["rel_error_pct"]) < GATE_PCT,
        "within_3pct": abs(kw["rel_error_pct"]) < 3.0,
    }
    row.update(kw.get("extra", {}))
    return row


def force_rows(force: dict) -> list[dict]:
    """Static force-balance read-out: no time stepping, so no transient to judge."""
    rows = []
    for case in force["cases"]:
        fs = case["force_side"]
        rows.append(_row(
            label=f"flux integral -int rho a_r dr, dx = {case['dx_m'] * 1e3:.1f} mm",
            group="static_force_balance",
            source="report_force_balance.json",
            config={"dx_m": case["dx_m"], "closure": None},
            value_Pa=fs["delta_p_force_Pa"],
            rel_error_pct=fs["deficit_pct"],
            extra={
                # A static diagnostic has no surface motion to breathe: the
                # admissibility criterion of a *relaxation* does not apply.
                "admissible": True,
                "admissibility_basis": "static CSF evaluation, no transient",
                "spurious_current_over_limit": None,
            },
        ))
    return rows


def relax_rows(ladder: dict) -> list[dict]:
    """Relaxation read-out, time-averaged absolute core pressure (C1 definition)."""
    rows = []
    for case in ladder["cases"]:
        vmax = case["vmax_over_vref"]
        rows.append(_row(
            label=f"relaxed <p_core>, {case['case']}",
            group="relaxation",
            source="raw/relax_ladder.json",
            config={
                "dx_m": ladder["dx_m"], "sound_speed": ladder["sound_speed"],
                "closure": case["config"].get("closure", False),
                "donors": case["config"].get("donors"),
                "layers": case["config"].get("layers"),
            },
            value_Pa=case["tail_mean_Pa"],
            rel_error_pct=case["tail_rel_error_pct"],
            extra={
                "tail_rel_error_pct": case["tail_rel_error_pct"],
                "final_snapshot_rel_error_pct": case["final_abs_rel_error_pct"],
                "vmax_over_vref": vmax,
                "admissible": vmax < SPURIOUS_CURRENT_MAX,
                "admissibility_basis": (f"v_max/v_ref = {vmax:.3f} vs Adami limit "
                                        f"{SPURIOUS_CURRENT_MAX}"),
                "spurious_current_over_limit": vmax / SPURIOUS_CURRENT_MAX,
                "pressure_side_used": "absolute (p_core - 0), time averaged",
            },
        ))
    return rows


def floor_rows(floor: dict) -> list[dict]:
    """The floor benchmark's own read-out (closure off), for cross-reference."""
    rows = []
    for run in floor["runs"]:
        tail = run["absolute_time_average_tail"]
        rows.append(_row(
            label=(f"floor <p_core> tail, dx = {run['dx_m'] * 1e3:.1f} mm, "
                   f"c0 = {run['sound_speed']:g} m/s (closure off)"),
            group="floor_reference",
            source="laplace_pressure_floor/report.json",
            config={"dx_m": run["dx_m"], "sound_speed": run["sound_speed"],
                    "closure": False},
            value_Pa=tail["delta_p_Pa"],
            rel_error_pct=tail["rel_error_pct"],
            extra={
                "admissible": (run["spurious_current"]["vmax_over_vref_final"]
                               < SPURIOUS_CURRENT_MAX),
                "admissibility_basis": ("v_max/v_ref = "
                                        f"{run['spurious_current']['vmax_over_vref_final']:.3f}"),
                "spurious_current_over_limit":
                    run["spurious_current"]["vmax_over_vref_final"] / SPURIOUS_CURRENT_MAX,
            },
        ))
    return rows


def mark_independent_evidence(rows: list[dict]) -> list[dict]:
    """Flag rows that are the *same run* reported twice, and say what duplicates what.

    The relaxation ladder carries two "interior" rungs that differ only in
    ``surface_pressure_layers``.  On this interface they are bit-identical: the
    surface is one particle thick, every receiver finds a non-free donor in the
    first pass, so the second pass has nothing left to fill.  A table that lists
    both rows looks like two independent confirmations of +8.8 % when it is one.
    """
    seen: list[dict] = []
    for row in rows:
        dup_of = None
        for prev in seen:
            if prev["group"] != row["group"] or prev["config"]["dx_m"] != row["config"]["dx_m"]:
                continue
            same_value = abs(prev["value_Pa"] - row["value_Pa"]) <= IDENTICAL_TOL * max(
                1.0, abs(row["value_Pa"]))
            same_tail = abs(prev.get("tail_rel_error_pct", 0.0)
                            - row.get("tail_rel_error_pct", 0.0)) <= IDENTICAL_TOL
            same_final = abs(prev.get("final_snapshot_rel_error_pct", 0.0)
                             - row.get("final_snapshot_rel_error_pct", 0.0)) <= IDENTICAL_TOL
            same_v = abs(prev.get("vmax_over_vref", 0.0)
                         - row.get("vmax_over_vref", 0.0)) <= IDENTICAL_TOL
            if same_value and same_tail and same_final and same_v:
                dup_of = prev["label"]
                break
        row["independent_evidence"] = dup_of is None
        row["duplicate_of"] = dup_of
        seen.append(row)
    return rows


def main() -> None:
    force = json.loads(FORCE.read_text())
    ladder = json.loads(LADDER.read_text()) if LADDER.exists() else None
    floor = json.loads(FLOOR.read_text()) if FLOOR.exists() else None
    dp = force["reference"]["analytic_dp_Pa"]

    rows = force_rows(force)
    if ladder:
        rows += relax_rows(ladder)
    if floor:
        rows += floor_rows(floor)
    rows = mark_independent_evidence(rows)

    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)

    def best(cands):
        if not cands:
            return None
        pick = min(cands, key=lambda r: abs(r["rel_error_pct"]))
        return {k: pick[k] for k in ("label", "group", "value_Pa", "rel_error_pct",
                                     "within_15pct", "admissible", "config")}

    def usable(group_rows):
        return [r for r in group_rows if r["admissible"] and r["independent_evidence"]]

    cap_rows = by_group.get(CAPABILITY_GROUP, [])
    cap_usable = usable(cap_rows)
    in_band_adm = [r for r in cap_usable if r["within_15pct"]]

    group_summary = {}
    for name, group_rows in by_group.items():
        g_usable = usable(group_rows)
        group_summary[name] = {
            "n_readouts": len(group_rows),
            "n_independent": sum(1 for r in group_rows if r["independent_evidence"]),
            "n_within_15pct": sum(1 for r in group_rows if r["within_15pct"]),
            "n_within_15pct_admissible": sum(1 for r in g_usable if r["within_15pct"]),
            "best_any": best(group_rows),
            "best_admissible": best(g_usable),
            "answers_capability_question": name == CAPABILITY_GROUP,
        }

    # Defects are *derived*, not asserted: each one carries the numbers that make
    # it a defect, so a reader can re-check the claim without trusting the prose.
    defects = []
    relax = by_group.get("relaxation", [])
    off = next((r for r in relax if r["config"]["closure"] is False), None)
    inadmissible_in_band = [r for r in relax if r["within_15pct"] and not r["admissible"]]
    for row in inadmissible_in_band:
        defects.append({
            "id": "in-band-but-inadmissible",
            "row": row["label"],
            "why": (f"tail error {row['rel_error_pct']:+.1f} % is inside the +/-{GATE_PCT:g} % "
                    f"band, but the same run's spurious current is "
                    f"{row['spurious_current_over_limit']:.1f}x the Adami limit and its final "
                    f"snapshot is {row['final_snapshot_rel_error_pct']:+.1f} % off: the surface "
                    "is breathing, so the time average is not a static Laplace measurement."),
            "consequence": ("it may not be quoted as 'the best closure variant' nor as a "
                            "capability win"),
        })
    for row in relax:
        if not row["independent_evidence"]:
            defects.append({
                "id": "duplicate-row-counted-as-variant",
                "row": row["label"],
                "why": (f"bit-identical to '{row['duplicate_of']}' (value, tail, final "
                        "snapshot and spurious current all equal to 1e-12): the "
                        "surface_pressure_layers knob is inert on a one-particle-thick "
                        "interface, every receiver is filled in the first pass"),
                "consequence": "the two table rows are one piece of evidence, not two",
            })
    if off is not None and ladder is not None:
        drift = abs(force["cases"][1]["force_side"]["deficit_pct"]
                    - force["cases"][0]["force_side"]["deficit_pct"])
        defects.append({
            "id": "resolution-not-laddered",
            "row": "relaxation group",
            "why": ("the whole relaxation group sits at one grid and one sound speed "
                    f"(dx = {ladder['dx_m'] * 1e3:.1f} mm, c0 = {ladder['sound_speed']:g} m/s); "
                    "the static force balance already moves by "
                    f"{drift:.1f} percentage points between the two grids it covers, so the "
                    "closure read-out cannot be assumed grid-converged"),
            "consequence": "WITHHELD until rung dx = 1.5 mm (and one c0 change) is recorded",
        })

    status = "WITHHELD"
    if in_band_adm:
        # Inside the band on admissible evidence, but the group lives on one grid.
        status = "PROVISIONAL" if any(d["id"] == "resolution-not-laddered"
                                      for d in defects) else "VERIFIED"

    matrix = {
        "benchmark": "laplace_pressure_closure_verdict",
        "question": ("which Young-Laplace read-outs are inside the acceptance band, and is "
                     "each one admissible and independent evidence?"),
        "reference": {"sigma_N_per_m": force["reference"]["sigma_N_per_m"],
                      "radius_m": force["reference"]["radius_m"],
                      "analytic_dp_Pa": dp},
        "gate": {"band_pct": GATE_PCT,
                 "admissibility": f"v_max/v_ref < {SPURIOUS_CURRENT_MAX} (Adami et al. 2012)",
                 "definitions_note": ("all read-outs are absolute pressure (p_core - 0); the "
                                      "deprecated p_core - p(0.80-0.95 R) definition is not used")},
        "readouts": rows,
        "summary": {
            "n_readouts": len(rows),
            "n_independent": sum(1 for r in rows if r["independent_evidence"]),
            "answers_capability_question": group_summary[CAPABILITY_GROUP],
            "by_group": group_summary,
        },
        "defects": defects,
        "verdict": {
            "status": status,
            "reason": (
                "no admissible read-out is inside the band"
                if not in_band_adm else
                f"{len(in_band_adm)} admissible read-out(s) inside the band, but not yet "
                "shown to be resolution-converged"
            ),
            "not_capability_evidence": [
                r["label"] for r in rows if r["group"] != CAPABILITY_GROUP],
            "settled": ("the discrete CSF force holds sigma/R to "
                        f"{force['cases'][0]['force_side']['deficit_pct']:+.1f} % at "
                        f"dx = {force['cases'][0]['dx_m'] * 1e3:.1f} mm, so the missing "
                        "jump is not a force-amplitude error"),
        },
    }
    OUT.write_text(json.dumps(matrix, indent=2) + "\n")

    print(f"{len(rows)} read-outs, {matrix['summary']['n_independent']} independent")
    for row in rows:
        print(f"  {row['label']:62s} {row['rel_error_pct']:+7.1f} %  "
              f"band={'Y' if row['within_15pct'] else 'n'}  "
              f"adm={'Y' if row['admissible'] else 'n'}  "
              f"indep={'Y' if row['independent_evidence'] else 'n'}")
    print("capability group best:", group_summary[CAPABILITY_GROUP]["best_admissible"])
    print("verdict:", status, "-", matrix["verdict"]["reason"])
    print("defects:", len(defects))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

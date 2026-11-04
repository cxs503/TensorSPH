#!/usr/bin/env python3
"""Verification-claims ledger audit for TensorSPH (mission m3544).

Why this exists
---------------
TensorSPH has ten verification documents that all grade the *same* physical
quantity -- the Young-Laplace pressure jump ``Delta p = sigma/R`` for the 2-D
droplet case (sigma = 7.2 N/m, R = 0.02 m, sigma/R = 360 Pa).  They quote
numbers between +5.6% and -93.6%.  Nothing in the repo said which of those a
downstream reader (or a platform capability audit) is allowed to cite, and two
of the cited artifacts were silently overwritten in place by later runs.

This script does not try to parse prose.  It takes a hand-vetted manifest
(``claims_manifest.json``) whose every row names the document, a verbatim
quote, and the committed JSON artifact that backs the number, and then
machine-checks four things per row:

  QUOTE            the quote really occurs in the document;
  PROVENANCE       the artifact + JSON pointer still exists and still holds
                   the quoted error (catches artifact overwrite-in-place);
  ARITHMETIC       stored rel_error_pct == (measured - reference)/reference;
  CONSISTENCY      for snapshots: is the headline reading the *settled* state,
                   or was it taken while the transient was still decaying, and
                   is the quoted error resolvable against the run's own
                   temporal noise?

Outputs ``ledger.json`` (machine readable, consumed by the test gate and by
``make_doc.py``) and ``ledger.png``.

Usage:
    python benchmarks/verification_ledger/audit.py [--no-plot]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "claims_manifest.json"
LEDGER = HERE / "ledger.json"
FIGURE = HERE / "ledger.png"

# Marker that every document holding a non-authoritative claim must carry.
MARKER = "台帐 m3544"

# Tolerance (percentage points) when comparing a quoted error with the value
# recomputed from the artifact.  Artifacts store full float precision, docs
# round to one decimal, so 0.1 pt is the natural quantum.
TOL_PCT = 0.15
GATE_15 = 15.0
GATE_3 = 3.0


def resolve_pointer(doc, pointer: str):
    """Minimal RFC-6901 JSON pointer resolver."""
    if pointer in ("", "/"):
        return doc
    cur = doc
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(token)]
        elif isinstance(cur, dict):
            cur = cur[token]
        else:  # pragma: no cover - defensive
            raise KeyError(pointer)
    return cur


def rel(p: str) -> Path:
    return REPO / p


def check_quote(claim) -> dict:
    path = rel(claim["doc"])
    if not path.exists():
        return {"status": "MISSING_DOC", "detail": f"{claim['doc']} not found"}
    text = path.read_text(encoding="utf-8")
    found = claim["quote"] in text
    return {
        "status": "OK" if found else "QUOTE_ABSENT",
        "detail": "" if found else f"quote {claim['quote']!r} not present in {claim['doc']}",
    }


def check_provenance(claim, case) -> dict:
    """Does the cited artifact still carry the number the document quotes?"""
    out = {"artifact": claim["cited_artifact"], "pointer": claim["artifact_pointer"]}
    path = rel(claim["cited_artifact"])
    if not path.exists():
        out.update(status="ARTIFACT_MISSING",
                   detail=f"{claim['cited_artifact']} does not exist")
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        stored = resolve_pointer(data, claim["artifact_pointer"])
    except (KeyError, IndexError, ValueError) as exc:
        out.update(status="POINTER_MISSING", detail=f"{claim['artifact_pointer']}: {exc}")
        return out

    out["artifact_error_pct"] = float(stored)
    delta = abs(float(stored) - float(claim["reported_error_pct"]))
    out["delta_pct_points"] = round(delta, 4)

    if delta > TOL_PCT:
        out.update(
            status="OVERWRITTEN",
            detail=(
                f"document quotes {claim['reported_error_pct']:+.2f}% but the artifact at "
                f"this path now holds {float(stored):+.2f}% -- the path was reused by a "
                "later run and the original evidence for this claim is gone"
            ),
        )
    else:
        out.update(status="RESOLVED", detail="")

    # Arithmetic: does the artifact's own stored error follow from its raw values?
    if claim.get("measured_pointer") and claim.get("reference_pointer"):
        try:
            measured = float(resolve_pointer(data, claim["measured_pointer"]))
            reference = float(resolve_pointer(data, claim["reference_pointer"]))
        except (KeyError, IndexError, ValueError):
            out["arithmetic"] = {"status": "POINTER_MISSING"}
        else:
            recomputed = (measured - reference) / reference * 100.0
            out["arithmetic"] = {
                "status": "OK" if abs(recomputed - float(stored)) <= TOL_PCT else "MISMATCH",
                "measured": measured,
                "reference": reference,
                "recomputed_pct": round(recomputed, 4),
                "stored_pct": round(float(stored), 4),
            }
            if abs(recomputed - float(stored)) > TOL_PCT:
                out["arithmetic"]["detail"] = (
                    f"artifact stores {float(stored):+.2f}% but its own raw values give "
                    f"{recomputed:+.2f}%"
                )
    return out


def tail_window(values, fraction: float = 0.4):
    n = max(2, int(math.ceil(len(values) * fraction)))
    return values[-n:]


def check_consistency(claim, case) -> dict | None:
    """For snapshot readings, is the quoted number the settled state?"""
    pointer = claim.get("trajectory_pointer")
    if not pointer:
        return None
    path = rel(claim["cited_artifact"])
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        traj = [float(v) for v in resolve_pointer(data, pointer)]
    except (KeyError, IndexError, ValueError):
        return None
    if len(traj) < 3:
        return None

    reference = float(case["reference_pa"])
    quoted = float(claim["reported_error_pct"])
    quoted_dev_pa = abs(quoted) / 100.0 * reference
    endpoint = traj[-1]
    endpoint_err_pct = (endpoint - reference) / reference * 100.0

    tail = tail_window(traj)
    mean = sum(tail) / len(tail)
    var = sum((v - mean) ** 2 for v in tail) / (len(tail) - 1)
    std = math.sqrt(var)
    swing = max(traj) - min(traj)

    out = {
        "trajectory": claim["trajectory_pointer"],
        "n_samples": len(traj),
        "first": round(traj[0], 2),
        "endpoint": round(endpoint, 2),
        "endpoint_error_pct": round(endpoint_err_pct, 2),
        "tail_mean": round(mean, 2),
        "tail_std": round(std, 2),
        "swing": round(swing, 2),
        "quoted_deviation_pa": round(quoted_dev_pa, 2),
        "noise_ratio": round(quoted_dev_pa / std, 3) if std > 0 else None,
        "settled": abs(endpoint_err_pct) <= GATE_15,
    }
    # The quoted agreement is only meaningful if it is larger than the run's own
    # temporal scatter; otherwise "within 15%" is indistinguishable from luck.
    out["resolvable"] = bool(std > 0 and quoted_dev_pa >= std)
    # Is the headline reading the settled state at all?
    out["snapshot_matches_endpoint"] = (
        abs(float(claim.get("reported_error_pct", 0)) - endpoint_err_pct) <= GATE_15
    )
    problems = []
    if not out["resolvable"]:
        problems.append(
            f"quoted deviation {quoted_dev_pa:.1f} Pa is {out['noise_ratio']:.2f}x the run's "
            f"own tail scatter ({std:.1f} Pa) -- inside the noise band"
        )
    if not out["snapshot_matches_endpoint"]:
        problems.append(
            f"quoted {quoted:+.1f}% vs trajectory endpoint {endpoint_err_pct:+.1f}% "
            f"({endpoint:.1f} Pa) -- the headline is not the settled state"
        )
    if not out["settled"]:
        problems.append(
            f"trajectory endpoint {endpoint_err_pct:+.1f}% itself misses the 15% bar"
        )
    out["status"] = "FLAGGED" if problems else "OK"
    out["problems"] = problems
    return out


def check_marker(claim) -> dict:
    path = rel(claim["doc"])
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    has = MARKER in text
    return {"status": "OK" if has else "MARKER_ABSENT",
            "detail": "" if has else f"{claim['doc']} lacks the '{MARKER}' pointer block"}


def build_ledger() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    case = manifest["case"]
    authoritative_id = manifest["authoritative_claim"]

    rows = []
    for claim in manifest["claims"]:
        row = {
            "id": claim["id"],
            "doc": claim["doc"],
            "section": claim.get("section", ""),
            "quote": claim["quote"],
            "definition_class": claim["definition_class"],
            "definition": claim["definition"],
            "reported_error_pct": claim["reported_error_pct"],
            "reported_verdict": claim["reported_verdict"],
            "authoritative": claim["id"] == authoritative_id,
            "note": claim.get("note", ""),
            "checks": {
                "quote": check_quote(claim),
                "provenance": check_provenance(claim, case),
                "marker": check_marker(claim),
            },
        }
        consistency = check_consistency(claim, case)
        if consistency is not None:
            row["checks"]["consistency"] = consistency
        rows.append(row)

    # ---- aggregate findings -------------------------------------------------
    errors = [r["reported_error_pct"] for r in rows]
    spread = max(errors) - min(errors)

    overwritten = [r["id"] for r in rows if r["checks"]["provenance"]["status"] == "OVERWRITTEN"]
    flagged = [r["id"] for r in rows
               if r["checks"].get("consistency", {}).get("status") == "FLAGGED"]
    unmarked = [r["id"] for r in rows
                if not r["authoritative"] and r["checks"]["marker"]["status"] != "OK"]
    passed = [r["id"] for r in rows
              if r["reported_verdict"].upper().startswith("PASS")]
    unsupported_pass = [
        r["id"] for r in rows
        if r["id"] in passed
        and (r["checks"]["provenance"]["status"] != "RESOLVED"
             or r["checks"].get("consistency", {}).get("status") == "FLAGGED")
    ]

    derived = derive_trajectories(manifest, case)
    authoritative = next(r for r in rows if r["authoritative"])
    return {
        "ledger": manifest["ledger"],
        "mission": manifest["mission"],
        "quantity": manifest["quantity"],
        "case": case,
        "definition_classes": manifest["definition_classes"],
        "gates": {"platform_15pct": GATE_15, "publication_3pct": GATE_3,
                  "quote_tolerance_pct_points": TOL_PCT, "marker": MARKER},
        "claims": rows,
        "summary": {
            "n_claims": len(rows),
            "n_docs": len({r["doc"] for r in rows}),
            "error_range_pct": [round(min(errors), 2), round(max(errors), 2)],
            "spread_pct_points": round(spread, 2),
            "provenance_resolved": sorted(
                r["id"] for r in rows if r["checks"]["provenance"]["status"] == "RESOLVED"),
            "provenance_overwritten": sorted(overwritten),
            "consistency_flagged": sorted(flagged),
            "docs_needing_marker": sorted(unmarked),
            "claims_asserting_pass": sorted(passed),
            "unsupported_pass_claims": sorted(unsupported_pass),
            "authoritative_claim": authoritative_id,
            "authoritative_error_pct": round(authoritative["reported_error_pct"], 2),
            "authoritative_definition": authoritative["definition"],
            "evidence_grade": manifest["evidence_grade"],
            "evidence_grade_reason": manifest["evidence_grade_reason"],
            "within_15pct": abs(authoritative["reported_error_pct"]) <= GATE_15,
            "within_3pct": abs(authoritative["reported_error_pct"]) <= GATE_3,
            "derived_trajectories": derived,
        },
    }


def derive_trajectories(manifest: dict, case: dict) -> list:
    """Tail statistics recomputed straight from every cited artifact trajectory.

    These are *not* document claims; they are what the committed artifacts
    themselves imply, which is what the ledger needs in order to say whether
    any artifact supports the +5.6% headline.
    """
    reference = float(case["reference_pa"])
    out = []
    for claim in manifest["claims"]:
        pointer = claim.get("trajectory_pointer")
        if not pointer:
            continue
        path = rel(claim["cited_artifact"])
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            traj = [float(v) for v in resolve_pointer(data, pointer)]
        except (KeyError, IndexError, ValueError):
            continue
        tail = tail_window(traj)
        mean = sum(tail) / len(tail)
        std = math.sqrt(sum((v - mean) ** 2 for v in tail) / (len(tail) - 1))
        out.append({
            "claim": claim["id"],
            "artifact": claim["cited_artifact"],
            "trajectory": pointer,
            "n_samples": len(traj),
            "raw_range_pa": [round(min(traj), 2), round(max(traj), 2)],
            "tail_mean_pa": round(mean, 2),
            "tail_std_pa": round(std, 2),
            "tail_mean_error_pct": round((mean - reference) / reference * 100.0, 2),
            "min_abs_error_achievable_pct": round(
                min(abs((v - reference) / reference * 100.0) for v in traj), 2),
            "note": (
                "derived by this audit from the committed trajectory, not a claim "
                "made by any document"
            ),
        })
    return out


def plot(ledger: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    claims = ledger["claims"]
    ids = [c["id"] for c in claims]
    errs = [c["reported_error_pct"] for c in claims]
    colors = ["#c0392b" if c["authoritative"] else "#7f8c8d" for c in claims]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    bars = ax.barh(ids, errs, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", lw=0.8)
    for i, (gate, style, label) in enumerate(((GATE_15, "--", "15% platform bar"),
                                              (GATE_3, ":", "3% publication bar"))):
        ax.axvline(gate, color="#2980b9", ls=style, lw=1.2,
                   label=label if i == 0 else None)
        ax.axvline(-gate, color="#2980b9", ls=style, lw=1.2,
                   label=label if i == 1 else None)
    ax.set_xlabel("quoted error vs $\\sigma/R = 360$ Pa  [%]")
    ax.set_title("Young–Laplace $\\Delta p$: six claims, one physical quantity")
    ax.grid(axis="x", alpha=0.3)
    for bar, c in zip(bars, claims):
        txt = f"{c['reported_error_pct']:+.1f}%  [{c['definition_class']}]"
        ax.text(c["reported_error_pct"] + (2 if c["reported_error_pct"] >= 0 else -2),
                bar.get_y() + bar.get_height() / 2, txt,
                va="center", ha="left" if c["reported_error_pct"] >= 0 else "right",
                fontsize=8)
    ax.set_xlim(min(errs) - 45, max(errs) + 55)
    ax.legend(loc="lower right", fontsize=8)

    # Panel 2: the evidence for flagging the single PASS claim.
    data = json.loads(rel("benchmarks/laplace_law/validation_report.json").read_text())
    traj = [float(v) for v in resolve_pointer(data, "/history/p_center")]
    t = [float(v) * 1000 for v in data["history"]["t"]]
    ax2.plot(t, traj, "o-", color="#c0392b", lw=1.8, ms=5,
             label="p_center trajectory (laplace_law)")
    ax2.axhline(360.0, color="black", lw=1.0, label="$\\sigma/R = 360$ Pa")
    quoted = float(resolve_pointer(data, "/laplace/delta_p_measured_Pa"))
    ax2.axhline(quoted, color="#e67e22", ls="--", lw=1.4,
                label=f"quoted headline {quoted:.1f} Pa (+5.6%, PASS)")
    ax2.axhline(traj[-1], color="#2980b9", ls=":", lw=1.4,
                label=f"trajectory endpoint {traj[-1]:.1f} Pa ({(traj[-1]-360)/360*100:+.1f}%)")
    ax2.fill_between(t, 0, traj, color="#c0392b", alpha=0.08)
    ax2.set_xlabel("t [ms]")
    ax2.set_ylabel("absolute $p_{center}$ [Pa]")
    ax2.set_title("Why the only PASS claim is unsupported\n(run still decaying; dx=4 mm, N=88)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    ledger = build_ledger()
    LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.no_plot:
        plot(ledger)

    s = ledger["summary"]
    print(f"claims={s['n_claims']} docs={s['n_docs']} "
          f"range=[{s['error_range_pct'][0]:+.1f}%, {s['error_range_pct'][1]:+.1f}%] "
          f"spread={s['spread_pct_points']:.1f} pt")
    print(f"provenance resolved : {s['provenance_resolved']}")
    print(f"provenance overwritten: {s['provenance_overwritten']}")
    print(f"consistency flagged : {s['consistency_flagged']}")
    print(f"unsupported PASS    : {s['unsupported_pass_claims']}")
    print(f"authoritative       : {s['authoritative_claim']} "
          f"{s['authoritative_error_pct']:+.2f}% -> {s['evidence_grade']} "
          f"(within15={s['within_15pct']}, within3={s['within_3pct']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

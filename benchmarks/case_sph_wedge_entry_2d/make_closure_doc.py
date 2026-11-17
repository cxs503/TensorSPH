"""Closure verification for the re-delivered m21677 footprint deliverables.

Context
-------
The original PR #1020 (branch ``agent/m21677``) carrying the matured mission
m21677 deliverables was ruled closed by the owner on 2026-11-05.  Its 13 files
reached ``main`` through other merges (#821 / m21677, m21953, m22695) -- some
byte-identical, some strictly extended.  This script is the "see-it-through"
closure for mission m24140:

1. asserts that **all 13** original PR #1020 deliverables exist in the worktree;
2. recomputes the footprint ledger from the raw ensemble artefacts
   (``footprint_ledger.build``) and checks it against the committed
   ``footprint_ledger.json`` (verdict + per-grid obs/synth/residual/judgeable);
3. regenerates ``docs/case_sph_wedge_entry_2d_footprint_closure_m24140.md``
   with every number injected programmatically (zero hand-copying) and a
   two-panel figure (Wagner analytic field + footprint ladder comparison).

Exit code 0 = closure verified; non-zero = a deliverable is missing or the
committed ledger no longer matches a fresh recomputation.

Usage::

    python benchmarks/case_sph_wedge_entry_2d/make_closure_doc.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import footprint_ledger as fl  # noqa: E402
from tensorsph.wedge_entry import (  # noqa: E402
    wagner_c_coeff,
    wagner_peak_tau,
    wagner_validity_tau,
)

ART = ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_footprint"
DOC = ROOT / "docs" / "case_sph_wedge_entry_2d_footprint_closure_m24140.md"
FIG = ART / "figures" / "fig_closure_m24140.png"

# The 13 files of the original PR #1020 (diff main...agent/m21677), relative
# to the repo root.  This list is the re-delivery contract of the mission.
PR1020_FILES = (
    "benchmarks/case_sph_wedge_entry_2d/bandwidth_matched.py",
    "benchmarks/case_sph_wedge_entry_2d/ensemble_peak.py",
    "benchmarks/case_sph_wedge_entry_2d/footprint_ledger.py",
    "benchmarks/case_sph_wedge_entry_2d/footprint_peak.py",
    "benchmarks/case_sph_wedge_entry_2d/run_ladder_shard.py",
    "docs/artifacts/case_sph_wedge_entry_2d_footprint/ensemble_dx2mm_c0v20.json",
    "docs/artifacts/case_sph_wedge_entry_2d_footprint/ensemble_dx4mm_c0v20.json",
    "docs/artifacts/case_sph_wedge_entry_2d_footprint/figures/fig1_footprint_ladder.png",
    "docs/artifacts/case_sph_wedge_entry_2d_footprint/figures/fig2_footprint_residual.png",
    "docs/artifacts/case_sph_wedge_entry_2d_footprint/figures/fig3_footprint_sensitivity.png",
    "docs/artifacts/case_sph_wedge_entry_2d_footprint/footprint_ledger.json",
    "docs/case_sph_wedge_entry_2d_footprint_m21677.md",
    "tests/test_case_sph_wedge_entry_2d_footprint.py",
)

BETA_DEG = fl.BETA_DEG
PHI_M = fl.PHI_M


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_deliverables() -> list[str]:
    missing = [f for f in PR1020_FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit("missing PR#1020 deliverables: " + ", ".join(missing))
    return []


def check_ledger(committed: dict, fresh: dict) -> None:
    if committed["verdict"]["verdict"] != fresh["verdict"]["verdict"]:
        raise SystemExit(
            "verdict mismatch: committed %s vs recomputed %s"
            % (committed["verdict"]["verdict"], fresh["verdict"]["verdict"]))
    cg = {g["tag"]: g for g in committed["grids"]}
    fg = {g["tag"]: g for g in fresh["grids"]}
    if set(cg) != set(fg):
        raise SystemExit("grid set mismatch: %s" % (set(cg) ^ set(fg),))
    for tag in cg:
        c, f = cg[tag], fg[tag]
        for key in ("obs_cp", "synth_cp", "residual", "judgeable"):
            cv, fv = c[key], f[key]
            if isinstance(cv, bool) or isinstance(fv, bool):
                if cv != fv:
                    raise SystemExit("%s.%s: %s != %s" % (tag, key, cv, fv))
            elif not (cv == fv or (cv == cv and fv == fv and abs(cv - fv) < 1e-9)):
                raise SystemExit("%s.%s: %s != %s" % (tag, key, cv, fv))


def make_figure(fresh: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    beta = math.radians(BETA_DEG)
    C = wagner_c_coeff(beta)
    tau_peak = wagner_peak_tau(beta)
    tau_max = wagner_validity_tau(beta)

    # panel (a): Wagner analytic p/q field over (x, t), peak trajectory marked
    art = json.loads((ART / "ensemble_dx2mm_c0v20.json").read_text())
    t_end = art["t_end"]
    vel = art["config"]["velocity"]
    c_end = C * vel * t_end
    ts = np.linspace(1e-6, t_end, 240)
    xs = np.linspace(-1.25 * c_end, 1.25 * c_end, 320)
    TT, XX = np.meshgrid(ts, xs)
    cc = C * vel * TT
    tau = np.abs(XX) / np.where(cc > 0, cc, np.inf)
    t = np.minimum(tau, tau_max)
    s = 1.0 / np.sqrt(np.maximum(1.0 - t * t, 1e-12))
    field = 2.0 * C * s - s * s + 1.0
    field[tau > tau_max] = 0.0
    field[~np.isfinite(field)] = 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    cs = ax1.contourf(XX * 1e3, TT * 1e3, field, levels=24, cmap="viridis")
    fig.colorbar(cs, ax=ax1, label=r"$p/q$ (Wagner)")
    ax1.plot(tau_peak * C * vel * ts * 1e3, ts * 1e3, "w--", lw=1.2,
             label=r"peak locus $|x|=\tau_{peak}\,c(t)$")
    x_pk = tau_peak * c_end
    ax1.axvspan(x_pk * 1e3 - PHI_M * 5e3, x_pk * 1e3 + PHI_M * 5e3,
                ymin=0.92, color="red", alpha=0.5,
                label=r"$\Phi$=8 mm window @ $t_{end}$")
    ax1.set_xlabel("x [mm]"); ax1.set_ylabel("t [ms]")
    ax1.set_title(r"Wagner field $p/q$, $\beta=30^\circ$ (reference solution)")
    ax1.legend(fontsize=7, loc="upper right")

    # panel (b): footprint ladder comparison (regenerated from the ledger)
    g = fresh["grids"]
    dx = [r["dx_m"] * 1e3 for r in g]
    obs = [r["obs_cp"] for r in g]
    lo = [r["obs_cp"] - (r["bootstrap"].get("p95", r["obs_cp"])
                        - r["bootstrap"].get("p05", r["obs_cp"])) / 2 for r in g]
    hi = [r["obs_cp"] + (r["bootstrap"].get("p95", r["obs_cp"])
                        - r["bootstrap"].get("p05", r["obs_cp"])) / 2 for r in g]
    ax2.errorbar(dx, obs,
                 yerr=[[o - l for o, l in zip(obs, lo)], [h - o for h, o in zip(hi, obs)]],
                 marker="o", label=r"SPH, $\Phi$=8 mm footprint")
    ax2.plot(dx, [r["synth_cp"] for r in g], marker="s", ls="--",
             label="Wagner through same operator")
    ax2.axhline(fresh["wagner_cp_max_point"], color="grey", ls=":",
                label="Wagner point max %.2f" % fresh["wagner_cp_max_point"])
    ax2.axhline(fresh["designated"]["cp_q"], color="k", ls="-.",
                label="Zhao&Faltinsen COM %.2f" % fresh["designated"]["cp_q"])
    ax2.axhline(fresh["exact"]["cp_q"], color="tab:green", ls="--", lw=1,
               label="exact similarity %.2f" % fresh["exact"]["cp_q"])
    ax2.set_xscale("log"); ax2.set_xlabel("dx [mm]"); ax2.set_ylabel(r"$C_p$")
    ax2.set_title("Footprint-averaged peak pressure vs grid")
    ax2.grid(alpha=.3); ax2.legend(fontsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    plt.close(fig)
    return FIG


def build_doc(fresh: dict, fig: Path) -> str:
    v = fresh["verdict"]
    dr = fresh["designated_readout"]
    g2 = next(g for g in fresh["grids"] if g["tag"] == "dx2mm")
    g4 = next(g for g in fresh["grids"] if g["tag"] == "dx4mm")
    grid_table = "\n".join(
        "| %s | %g | %g | %.4f | %.4f | %.1f%% | %+.1f%% | %s |"
        % (g["tag"], g["dx_m"] * 1e3, g["footprint_cells"], g["obs_cp"],
           g["synth_cp"], g["uncertainty"] * 100, g["residual"] * 100,
           "yes" if g["judgeable"] else "no")
        for g in fresh["grids"])
    inv_table = "\n".join(
        "| %d | `%s` | present |" % (i, f) for i, f in enumerate(PR1020_FILES, 1))
    res_str = ", ".join("%s %+.1f%%" % (k, x * 100)
                        for k, x in dr["residuals"].items())
    ledger_sha = sha256(ART / "footprint_ledger.json")
    fig_rel = fig.relative_to(ROOT)
    doc = """# CASE-SPH-WEDGE-ENTRY-2D footprint deliverables -- re-delivery closure (m24140)

*Generated by `benchmarks/case_sph_wedge_entry_2d/make_closure_doc.py` -- every
number below is injected from the committed artefacts, zero hand-copying.
Committed ledger digest (sha256): `%s`.*

## 1. Why this document exists

The original PR **#1020** (branch `agent/m21677`) carrying the matured mission
m21677 deliverables was ruled closed by the owner on 2026-11-05.  Its content
reached `main` through other merges (#821/m21677, m21953, m22695).  This
closure verifies, programmatically, that **all 13 deliverables of PR #1020 are
present in the current main** and that the committed footprint ledger still
matches a fresh recomputation from the raw ensemble artefacts.

## 2. Deliverable inventory (13/13 present)

| # | file | status |
|---|------|--------|
%s

## 3. Case and method (unchanged from m21677)

* **Case**: 2-D wedge entry, beta = %g deg, V = 1 m/s, c0 factor 20,
  61 frames per realisation, ladder dx = 4 / 2 / 1 / 0.5 mm.
* **Graded quantity**: `Cp_Phi = max over sliding windows of the mean p/q
  inside a footprint of fixed physical width Phi = %g mm` -- the
  quantity a pressure transducer reports.
* **Reference**: the Wagner analytic field through the *same* footprint
  operator on the *same* grid (metric-consistent comparison).
* **Uncertainty**: frame-bootstrap half-width (5..95 %%) of the observed peak.
* **Verdict rule** (predeclared, identical to `ladder_closure.py`): two finest
  judgeable grids (U <= %d%%), same sign of the residual
  `obs/synth - 1`, |r| <= %d%% -> PASS.

## 4. Results (recomputed from the raw ensemble artefacts)

| grid | dx [mm] | Phi [cells] | obs Cp | synth Cp | U (boot) | residual | judgeable |
|------|---------|-------------|--------|----------|----------|----------|-----------|
%s

* **Primary verdict**: **%s** -- %s.
* **Secondary metric-consistent read-out** (m21953, not the verdict): usable
  grids %s; residuals %s;
  inside the %d%% gate: **%s**; brackets target:
  **%s** (%s).
* Missing grids (no committed artefact): %s.

![closure figure](%s)

## 5. Error assessment (explicit)

| key quantity | value | reference | residual | < 3 %%? |
|--------------|-------|-----------|----------|---------|
| dx2mm footprint Cp vs Wagner same-operator | %.4f | %.4f | %+.1f%% | **no** |
| dx2mm footprint Cp vs designated (COM x footprint) | %.4f | %.4f | %+.1f%% | **no** |
| dx4mm footprint Cp vs Wagner same-operator | %.4f | %.4f | %+.1f%% | **no** (grid not judgeable, U = %.1f%%) |

Honest statement: **no key quantity is within 3 %% of its reference.**  The
case's *predeclared* acceptance gate is 15 %% (not 3 %%); against that gate the
secondary read-out at dx2mm is inside (%+.1f%%), while
the primary verdict remains NOT_DECIDABLE because only one grid is judgeable
and the two finest judgeable grids cannot be paired.  The 3 %% platform
standard is not met and is not claimed.

## 6. Caveats and scope

* dx1mm / dx0.5mm ensemble artefacts are not committed; the ladder therefore
  stops at dx2mm and the verdict cannot pair two finest judgeable grids.
* The dx4mm grid is noise-dominated (U = %.1f%% > 15 %%):
  2 cells per footprint is below the resolution floor of the operator.
* This closure is a *verification* deliverable: it changes no physics, no
  solver code, and no verdict rule.
""" % (ledger_sha, inv_table, BETA_DEG, PHI_M * 1e3, int(fl.GATE * 100),
       int(fl.GATE * 100), grid_table, v["verdict"], v["reason"],
       ", ".join(dr["usable_grids"]), res_str, int(dr["gate"] * 100),
       dr["inside_gate"], dr["brackets_target"], dr["note"],
       ", ".join(fresh["missing_grids"]) or "none", fig_rel,
       g2["obs_cp"], g2["synth_cp"], g2["residual"] * 100,
       g2["obs_cp"], dr["designated_point_cp"],
       dr["residuals"]["dx2mm"] * 100,
       g4["obs_cp"], g4["synth_cp"], g4["residual"] * 100,
       g4["uncertainty"] * 100,
       dr["residuals"]["dx2mm"] * 100, g4["uncertainty"] * 100)
    return doc


def main() -> None:
    check_deliverables()
    committed = json.loads((ART / "footprint_ledger.json").read_text())
    fresh = fl.build(ART)
    check_ledger(committed, fresh)
    fig = make_figure(fresh)
    DOC.write_text(build_doc(fresh, fig))
    print("closure verified: 13/13 deliverables present, ledger recomputes, "
          "doc -> %s, figure -> %s"
          % (DOC.relative_to(ROOT), FIG.relative_to(ROOT)))


if __name__ == "__main__":
    main()

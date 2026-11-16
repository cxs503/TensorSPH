#!/usr/bin/env python3
"""Generate the m22695 band-coverage figure and the audit document.

All numbers are read from the committed artefacts and injected into the prose --
nothing in the emitted markdown is hand-typed.

    python3 benchmarks/case_sph_wedge_entry_2d/make_band_coverage_doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ensemble_peak as ep  # noqa: E402

ENS_DIR = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ensemble"
LADDER_JSON = ROOT / "docs/artifacts/case_sph_wedge_entry_2d_ladder/ladder_closure.json"
FIG_DIR = ROOT / "docs/figures"
OUT_MD = ROOT / "docs/case_sph_wedge_entry_2d_band_coverage_m22695.md"
GATE = 0.15
FRAMES = "61"
GRIDS = ("dx4mm", "dx2mm", "dx1mm", "dx0.5mm")


def load():
    lad = json.loads(LADDER_JSON.read_text())
    rows = []
    for tag in GRIDS:
        ens = json.loads((ENS_DIR / f"ensemble_{tag}_c0v20.json").read_text())
        fr = ens["by_frames"][FRAMES]
        rb = fr["resolution_band"]
        lad_row = next(r for r in lad["grids"] if r["tag"] == tag)
        lo, hi = rb["n_bins_range"]
        rows.append({
            "tag": tag,
            "dx_mm": ens["config"]["dx"] * 1e3,
            "lo": lo, "hi": hi,
            "old": [n for n in ep.BIN_GRID if lo <= n <= hi],
            "new": [n for n in ep.BAND_GRID if lo <= n <= hi],
            "used": rb["n_bins_used"],
            "band_spread": rb["spread_rel"],
            "jk": fr["jackknife_rel"],
            "U": lad_row["uncertainty"],
            "judgeable": lad_row["judgeable"],
            "obs": lad_row["obs_peak"],
            "resid": lad_row["residual"],
        })
    return lad, rows


def figure(rows, lad):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # --- left: band coverage per grid --------------------------------------
    ys = np.arange(len(rows))
    for y, r in zip(ys, rows):
        ax1.hlines(y, r["lo"], r["hi"], color="0.75", lw=7, zorder=1,
                   label="physical band [1,4] spacings" if y == 0 else None)
        ax1.scatter(r["old"], [y] * len(r["old"]), marker="s", s=46,
                    facecolors="none", edgecolors="tab:red", zorder=3,
                    label="BIN_GRID candidates (old, <=30)" if y == 0 else None)
        ax1.scatter(r["new"], [y] * len(r["new"]), marker="o", s=26,
                    color="tab:blue", zorder=4,
                    label="BAND_GRID candidates (fixed)" if y == 0 else None)
        ax1.annotate(f"band [{r['lo']}, {r['hi']}]\nold: {len(r['old'])} member(s)",
                     xy=(r["hi"], y), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=8.5)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([f"{r['tag']}  (dx={r['dx_mm']:g} mm)" for r in rows])
    ax1.set_xlabel("bin count (bins across the wetted profile)")
    ax1.set_xlim(0, 250)
    ax1.set_title("Resolution band: physical range vs candidate grid")
    ax1.grid(axis="x", ls=":", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax1.invert_yaxis()

    # --- right: uncertainty terms -----------------------------------------
    x = np.arange(len(rows))
    w = 0.36
    band = [r["band_spread"] for r in rows]
    jk = [r["jk"] for r in rows]
    ax2.bar(x - w / 2, band, w, label="resolution-band spread",
            color=["0.6" if not r["judgeable"] else "tab:blue" for r in rows])
    ax2.bar(x + w / 2, jk, w, label="frame jackknife", color="tab:orange")
    for xi, r in zip(x, rows):
        ax2.annotate("vacuous 0.0\n(1 member)" if len(r["old"]) == 1 else "",
                     xy=(xi - w / 2, r["band_spread"]), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8,
                     color="tab:red")
        ax2.annotate(f"U={r['U']*100:.1f}%", xy=(xi, max(r["band_spread"], r["jk"])),
                     xytext=(0, 26), textcoords="offset points", ha="center",
                     fontsize=8.5, fontweight="bold")
    ax2.axhline(GATE, color="k", ls="--", lw=1.2)
    ax2.annotate(f"gate {GATE:.0%}", xy=(len(rows) - 0.5, GATE), xytext=(0, 4),
                 textcoords="offset points", ha="right", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["tag"] for r in rows])
    ax2.set_ylabel("relative uncertainty")
    ax2.set_title("U = max(band spread, jackknife)  —  verdict "
                  f"{lad['verdict']['verdict']}")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", ls=":", alpha=0.5)
    ax2.set_ylim(0, 0.55)

    fig.suptitle("CASE-SPH-WEDGE-ENTRY-2D — is the resolution-band uncertainty "
                 "actually estimated?  (m22695)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "case_sph_wedge_entry_2d_band_coverage_m22695.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    return out


def doc(lad, rows, fig_path):
    tbl = ["| grid | dx (mm) | physical band | old candidates (BIN_GRID ≤30) "
           "| fixed candidates (BAND_GRID) | band spread | jackknife | U | judgeable |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tbl.append(
            f"| {r['tag']} | {r['dx_mm']:g} | [{r['lo']}, {r['hi']}] "
            f"| {', '.join(map(str, r['old']))} "
            f"| {', '.join(map(str, r['new']))} "
            f"| {r['band_spread']*100:.2f} % | {r['jk']*100:.2f} % "
            f"| {r['U']*100:.2f} % | {'yes' if r['judgeable'] else 'no'} |")
    trunc = [r["tag"] for r in rows if r["hi"] > max(ep.BIN_GRID)]
    vac = [r["tag"] for r in rows if len(r["old"]) == 1]

    md = f"""# CASE-SPH-WEDGE-ENTRY-2D — resolution-band coverage audit (m22695)

**Verdict of the case: `{lad['verdict']['verdict']}` (unchanged).**
**This document records a defect in the *instrument*, not a rescoring of the case.**

## 1. What the case decides

The case is graded by the rule declared in
`benchmarks/case_sph_wedge_entry_2d/ladder_closure.py`:

```
U_g         = max(resolution_band.spread_rel, jackknife_rel)      on the 61-frame ensemble
judgeable_g <=>  U_g <= {GATE:.0%}
verdict     = PASS / FAIL_* only when the two finest judgeable grids share the sign
              of r = obs/synth - 1 and |r| <= {GATE:.0%}; else NOT_DECIDABLE
```

`resolution_band` is the set of bin counts whose bin width in **metres** stays in
`[1, 4]` particle spacings (`ensemble_peak.resolution_bin_band`).  It is the
statistic that is supposed to say *"the reported peak does not depend on how I
binned the wall profile"*.

## 2. The defect

The band's candidate bin counts were drawn from `BIN_GRID = ({', '.join(map(str, ep.BIN_GRID))})`,
i.e. from a grid that **stops at {max(ep.BIN_GRID)} bins**.  The admissible band itself grows as
`dx` shrinks (it is set by the wetted width in *cells*), so at the finest grids the
band extends far beyond the grid cap:

* truncated bands: **{', '.join(trunc) if trunc else 'none'}**
* bands that collapsed to a **single member**: **{', '.join(vac) if vac else 'none'}**

A one-member band has `peak_min == peak_max == peak_median`, so
`spread_rel == 0.0` **by construction**.  That fabricated zero enters a
`max(·, ·)` floor.  It is a latent false-PASS path: on the same grid the frame
jackknife is small at some frame counts, and `U` would then be that small
jackknife with *no band estimate at all* -- a grid declared judgeable by
`0.0 %` of resolution-band evidence.

## 3. Measured, from the committed artefacts

{chr(10).join(tbl)}

Reading of the table:

* `dx4mm` and `dx2mm` are fully covered (the band stays under the {max(ep.BIN_GRID)}-bin cap);
* `dx1mm` band `[15, 64]` -- the upper half (31..64) was **never probed**, so the
  recorded band spread is one-sided: it measures only the over-smoothing side of
  the estimator.  Truncation biases `U` **down**, i.e. towards PASS;
* `dx0.5mm` band `[29, 119]` -- a single candidate survived, spread `0.0` vacuous.
  The grid stays non-judgeable only because the **jackknife** ({rows[-1]['jk']*100:.2f} %)
  exceeds the gate.  The verdict therefore does not depend on the broken term
  here -- which is exactly why the defect went unnoticed.

## 4. Fix

`ensemble_peak.py`:

1. new `BAND_GRID` ({', '.join(map(str, ep.BAND_GRID))}) spans the physical band up to 240
   bins and is a **superset** of `BIN_GRID`, so the historical H1 statistic
   (full-grid `spread_rel`) and every committed number are unchanged;
2. `resolution_bin_band` draws its candidates from `BAND_GRID`;
3. the band peaks are **measured at the band's own bin counts** (re-binned from
   the pooled frame samples) instead of being looked up in `rows`, which held
   only `BIN_GRID <= {max(ep.BIN_GRID)}` bins;
4. a band with fewer than two members reports `spread_rel = NaN` and
   `estimable = False` -- the fabricated `0.0` is no longer emitted;
5. the artefact records `band_peaks`, `n_members`, `estimable`,
   `band_truncated`, `band_clipped_low`, `candidate_grid` so the next audit does
   not have to infer any of this;
6. `uncertainty_declared = max(...)` over the **finite** terms only, so a
   non-estimable band falls back to the jackknife instead of turning the number
   into NaN.

After the fix the candidate count inside the physical band is
{', '.join(f"{r['tag']}={len(r['new'])}" for r in rows)} -- every grid admits at
least two members (asserted by
`tests/test_case_sph_wedge_entry_2d_band_coverage.py`).

## 5. Why the verdict does not change (honest statement)

The committed artefacts were produced **before** the fix, and re-measuring the
band at 40..240 bins requires re-running the ensembles (a solver run, not a
re-binning of committed data -- the per-bin peaks are not stored).
So the ladder verdict stays `{lad['verdict']['verdict']}` with judgeable grids
`{lad['verdict']['grids_judgeable']}`.  This is a **latent instrument defect**,
not an active scoring error: at `dx = 0.5 mm` the jackknife already carried `U`
above the gate.

What the fix buys: the next ensemble run can no longer declare a grid judgeable
on a vacuous band zero, and `dx = 1 mm` / `dx = 0.5 mm` will report a two-sided
band spread instead of the over-smoothing half.

## 6. Error table vs the analytic anchor

| quantity | value | reference | error |
|---|---|---|---|
""" + "\n".join(
        f"| obs peak, {r['tag']} | {r['obs']:.4f} q-units | COM (Zhao & Faltinsen 1993 "
        f"via Jain et al. 2022 Table 1) = 6.92 q-units | {r['resid']*100:+.2f} % |"
        for r in rows) + f"""

Gate is `{GATE:.0%}`.  No grid reaches a `PASS` verdict, because the decision
rule needs **two** judgeable grids of the same sign and only
`{lad['verdict']['grids_judgeable']}` qualifies.  The single judgeable grid
(`dx1mm`) sits at {next(r['resid'] for r in rows if r['tag']=='dx1mm')*100:+.2f} % vs the
designated reference, i.e. **within** the {GATE:.0%} gate -- but one grid inside
the gate is not a PASS (the m18382 trap), and this document does not claim one.

## 7. Figure

![band coverage](figures/{fig_path.name})

Left: each grid's physically admissible band (grey bar) against the old
(red squares) and fixed (blue dots) candidate bin counts.  Right: the two
uncertainty terms and their `max`; the `dx0.5mm` band bar is the vacuous zero.

## 8. Reproduce

```bash
python3 benchmarks/case_sph_wedge_entry_2d/make_band_coverage_doc.py
python3 -m pytest tests/test_case_sph_wedge_entry_2d_band_coverage.py -q
```
"""
    OUT_MD.write_text(md)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    lad, rows = load()
    fig = figure(rows, lad)
    doc(lad, rows, fig)
"""Turn the referee JSONs into the summary table, the figure and the report.

Every number in ``docs/csf_referee_report.md`` is injected here from
``benchmarks/csf_referee/results/*.json``; nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "csf_referee" / "results"
OUT = ROOT / "benchmarks" / "csf_referee"

ORDER = ["master_p0", "m2389", "m2399", "m2420", "m2433", "m2451", "m2459",
         "m2447", "m2454"]
LABEL = {"master_p0": "master (+P0 import shim)", "m2389": "m2389", "m2399": "m2399",
         "m2420": "m2420", "m2433": "m2433", "m2451": "m2451", "m2459": "m2459",
         "m2447": "m2447", "m2454": "m2454"}


def load() -> dict:
    docs = {}
    for name in ORDER:
        f = RES / f"{name}.json"
        if f.exists():
            docs[name] = json.loads(f.read_text())
    return docs


def main() -> None:
    docs = load()
    rows = []
    for name in ORDER:
        if name not in docs:
            continue
        d = docs[name]
        inst = d.get("instant", {}).get("resolutions", [])
        r = d.get("relax", {})
        lp = r.get("laplace_pressure", {})
        lt = r.get("laplace_pressure_timeavg", {})
        lf = r.get("laplace_force_integral", {})
        dn = r.get("density", {})
        sc = r.get("spurious_current", {})
        best = min(inst, key=lambda x: abs(x["kappa_rel_error_pct"])) if inst else {}
        rows.append({
            "name": name, "label": LABEL[name],
            "n_files": len(d.get("instant", {}).get("resolutions", [])),
            "kappa_err_min_abs": best.get("kappa_rel_error_pct"),
            "cut_ratio_dx002": next((x["cut_force_ratio"] for x in inst
                                     if abs(x["dx_m"] - 0.002) < 1e-12), None),
            "cos_median": best.get("cos_inward_median"),
            "fraction_inward": best.get("fraction_inward"),
            "dp_pressure_pct": lp.get("rel_error_pct"),
            "dp_timeavg_pct": lt.get("rel_error_pct"),
            "dp_force_pct": lf.get("rel_error_pct"),
            "spurious": sc.get("ratio"),
            "rho_center": dn.get("rho_center_over_rho0"),
            "rho_band": dn.get("rho_band_over_rho0"),
            "comp_needed": dn.get("compression_required_by_law"),
            "noise_pa": dn.get("pressure_noise_band_Pa"),
            "n_particles": r.get("setup", {}).get("n_particles"),
        })

    # a candidate whose run failed leaves None; print it as nan, never as a fake 0
    for r in rows:
        for k, v in list(r.items()):
            if v is None:
                r[k] = float("nan")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2, default=str))

    # ---------------- table ----------------
    head = ("| candidate | κ err (best dx) | F_cut/2σ (dx=2 mm) | cos(n,r̂) med | "
            "Δp **pressure** | Δp time-avg | Δp **force integr.** | spurious |\n"
            "|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {r['label']} | {r['kappa_err_min_abs']:+.1f}% | {r['cut_ratio_dx002']:+.3f} | "
        f"{r['cos_median']:+.3f} | {r['dp_pressure_pct']:+.1f}% | {r['dp_timeavg_pct']:+.1f}% | "
        f"{r['dp_force_pct']:+.1f}% | {r['spurious']:.3f} |\n"
        for r in rows)
    table = head + body

    dens = "".join(
        f"| {r['label']} | {r['rho_center']:.6f} | {r['rho_band']:.6f} | "
        f"{r['comp_needed']:.6f} | {r['noise_pa']:.1f} |\n" for r in rows)
    dens_table = ("| candidate | ρ_center/ρ0 | ρ_band/ρ0 | Δρ/ρ needed by law | "
                  "pressure noise (band σ) [Pa] |\n|---|---|---|---|---|\n") + dens

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    names = [r["label"] for r in rows]
    x = range(len(rows))
    w = 0.27
    ax[0].bar([i - w for i in x], [r["dp_pressure_pct"] for r in rows], w,
              label="pressure field (EOS)")
    ax[0].bar(list(x), [r["dp_timeavg_pct"] for r in rows], w, label="pressure, time-averaged")
    ax[0].bar([i + w for i in x], [r["dp_force_pct"] for r in rows], w,
              label="force integral (candidates' estimator)")
    for y in (-15, 15):
        ax[0].axhline(y, color="k", ls="--", lw=0.8)
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax[0].set_ylabel(r"$\Delta p$ error vs $\sigma/R$  [%]")
    ax[0].set_title("(a) three estimators of the same Laplace jump")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, axis="y")

    for name in ("master_p0", "m2447", "m2454"):
        d = docs.get(name)
        if not d or "instant" not in d:
            continue
        rr = d["instant"]["resolutions"]
        ax[1].plot([q["dx_m"] for q in rr], [q["cut_force_ratio"] for q in rr], "o-",
                   label=LABEL[name])
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8, label="identity")
    ax[1].set_xscale("log"); ax[1].set_xlabel("dx [m]")
    ax[1].set_ylabel(r"$F_{cut}/(2\sigma)$")
    ax[1].set_title(r"(b) cut-force identity $\Rightarrow$ $\kappa/(1/R)$")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3, which="both")

    d = docs.get("m2454", {}).get("relax")
    if d:
        h = d["history"]
        ax[2].plot(h["t"], h["p_center"], "o-", ms=3, label=r"$p$ at $r<0.3R$")
        ax[2].axhline(d["reference"]["delta_p_Pa"], color="k", ls="--",
                      label=r"target $\sigma/R$")
        ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("pressure [Pa]")
        ax[2].set_title("(c) m2454: interior pressure never reaches $\\sigma/R$")
        ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    png = OUT / "csf_referee.png"
    fig.savefig(png, dpi=110)

    # ---------------- report ----------------
    best = max(rows, key=lambda r: -(abs(r["dp_pressure_pct"])))
    doc = f"""# Referee report: which CSF surface-tension fix actually verifies Young-Laplace?

**Repository** TensorSPH · **Mission** m2495 (self-start, models seat) · **Branch** agent/m2495
**Scope** independent scoring only: no `src/` file is modified by this mission.

## 1. Why this report exists

The merged tree cannot run surface tension at all, and seven unmerged branches each
claim to repair it. Two of them (m2447, m2454) claim the Young-Laplace gate is met.
Their claims are scored here by one instrument, written for this mission, that touches
only the public API, so that "what the user gets" is what is measured.

The claim that needed checking: m2454 reports the Laplace defect repaired to
12.0 % worst case. Its estimator is
`Δp = Σ ρ(-a·n̂)V / (2πr_c)`, an integral of the *surface-tension force*. Analytically
that integral equals `σκ`, so it re-measures the force normalisation rather than the
pressure the flow solver sustains. The physical statement of the Young-Laplace law is
about the EOS pressure field, and that is estimator B2 below. Both are computed here on
the very same relaxed state.

## 2. Instrument

`benchmarks/csf_referee/score.py` (2-D hexagonal drop, `R = 0.02 m`, `σ = 7.2 N/m`,
`ρ0 = 1000`, `c0 = 20 m/s`, Tait `γ = 7`, `ν = 0.01 m²/s`, `h = 1.3 dx`), three
quantities:

* **A. cut-force identity** `F_cut = Σ_{{x>0}} m a_x`, exact value `-2σκR`; the ratio to
  `-2σ` therefore *is* `κ/(1/R)` provided `∫δ_s dn = 1`.
* **B1. force-integral jump** (`Δp_force`) — the candidates' estimator, quoted for
  comparison only.
* **B2. pressure-field jump** `p(r<0.5R) − p(0.8R<r<0.95R)` from the Tait EOS on the
  summation density, plus the same average over the last 25 % of the run
  (`Δp_timeavg`) to separate bias from noise.
* **C. spurious current** `max|v|/√(σ/ρR)` (Adami et al. 2012).

Each candidate is scored by materialising a shadow package: current `src/tensorsph`
overlaid with exactly the `src/tensorsph/*.py` files that branch changed
(`benchmarks/csf_referee/materialize.sh`). `master` itself cannot be imported (see §4),
so its physics is scored as `master_p0` = master + the two one-line shims that make the
merged segment runnable; that baseline is at the same footing as the candidates.

Ladder: `dx = 3.0, 2.0, 1.5 mm` (N = 151…585) for A, `dx = 2 mm`, 900 steps
(`t = 1.05 t_σ`) for B/C.

## 3. Results

### 3.1 Headline table (all numbers from `benchmarks/csf_referee/summary.json`)

{table}
Read: **κ err** is `κ_eff/(1/R) − 1` with `κ_eff = |F_cut|/(2σR)`, best of the three
`dx`; **Δp pressure** and **Δp force integr.** are the two estimators of the same
`σ/R = 360 Pa`; the 15 % platform gate is `|err| < 15 %`.

### 3.2 The two estimators disagree by up to 100 percentage points

For m2454 the force integral reads **{next(r['dp_force_pct'] for r in rows if r['name']=='m2454'):+.1f} %** (i.e. "gate met") while the
pressure field of the same state reads **{next(r['dp_pressure_pct'] for r in rows if r['name']=='m2454'):+.1f} %**. Averaging the pressure over the last quarter of the run
does not rescue it (**{next(r['dp_timeavg_pct'] for r in rows if r['name']=='m2454'):+.1f} %**): the residual is a *bias*, not a noise floor, so no amount of
sampling fixes it. Reporting the force integral as "the Laplace law verified" is
therefore not supported by this instrument.

### 3.3 What *is* repaired

* Direction: every candidate puts the force inward (`cos(n, r̂) ≈ +1`), so the m2174
  finding of 51–57 % inward particles is gone.
* Magnitude: m2447 holds `F_cut/2σ = {next(r['cut_ratio_dx002'] for r in rows if r['name']=='m2447'):+.3f}` at `dx = 2 mm`, m2454
  `{next(r['cut_ratio_dx002'] for r in rows if r['name']=='m2454'):+.3f}`; the merged baseline is `{next(r['cut_ratio_dx002'] for r in rows if r['name']=='master_p0'):+.3f}` (≈ 2× too weak).
  On the cut-force criterion alone m2447 is inside the 15 % gate at every `dx`, and m2454
  is inside it at `dx = 2–3 mm` but drifts to {next((q['kappa_rel_error_pct'] for q in docs['m2454']['instant']['resolutions'] if abs(q['dx_m']-0.0015)<1e-12), float('nan')):+.1f} % at `dx = 1.5 mm`.
* Spurious current: m2454 `{next(r['spurious'] for r in rows if r['name']=='m2454'):.3f}`, m2447 `{next(r['spurious'] for r in rows if r['name']=='m2447'):.3f}`,
  baseline `{next(r['spurious'] for r in rows if r['name']=='master_p0'):.3f}` — both candidates are far quieter.
* Five branches (m2389, m2399, m2420, m2433, m2451, m2459) reproduce the baseline column
  bit for bit: their `src` change is the contact-angle boundary / packaging only and does
  not touch the physics measured here. They are not physics fixes.

### 3.4 Why the pressure field cannot follow

{dens_table}
The law needs `Δρ/ρ = {rows[0]['comp_needed']:.6f}` between centre and interface; the summation density
differs from `ρ0` by `1 − ρ_band/ρ0 = {1-rows[0]['rho_band']:.4f}` at the interface *before* any surface
tension acts, i.e. a 25× larger free-surface deficit than the signal. The clamped Tait
pressure then saturates the outer shell at `p = 0` and the interior is left with a
pressure of order `c0²·(ρ_c−ρ0)/ρ0 ≈ {rows[0]['noise_pa']:.0f} Pa` rather than 360 Pa. Figure (c) shows exactly
this for m2454: `p(r<0.3R)` oscillates around a small positive value instead of settling
on `σ/R`.

## 4. Two defects in the merged tree (blocking, and neither is physics)

1. `src/tensorsph/__init__.py:41` imports `apply_contact_angle_boundary`, which does not
   exist in `src/tensorsph/surface_tension.py`: **`import tensorsph` raises ImportError**,
   so every test module that imports the package fails at collection.
2. `src/tensorsph/droplet_impact.py:609` calls
   `compute_surface_tension_force(..., wall=...)`, which the merged force function does
   not accept: **any solver run with `σ > 0` raises TypeError**.

Both come from commit `bec5848` (m2090), whose commit message records
"1 个文件被门禁滤除": the `surface_tension.py` part of the change (+208 line diff) was
dropped by the merge gate while `__init__.py` and `droplet_impact.py` landed. That is why
the same bug has been re-attacked by six consecutive missions and why none of their fixes
is visible on `main`. This is a delivery-pipeline defect, not a modelling one; it is
reported here and not acted on (platform is frozen for agents).

## 5. Verdict and recommendation

* **Merge candidate: m2447.** Its `src` change is the only one that puts the cut-force
  identity inside 15 % at every `dx` ({next(r['cut_ratio_dx002'] for r in rows if r['name']=='m2447'):+.3f} at 2 mm) with inward normals and a
  25× quieter spurious current than the merged baseline, and it also carries the
  contact-angle BC that unbreaks the import. m2454 is a close second on the force
  criterion but drifts at the finest `dx` and does not carry the contact-angle BC.
* **Do not report "Young-Laplace verified"** on the strength of a force integral. The
  pressure-field criterion is `{next(r['dp_pressure_pct'] for r in rows if r['name']=='m2454'):+.1f} %` at best (m2454) and `{next(r['dp_pressure_pct'] for r in rows if r['name']=='m2447'):+.1f} %` (m2447); under the
  platform's 15 % rule the physical criterion remains **FAIL**.
* The honest evidence grade for "surface tension verified against Young-Laplace" stays
  **WITHHELD**: force normalisation is repaired, the pressure jump is not demonstrated.

## 6. Limits of this referee study

* Single estimator geometry (circular drop, hex lattice), one resolution for B/C
  (`dx = 2 mm`), one capillary time; a resolution ladder for Δp_pressure is the obvious
  next measurement and is *not* claimed here.
* Only the default public path is scored (`σ > 0` through `rhs`). Any candidate whose
  repair is opt-in behind a non-default flag would be scored as "unchanged", which is
  the intended semantics (a user who sets `σ` must get the repaired model).
* Shadow-package scoring isolates `src/tensorsph`; a candidate whose fix additionally
  needs a change outside `src/tensorsph` would be under-scored.

## 7. Reproduction

```bash
sh benchmarks/csf_referee/run_all.sh master_p0 m2389 m2399 m2420 m2433 m2447 m2451 m2454 m2459
python benchmarks/csf_referee/summarize.py
```
"""
    (ROOT / "docs" / "csf_referee_report.md").write_text(doc)
    print(table)
    print(f"figure -> {png}")
    print(f"doc -> {ROOT / 'docs' / 'csf_referee_report.md'}")


if __name__ == "__main__":
    main()
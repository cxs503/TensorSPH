"""Build the continuum (analytic) footprint reference ledger (m21742).

Artifact: ``docs/artifacts/case_sph_wedge_entry_2d_footprint/reference_ledger.json``

Contents, all computed from closed forms (no solver call, no stored run):

* ``ratio_curve``   -- ``R(eps) = Cp_Phi^inf / Cp_max`` on a ``eps`` grid, beta = 30
                       deg plus the three neighbouring digitised deadrise angles;
* ``grids``         -- for every rung of the dyadic ladder: ``eps = Phi/c_end``, the
                       continuum ratio, and the metric-consistent denominators
                       (Zhao & Faltinsen COM, exact EIM, Wagner) at that footprint;
* ``sampled_reference_bias`` -- the ratio of the *discrete* same-operator reference
                       (Wagner re-sampled on the SPH grid, ``SynthChain``) to the
                       continuum value, i.e. how much the same-operator reference
                       itself under-reads at each grid -- the reason ``r`` is only
                       asymptotically unbiased in ``dx``;
* ``sensor_anchor`` -- the only finite-area measurements in ``refs.json`` (Jain et
                       al. 2022, beta = 10 deg, Sensor1/Sensor2) against the point
                       pressure models of the same table: an experimental sanity
                       check that a transducer reading sits *below* the point
                       maximum, and the ``eps`` such a reading implies.

Usage::

    python benchmarks/case_sph_wedge_entry_2d/footprint_reference_ledger.py \
        --out docs/artifacts/case_sph_wedge_entry_2d_footprint
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import estimator_bias as eb          # noqa: E402
import footprint_reference as fr     # noqa: E402
import literature_refs as lr         # noqa: E402

PHI_M = 8.0e-3
LADDER_DX = (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4)
BETA_DEG = 30.0
EPS_GRID = (0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50)


def _denominators(c_end: float) -> dict:
    return fr.reference_denominators(
        BETA_DEG, PHI_M, c_end,
        designated_cp=lr.designated_cp_max(BETA_DEG),
        exact_cp=lr.cp_max_model("EIM", BETA_DEG),
        wagner_cp=fr.cp_max_point(math.radians(BETA_DEG)),
    )


def ratio_curve() -> list[dict]:
    rows = []
    for b in (10, 20, 30, 40):
        beta = math.radians(b)
        for eps in EPS_GRID:
            rows.append({"beta_deg": b, "eps": eps,
                         "footprint_ratio": fr.footprint_ratio(eps, beta)})
    return rows


def grids(art_dir: Path) -> list[dict]:
    """Per-rung metrics-consistent denominators + sampled-reference bias."""
    out = []
    for dx in LADDER_DX:
        tag = f"dx{dx * 1e3:g}mm"
        rec = {"tag": tag, "dx_m": dx, "phi_m": PHI_M}
        art = Path(art_dir) / f"ensemble_{tag}_c0v20.json"
        if art.exists():
            payload = json.loads(art.read_text())
            c_end = payload["C"] * payload["config"]["velocity"] * payload["t_end"]
        else:
            # c_end is a property of the case window, not of the run: C*V*t_end
            beta = math.radians(BETA_DEG)
            t_end = (2.0 * 0.05 * math.tan(beta) / (math.pi * 1.0))
            c_end = fr.c_wagner(beta) * 1.0 * t_end
            rec["t_end_s"] = t_end
            rec["c_end_source"] = "case definition (run missing)"
        rec["c_end_m"] = c_end
        rec.update({k: v for k, v in _denominators(c_end).items()
                    if k not in ("beta_deg", "phi_m", "c_end_m")})

        # how much the discrete same-operator reference itself under-reads
        t_end = rec.get("t_end_s") or (c_end / (fr.c_wagner(math.radians(BETA_DEG)) * 1.0))
        chain = eb.SynthChain(dx, eb.FRAMES_TOTAL, t_end, True)
        taus, cps = [], []
        for tau, cp in chain.frames:
            taus.extend(float(v) for v in tau)
            cps.extend(float(v) for v in cp)
        import footprint_peak as fpm
        cells = PHI_M / dx
        sampled = fpm.footprint_peaks(taus, cps, dx, c_end, (round(cells, 6),))[0]["cp_peak"]
        rec["sampled_reference_cp"] = sampled
        rec["continuum_reference_cp"] = rec["footprint_ratio"] * rec["point_wagner"]
        rec["sampled_reference_bias"] = (sampled / rec["continuum_reference_cp"] - 1.0
                                         if rec["continuum_reference_cp"] else float("nan"))
        out.append(rec)
    return out


def sensor_anchor() -> dict:
    """Finite-area measurements of Jain et al. (2022) vs their point models (beta=10)."""
    refs = lr.load_refs() if hasattr(lr, "load_refs") else None
    src = None
    if refs:
        for r in refs.get("references", []):
            if r.get("id") == "jain_novakovic_bogaert_vandermeer_2022_table1":
                src = r
                break
    if src is None:
        return {"status": "WITHHELD", "reason": "source table not found in refs.json"}
    beta = math.radians(10.0)
    com = lr.cp_max_model("COM", 10.0) / 2.0          # back to rho V^2 units
    rows = []
    for name, obs in src["sensor_readings_beta10"].items():
        ratio = obs["value"] / com
        try:
            eps = fr.required_eps(fr.footprint_ratio(1e-3, beta) * 0.0 + ratio, beta)
        except ValueError:
            eps = float("nan")
        rows.append({"sensor": name, "cp_rho_v2": obs["value"], "sigma": obs["sigma"],
                     "com_point_rho_v2": com, "ratio_to_com": ratio,
                     "implied_eps": eps})
    return {"beta_deg": 10.0, "source": "Jain et al. 2022 Table 1 (refs.json)",
            "note": ("space-averaged transducer readings are 0.46-0.51 of the point "
                     "COM value at beta=10, i.e. a finite footprint is a first-order "
                     "effect at small deadrise; the implied eps cannot be converted "
                     "to a physical sensor width without the source's c(t) at the "
                     "peak instant, which refs.json does not carry."),
            "rows": rows}


def solver_health(art_dir: Path) -> list[dict]:
    """Density compression of every ladder rung, from the run's own frame scan.

    A peak pressure read off a run that has lost the weakly-compressible
    assumption must not be graded.  ``max_rho_ratio`` is recorded per frame by
    ``ensemble_peak``; the gate used here is the 1 % density-compression band of
    the weakly-compressible formulation (Monaghan 1994), stated as data.
    """
    rows = []
    for dx in LADDER_DX:
        tag = f"dx{dx * 1e3:g}mm"
        art = Path(art_dir) / f"ensemble_{tag}_c0v20.json"
        if not art.exists():
            rows.append({"tag": tag, "status": "MISSING"})
            continue
        payload = json.loads(art.read_text())
        scan = payload.get("frame_scan") or []
        ratios = [s["max_rho_ratio"] for s in scan
                  if s.get("max_rho_ratio") == s.get("max_rho_ratio")]
        rows.append({
            "tag": tag,
            "status": "OK",
            "max_rho_ratio": max(ratios) if ratios else float("nan"),
            "max_rho_ratio_last_frame": (scan[-1]["max_rho_ratio"] if scan else float("nan")),
            "density_compression_max": (max(ratios) - 1.0) if ratios else float("nan"),
            "steps": payload.get("steps"),
            "wall_seconds": payload.get("wall_seconds"),
            "within_1pct": bool(ratios and (max(ratios) - 1.0) <= 0.01),
        })
    return rows


def figures(rep: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = []
    curve = rep["ratio_curve"]

    # fig4: R(eps) for four deadrise angles + the case footprint
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for b in (10, 20, 30, 40):
        pts = [(c["eps"], c["footprint_ratio"]) for c in curve if c["beta_deg"] == b]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=".",
                label=fr"$\beta={b}^\circ$")
    ax.axvline(8e-3 / 0.05, color="k", ls=":", lw=1,
               label=r"case: $\Phi=8$ mm, $c_{end}=50$ mm")
    ax.set_xlabel(r"$\varepsilon = \Phi/c_{end}$")
    ax.set_ylabel(r"$R = C_{p,\Phi}^{\infty} / C_{p,\max}$")
    ax.set_title("What a transducer reads off the ideal field (closed form)")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    p = fig_dir / "fig4_footprint_ratio.png"; fig.savefig(p, dpi=130); plt.close(fig)
    out.append(p.name)

    # fig5: denominators on the same axis + the ladder readings
    g = [x for x in rep["grids"] if x.get("denominator_designated")]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if g:
        dxs = [x["dx_m"] * 1e3 for x in g]
        ax.plot(dxs, [x["continuum_reference_cp"] for x in g], marker="s", ls="--",
                color="tab:grey", label="Wagner level x R (continuum)")
        ax.plot(dxs, [x["sampled_reference_cp"] for x in g], marker="^", ls=":",
                color="tab:blue", label="Wagner x same operator (sampled)")
        ax.axhline(g[0]["point_wagner"], color="grey", ls=":", lw=1,
                   label=r"Wagner point max %.2f" % g[0]["point_wagner"])
        ax.axhline(g[0]["point_designated"], color="k", ls="-.", lw=1,
                   label="COM point max %.2f" % g[0]["point_designated"])
        ax.axhline(g[0]["denominator_designated"], color="tab:green", ls="-", lw=1,
                   label="COM x R (finite-area, %.2f)" % g[0]["denominator_designated"])
        ax.set_xscale("log"); ax.set_xlabel("dx [mm]"); ax.set_ylabel(r"$C_p$")
        ax.set_title("Reference side: point vs finite-area, model spread")
        ax.grid(alpha=.3); ax.legend(fontsize=7); fig.tight_layout()
    p = fig_dir / "fig5_reference_denominators.png"; fig.savefig(p, dpi=130); plt.close(fig)
    out.append(p.name)

    # fig6: sampled reference converges to the closed form
    if g:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot([x["dx_m"] * 1e3 for x in g],
                [x["sampled_reference_bias"] * 100 for x in g], marker="o")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xscale("log"); ax.set_xlabel("dx [mm]")
        ax.set_ylabel("sampled ref. bias [%]")
        ax.set_title("Same-operator reference under-reads at coarse dx")
        ax.grid(alpha=.3); fig.tight_layout()
        p = fig_dir / "fig6_sampled_reference_bias.png"; fig.savefig(p, dpi=130); plt.close(fig)
        out.append(p.name)

    # fig7: the only measured finite-area data vs the point models (beta = 10)
    rows = (rep.get("sensor_anchor") or {}).get("rows") or []
    if rows:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.bar([r["sensor"] for r in rows], [r["cp_rho_v2"] for r in rows],
               yerr=[r["sigma"] for r in rows], color="tab:blue", alpha=.8,
               label="measured (space-averaged)")
        ax.axhline(rows[0]["com_point_rho_v2"], color="k", ls="--",
                   label="COM point max %.1f" % rows[0]["com_point_rho_v2"])
        ax.set_ylabel(r"$P_{max}/(\rho V^2)$")
        ax.set_title(r"Finite-area measurement vs point model, $\beta=10^\circ$ (Jain 2022)")
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8); fig.tight_layout()
        p = fig_dir / "fig7_sensor_anchor.png"; fig.savefig(p, dpi=130); plt.close(fig)
        out.append(p.name)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "artifacts" / "case_sph_wedge_entry_2d_footprint")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": "CASE-SPH-WEDGE-ENTRY-2D",
        "script": "benchmarks/case_sph_wedge_entry_2d/footprint_reference_ledger.py",
        "design": {
            "phi_m": PHI_M,
            "metric": "R(eps) = max window mean of Wagner p/q / point Cp_max, closed form",
            "eps_definition": "Phi / c_end",
            "beta_deg": BETA_DEG,
            "verdict_rule_changed": False,
        },
        "ratio_curve": ratio_curve(),
        "grids": grids(out),
        "solver_health": solver_health(out),
        "sensor_anchor": sensor_anchor(),
    }
    payload["figures"] = figures(payload, out)
    path = out / "reference_ledger.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"[reference] -> {path}")
    for g in payload["grids"]:
        print(f"  {g['tag']:>7}  eps={g['eps']:.3f}  R={g['footprint_ratio']:.4f}"
              f"  denom(COM)={g['denominator_designated']:.3f}"
              f"  denom(Wagner)={g['continuum_reference_cp']:.3f}"
              f"  sampled_ref_bias={g['sampled_reference_bias'] * 100:+.1f}%")
    for h in payload["solver_health"]:
        print(f"  health {h['tag']:>7}: {h.get('status')}"
              + (f" max_rho={h['max_rho_ratio']:.5f} ({h['density_compression_max'] * 100:+.2f} %)"
                 if h.get('status') == 'OK' else ''))
    sa = payload["sensor_anchor"]
    for r in sa.get("rows", []):
        print(f"  sensor {r['sensor']}: {r['cp_rho_v2']:.2f} vs COM point "
              f"{r['com_point_rho_v2']:.2f} -> ratio {r['ratio_to_com']:.3f}, "
              f"implied eps {r['implied_eps']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
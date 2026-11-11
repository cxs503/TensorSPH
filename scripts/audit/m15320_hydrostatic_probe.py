"""m15320 audit probe: independent re-derivation of the CASE-HYDROSTATIC metric.

Purpose
-------
The audit must not trust the mission's own reduction code (`band_table`) or its
report. This probe:

1. builds a still-water pool at a resolution the mission never used
   (dx = 0.025 m -> 40 x 16 = 640 particles; the case used dx = 0.02 and 0.0125),
2. evaluates the *same* SPH right-hand side (Tait EOS + Adami wall + summation
   density) exactly once (no time stepping),
3. bins the raw particle pressures itself (mass-weighted mean per 0.05 m band)
   and compares against the closed form p(y) = rho0 g (H - y),
4. reports, separately:
   * `max_interior_band_error_pct`  -- whole-tank interior bands (band >= 1,
     away from the free surface): this is the case's shipped metric;
   * `complete_support_*`           -- same bands but only columns whose kernel
     support is not truncated by a side wall: the truncation-free operator;
   * `wall_column_*`                -- side-wall-adjacent columns, restricted to
     non-free-surface rows (so the p ~ 0 top row does not swamp the statistic).
5. checks the declared mechanism: the whole-tank error should be reproduced by
   (2 wall columns / Nx) x |wall column error|.

If the case's claimed physics is real, the complete-support error must be
< 0.1 %, the wall columns an order of magnitude worse, and the whole-tank error
close to the wall-share prediction.

Run:  PYTHONPATH=<repo>/src python3 scripts/audit/m15320_hydrostatic_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

RHO0 = 1000.0
G = 9.81
DEPTH = 0.4
WIDTH = 1.0
BAND = 0.05


def main(out: Path) -> int:
    dx = 0.025
    from tensorsph.droplet_impact import PairListCache, rhs, wall_lattice
    from tensorsph.vv.hydrostatic import HydrostaticConfig, make_pool

    cfg = HydrostaticConfig(dx=dx, tank_width=WIDTH, depth=DEPTH)
    pos, mass, solver = make_pool(cfg)
    wall = wall_lattice(solver, pos.device, top=DEPTH)
    h = solver.h_factor * dx
    cache = PairListCache(2.0 * h, 0.25 * h, fluid_count=pos.shape[0])
    pairs, _ = cache.get(torch.cat((pos, wall), dim=0))
    _, pressure, _ = rhs(pos, torch.zeros_like(pos), mass, solver, pairs=pairs,
                         wall=wall, dt=cfg.dt())

    y = pos[:, 1]
    x = pos[:, 0]
    ref = RHO0 * G * (DEPTH - y)
    support = 2.0 * h
    # kernel support truncated by the top boundary -> free-surface rows
    free = y > DEPTH - support
    # kernel support truncated by a side wall -> wall columns
    near_wall = (x < support) | (x > WIDTH - support)

    def band_rows(mask):
        rows = []
        for lo in [round(i * BAND, 10) for i in range(int(DEPTH / BAND))]:
            hi = round(lo + BAND, 10)
            sel = mask & (y >= lo) & (y < hi)
            if not bool(sel.any()):
                continue
            w = mass[sel]
            p = float((pressure[sel] * w).sum() / w.sum())
            # The analytic reference is evaluated at the mass-weighted mean
            # height of the bin: the bin mean of a *linear* profile is exact.
            ybar = float((y[sel] * w).sum() / w.sum())
            pref = RHO0 * G * (DEPTH - ybar)
            rows.append({
                "band": len(rows), "y": ybar, "particles": int(sel.sum()),
                "pressure_pa": p, "pressure_analytic_pa": pref,
                "rel_error_pct": (p - pref) / pref * 100.0,
            })
        return rows

    rows = band_rows(~free)                    # whole tank, interior rows
    rows_support = band_rows(~free & ~near_wall)  # columns with full support
    # interior = everything except the floor row (band 0), as the case defines it
    interior = [r for r in rows if r["band"] >= 1]
    support_interior = [r for r in rows_support if r["band"] >= 1]
    max_interior = max(abs(r["rel_error_pct"]) for r in interior) if interior else float("nan")
    max_support = max(abs(r["rel_error_pct"]) for r in support_interior) \
        if support_interior else float("nan")

    wm = near_wall & ~free
    wall_err = ((pressure[wm] - ref[wm]) / ref[wm] * 100.0)
    wall_mean = float(wall_err.mean())
    n_columns = int(round(WIDTH / dx))
    prediction = (2.0 / n_columns) * abs(wall_mean)

    payload = {
        "probe": "m15320 independent hydrostatic probe",
        "dx": dx, "particles": int(pos.shape[0]), "columns": n_columns,
        "bands_interior": rows,
        "bands_complete_support": rows_support,
        "max_interior_band_error_pct": max_interior,
        "max_complete_support_band_error_pct": max_support,
        "wall_column_mean_error_pct": wall_mean,
        "wall_column_particles": int(wm.sum()),
        "wall_column_particle_fraction": float(wm.sum()) / float(pos.shape[0]),
        "wall_over_interior_ratio": abs(wall_mean) / max_interior if max_interior else float("inf"),
        "predicted_tank_error_pct": prediction,
        "manifestation_ratio_measured_over_predicted": max_interior / prediction
        if prediction else float("inf"),
        "case_gate_limit_pct": 1.0,
        "case_metric_at_this_resolution_pct": max_interior,
        "note": "resolution dx=0.025 (Nx=40) is outside the case's own grid "
                "(dx=0.02/0.0125); the reduction is re-implemented here, band_table unused",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("bands_interior", "bands_complete_support")},
                     indent=1, ensure_ascii=False))
    for tag, rs in (("tank", rows), ("support", rows_support)):
        for r in rs:
            print(f"{tag:8s} band{r['band']} y={r['y']:.4f} p={r['pressure_pa']:8.2f} "
                  f"ref={r['pressure_analytic_pa']:8.2f} err={r['rel_error_pct']:+7.3f}%")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parents[2] / "artifacts" / "audit_m15320" / "hydrostatic_probe.json"
    raise SystemExit(main(target))
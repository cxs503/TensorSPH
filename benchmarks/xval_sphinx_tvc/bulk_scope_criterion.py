"""Why the ``bulk`` TVC scope is unstable in a resting column: first- vs
zeroth-moment scope criteria.

Anchor: ``XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION``.

``tests/test_tvc_hydrostatic.py`` records that ``tvc_scope="bulk"`` blows up a
hydrostatic column (max speed 2.9 m/s against 0.38 m/s for ``all``) and explains
it as "correcting only the interior while leaving the surface uncorrected
creates a mismatch".  That explanation names a *symptom* (a relative motion
between corrected and uncorrected particles) and not a mechanism: it does not
say which particles the criterion selects, how large their correction is, or
whether the selection is the cause.

This benchmark settles it by measuring the two quantities the explanation
leaves open:

Leg A  ``analytic``   On a uniform lattice carrying a *volume* field
                      ``V(x) = V0 exp(kappa y)`` (i.e. a density field
                      ``rho ~ exp(-kappa y)``, the hydrostatic shape), the
                      ported grad-1 integral is not zero: it is
                      ``g_i = -2 grad ln V_i = +2 grad ln rho_i``, independent
                      of ``h`` and ``dx``.  Measured against that closed form
                      over a ``kappa`` ladder.

Leg B  ``scope``      On the *actual* initial state of the resting column
                      (``make_column`` + ``wall_lattice`` + hydrostatic mass),
                      compare the particle set admitted by the bulk criterion
                      ``C_i >= c_ref * conc`` with the set that genuinely has
                      full kernel support (``2h`` clear of both free surfaces).
                      If the criterion is sound the two sets coincide; the
                      measured ``|g| h`` of each says whether they do.

Leg C  ``verdict``    Resting column, 200 steps, scope/threshold ladder, plus a
                      first-moment criterion ``|g_i| h <= tol`` (the quantity
                      the correction actually uses).  Decides whether the
                      instability tracks the *selection* rather than the scope
                      idea.

Run:  python benchmarks/xval_sphinx_tvc/bulk_scope_criterion.py [--quick]
Writes ``docs/xval_sphinx_tvc/data/bulk_scope_criterion.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorsph.dam_break import (  # noqa: E402
    DamBreakConfig, make_column, run_case,
)
from tensorsph.droplet_impact import hydrostatic_density, wall_lattice  # noqa: E402
from tensorsph.neighbor_search import cell_linked_pairs  # noqa: E402
from tensorsph.particle_shifting import (  # noqa: E402
    concentration, estimate_reference_concentration,
)
from tensorsph.transport_velocity_correction import (  # noqa: E402
    kernel_gradient_integral,
)

DATA = ROOT / "docs" / "xval_sphinx_tvc" / "data"


def lattice(nx: int, ny: int, dx: float) -> torch.Tensor:
    xs = torch.arange(nx, dtype=torch.float64) * dx
    ys = torch.arange(ny, dtype=torch.float64) * dx
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def leg_analytic(kappas=(0.0, 1.0e-3, 1.0e-2, 0.0245, 0.05, 0.1)) -> dict:
    """Leg A: the grad-1 integral of a density (volume) gradient."""
    dx = 0.02
    h = 1.3 * dx
    positions = lattice(80, 40, dx)
    pairs = cell_linked_pairs(positions, 2.0 * h)
    v0 = dx * dx
    rows = []
    for h_ratio in (1.1, 1.3, 1.5):
        hh = h_ratio * dx
        pr = pairs if h_ratio == 1.3 else cell_linked_pairs(positions, 2.0 * hh)
        margin = 2.0 * hh
        interior = ((positions[:, 0] > margin)
                    & (positions[:, 0] < positions[:, 0].max() - margin)
                    & (positions[:, 1] > margin)
                    & (positions[:, 1] < positions[:, 1].max() - margin))
        for kappa in kappas:
            volume = v0 * torch.exp(kappa * positions[:, 1])
            g = kernel_gradient_integral(positions, volume, hh, pairs=pr)
            # closed form: g = -2 grad ln V, here -2 kappa e_y
            predicted = torch.zeros_like(g)
            predicted[:, 1] = -2.0 * kappa
            gy = float(g[interior, 1].mean())
            gx = float(g[interior, 0].mean())
            rel = abs(gy - float(predicted[0, 1])) / abs(float(predicted[0, 1])) \
                if kappa > 0 else float("nan")
            rows.append({"h_over_dx": h_ratio, "kappa_per_m": kappa,
                         "g_y_measured": gy, "g_x_measured": gx,
                         "g_y_predicted": float(predicted[0, 1]),
                         "rel_error": rel,
                         "ratio_g_over_kappa": (gy / kappa if kappa else None)})
    return {"leg": "analytic", "dx": dx, "rows": rows}


def _column_state(dx: float, column_width: float = 0.2,
                  column_height: float = 0.4):
    cfg = DamBreakConfig(dx=dx, dt=1.0e-4, steps=2, output_every=1,
                         device="cpu", column_width=column_width,
                         column_height=column_height, pool_width=2.0,
                         tank_height=0.75, hydrostatic_init=True)
    device = torch.device("cpu")
    positions, mass = make_column(cfg, device)
    wall = wall_lattice(cfg, device, top=cfg.tank_top())
    rho = hydrostatic_density(positions[:, 1], cfg, surface=cfg.column_height)
    return cfg, positions, mass, wall, rho


def leg_scope(dx: float = 0.04, column_width: float = 0.2,
              bulk_c: float = 0.95) -> dict:
    """Leg B: which particles does the concentration criterion select?"""
    cfg, positions, mass, wall, rho = _column_state(dx, column_width)
    h = cfg.h_factor * cfg.dx
    volume = mass / rho                       # solver convention m / rho
    all_positions = torch.cat((positions, wall), dim=0)
    pairs = cell_linked_pairs(all_positions, 2.0 * h)
    wall_volume = torch.full((wall.shape[0],), cfg.dx ** positions.shape[1],
                             dtype=mass.dtype)
    g = kernel_gradient_integral(all_positions, torch.cat((volume, wall_volume)),
                                 h, pairs=pairs, wall_count=wall.shape[0])
    gh = torch.linalg.vector_norm(g, dim=1) * h
    c = concentration(positions, volume, h)
    c_ref = estimate_reference_concentration(c)
    selected = c >= bulk_c * c_ref
    # genuinely full support: 2h clear of the two free surfaces (the right side
    # of the column and the top); floor and left wall are particles, they
    # support the kernel instead of truncating it.
    full = ((positions[:, 1] < cfg.column_height - 2.0 * h)
            & (positions[:, 0] < cfg.column_width - 2.0 * h))

    def stats(mask):
        if int(mask.sum()) == 0:
            return {"n": 0}
        return {"n": int(mask.sum()),
                "g_h_median": float(gh[mask].median()),
                "g_h_p95": float(gh[mask].quantile(0.95)),
                "g_h_max": float(gh[mask].max()),
                "concentration_min": float(c[mask].min()),
                "volume_rel_spread": float((volume[mask].max()
                                            - volume[mask].min())
                                           / volume[mask].mean())}

    # the spurious shift the criterion lets through, in units of h, at the
    # default coefficient alpha = 0.05
    alpha = 0.05
    return {"leg": "scope", "dx": dx, "column_width": column_width,
            "h": h, "bulk_c": bulk_c, "c_ref": c_ref,
            "n_fluid": int(positions.shape[0]),
            "density_gradient_term_g_h": 2.0 * h * cfg.gravity
            / cfg.sound_speed ** 2,
            "selected": stats(selected),
            "full_support": stats(full),
            "selected_and_full_support": int((selected & full).sum()),
            "alpha_default": alpha,
            "predicted_shift_over_h_selected": alpha * stats(selected)["g_h_max"],
            "predicted_shift_over_h_full_support":
                alpha * stats(full).get("g_h_max", float("nan"))}


def leg_verdict(dx: float = 0.04, steps: int = 200,
                column_width: float = 0.2) -> dict:
    """Leg C: 200-step resting column, scope/threshold ladder."""
    base = dict(dx=dx, dt=1.0e-4, steps=steps, output_every=max(1, steps // 4),
                device="cpu", column_width=column_width, column_height=0.4,
                pool_width=2.0, tank_height=0.75, hydrostatic_init=True)
    rows = []

    def run(label, **kw):
        t0 = time.time()
        cfg = DamBreakConfig(**base, **kw)
        r = run_case(cfg)
        rows.append({"label": label, "config": {k: v for k, v in kw.items()},
                     "max_speed": r.max_speed,
                     "max_density_compression": r.max_density_compression,
                     "max_tvc_shift": r.max_tvc_shift,
                     "max_tvc_shift_over_h": r.max_tvc_shift
                     / (cfg.h_factor * cfg.dx),
                     "wall_clock_s": round(time.time() - t0, 2)})
        return rows[-1]

    run("no_tvc", tvc_every=0)
    run("all", tvc_every=1, tvc_alpha=0.05, tvc_scope="all")
    for c in (0.95, 0.99, 0.995, 0.999):
        run(f"bulk_c{c:g}", tvc_every=1, tvc_alpha=0.05, tvc_scope="bulk",
            tvc_bulk_c=c)
    for tol in (0.05, 0.02, 0.005):
        run(f"bulk_g{tol:g}", tvc_every=1, tvc_alpha=0.05,
            tvc_scope="bulk_g", tvc_bulk_g=tol)
    return {"leg": "verdict", "dx": dx, "steps": steps,
            "column_width": column_width, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="operator legs only, skip the 200-step solver ladder")
    args = ap.parse_args()
    payload = {"anchor": "XVAL-SPH-TRANSPORT-VELOCITY-CORRECTION",
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "torch": torch.__version__}
    payload["analytic"] = leg_analytic()
    payload["scope_dx004"] = leg_scope(0.04)
    payload["scope_dx002"] = leg_scope(0.02)
    if not args.quick:
        payload["verdict"] = leg_verdict()
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / ("bulk_scope_criterion_quick.json" if args.quick
                  else "bulk_scope_criterion.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps(payload["scope_dx004"], indent=2))
    if "verdict" in payload:
        for row in payload["verdict"]["rows"]:
            print(f"{row['label']:>12s}  v_max={row['max_speed']:.3f} m/s  "
                  f"comp={row['max_density_compression']:.4f}  "
                  f"shift/h={row['max_tvc_shift_over_h']:.4f}  "
                  f"[{row['wall_clock_s']}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
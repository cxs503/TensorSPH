"""Mach-number window of the Taylor-Green rung (regime study).

The rung gates the energy decay rate of a periodic vortex. That measurement is
only meaningful while the discrete pressure field can represent the
``O(rho U0^2)`` field that balances the (exact) convective term, i.e. while the
acoustic density fluctuation needed to carry it stays well above the summation
density noise of the lattice. This script measures where that holds by sweeping
the peak velocity at fixed viscosity and resolution.

    python benchmarks/verification_ladder/tgv_regime.py

Writes ``artifacts/verification_ladder/tgv_mach_window.json`` and
``fig5_tgv_mach_window.png``. The result is a *validity statement about the
anchor*, not a pass/fail gate: outside the window the rung reports the analytic
error honestly and the number is not a statement about the solver.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorsph.vv.tgv import TGVConfig, run_tgv  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "artifacts" / "verification_ladder"))
    parser.add_argument("--dx", type=float, default=0.05)
    parser.add_argument("--sound-speed", type=float, default=10.0)
    parser.add_argument("--t-end", type=float, default=0.6)
    parser.add_argument("--velocity", default="0.02,0.05,0.1,0.2,1.0")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for u0 in [float(v) for v in args.velocity.split(",")]:
        cfg = TGVConfig(dx=args.dx, t_end=args.t_end, U0=u0,
                        sound_speed=args.sound_speed)
        run = run_tgv(cfg)
        rows.append({
            "U0": u0,
            "mach": u0 / cfg.sound_speed,
            "rate": run["measured"]["energy_decay_rate"],
            "rate_analytic": run["reference"]["energy_decay_rate"],
            "error_pct": run["error_pct"],
            "r2": run["measured"]["energy_decay_rate_r2"],
        })
        print(f"U0={u0:g} M={u0 / cfg.sound_speed:.3f} "
              f"rate={rows[-1]['rate']:.4f} err={rows[-1]['error_pct']:.2f} %")

    payload = {"dx": args.dx, "sound_speed": args.sound_speed,
               "t_end": args.t_end, "viscosity": TGVConfig().viscosity,
               "rows": rows,
               "gate_pct": 3.0}
    (out / "tgv_mach_window.json").write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog([r["mach"] for r in rows], [abs(r["error_pct"]) for r in rows],
              "o-")
    ax.axhline(3.0, color="tab:red", ls="--", lw=1.0, label="3 % gate")
    ax.set_xlabel("Mach number $U_0/c_0$")
    ax.set_ylabel("energy decay-rate error [%]")
    ax.set_title("TGV rung: validity window in Mach number")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig5_tgv_mach_window.png", dpi=140)
    print(f"wrote {out / 'tgv_mach_window.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
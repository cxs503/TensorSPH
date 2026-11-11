"""Three-way cross-validation of the 2-D dam break (XVAL-SPH-DAMBREAK).

One case spec, three tools.  The spec is built once by :func:`build_spec` and
echoed by every tool into its own result file, so a run whose configuration
does not match the spec is detectable by diffing the files rather than by
trusting the operator.

Sub-commands
------------
``spec``    write ``case_spec.json`` (single source of truth for all tools).
``ours``    run TensorSPH (``tensorsph.dam_break.run_case``).
``pysph``   run PySPH (needs the PySPH venv; see ``repro_cmd`` in the spec).
``compare`` build ``xval_table.json`` + ``bug_suspects.json`` from the tool
            result files, evaluate the three metric gates.
``figures`` render the comparison curves from the tool result files.

Sampling definitions (identical for every tool, recorded in the spec):

* front  ``Z/a = (max_i x_i - a) / a`` over fluid particles,
* height ``h/H = (max_i y_i) / H`` over fluid particles,
* both interpolated on the shared ``tau`` grid, ``tau = t sqrt(2 g / a)``.

Usage::

    python scripts/xval_dambreak/xval.py spec
    python scripts/xval_dambreak/xval.py ours   --dx 0.01 --nu 1e-6
    /tmp/xv2/bin/python scripts/xval_dambreak/xval.py pysph --dx 0.01 --nu 1e-6
    python scripts/xval_dambreak/xval.py compare
    python scripts/xval_dambreak/xval.py figures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "xval_dambreak"
REF_MAIN = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952.json"
REF_CROSS = ROOT / "benchmarks" / "dam_break" / "martin_moyce_1952_crosscheck.json"

# ---------------------------------------------------------------- case spec --
# Every number below is a *hard* spec item: all tools must use it verbatim.
A = 0.2          # column width [m]
H = 0.4          # column height [m]  (H = 2a, Martin & Moyce aspect ratio 2)
TANK_W = 2.0     # tank width [m]     (front reaches x/a < 4.4, no wall hit)
TANK_H = 0.75    # tank height [m]
RHO0 = 1000.0
C0 = 20.0        # artificial speed of sound [m/s]  (Mach ~0.2 at sqrt(gH))
GAMMA = 7.0      # Tait exponent
G = 9.81
HDX = 1.3        # h / dx
WALL_LAYERS = 3
NU_WATER = 1.0e-6  # kinematic viscosity of water at 20 C [m^2/s]
T_END = 0.30       # simulated time [s] -> tau_max = 2.97
TAUS = [0.2 * k for k in range(0, 15)]   # 0.0 .. 2.8, shared sampling grid


def _grid(extent: float, dx: float) -> list[float]:
    """Cell centres of the lattice that fits inside ``[0, extent]`` (floor)."""
    count = int(math.floor(extent / dx + 1.0e-9))
    if count < 1:
        raise ValueError(f"dx={dx:g} too coarse for extent {extent:g}")
    return [(k + 0.5) * dx for k in range(count)]


def fluid_lattice(dx: float) -> tuple[list[float], list[float]]:
    """Fluid column lattice -- identical to ``tensorsph.dam_break.make_column``."""
    xs, ys = _grid(A, dx), _grid(H, dx)
    x = [xi for yi in ys for xi in xs]
    y = [yi for yi in ys for xi in xs]
    return x, y


def wall_lattice(dx: float, layers: int = WALL_LAYERS) -> tuple[list[float], list[float]]:
    """Static wall lattice -- identical to ``tensorsph.droplet_impact.wall_lattice``.

    Three layers, half a spacing outside the fluid domain, extended around the
    corners; bottom and both side walls, no lid.
    """
    span_x = []
    x0 = -(layers - 0.5) * dx
    n = int(math.floor((TANK_W + 2.0 * (layers - 0.5) * dx - x0) / dx + 1.0e-9))
    if abs(x0 + n * dx - (TANK_W + (layers - 0.5) * dx)) > 1.0e-9:
        n += 1
    span_x = [x0 + k * dx for k in range(n)]
    columns = [(k + 0.5) * dx for k in range(int(math.floor(TANK_H / dx + 1.0e-9)))]
    x, y = [], []
    for layer in range(1, layers + 1):
        yy = 0.5 * dx - layer * dx
        for xi in span_x:
            x.append(xi)
            y.append(yy)
    for layer in range(1, layers + 1):
        for xx in (0.5 * dx - layer * dx, TANK_W - 0.5 * dx + layer * dx):
            for yi in columns:
                x.append(xx)
                y.append(yi)
    return x, y


def tau_of_time(t: float, a: float = A, g: float = G) -> float:
    return t * math.sqrt(2.0 * g / a)


def time_of_tau(tau: float, a: float = A, g: float = G) -> float:
    return tau / math.sqrt(2.0 * g / a)


def build_spec(dx: float, nu: float, t_end: float = T_END) -> dict:
    """The one case spec every tool must reproduce."""
    dt = 0.5 * 0.3 * HDX * dx / C0   # half the acoustic Courant limit
    dt = float(f"{dt:.3e}")
    nsteps = int(round(t_end / dt))
    return {
        "case": "XVAL-SPH-DAMBREAK",
        "geometry": {"a": A, "H": H, "tank_width": TANK_W, "tank_height": TANK_H,
                     "dim": 2, "initial_column_against_left_wall": True,
                     "downstream_bed": "dry"},
        "discretisation": {
            "dx": dx, "h_over_dx": HDX, "h": HDX * dx, "lattice": "square, cell centres",
            "wall_layers": WALL_LAYERS, "wall_offset": "half a spacing outside domain",
            "n_fluid": len(fluid_lattice(dx)[0]), "n_wall": len(wall_lattice(dx)[0]),
            "particle_mass": RHO0 * dx * dx, "mass_uniform": True,
        },
        "constitutive": {"rho0": RHO0, "c0": C0, "gamma": GAMMA, "gravity": G,
                         "eos": "Tait p = B[(rho/rho0)^gamma - 1], B = rho0 c0^2/gamma",
                         "viscosity_model": "laminar (Morris/Laplacian) kinematic nu",
                         "nu": nu, "reynolds_estimate": math.sqrt(G * H) * H / nu,
                         "surface_tension": 0.0, "artificial_viscosity": 0.0},
        "time": {"dt": dt, "t_end": t_end, "nsteps": nsteps,
                 "acoustic_courant": dt * C0 / (HDX * dx)},
        "bc": {"bottom": "static wall particles", "left": "static wall particles",
               "right": "static wall particles",
               "wall_pressure": "extrapolated from fluid (Adami et al. 2012 eq.11)",
               "tangential": "free slip (no wall friction)"},
        "sampling": {
            "tau_grid": TAUS, "tau_definition": "tau = t sqrt(2 g / a)",
            "front": "Z/a = (max_i x_i - a)/a over fluid particles",
            "height": "h/H = (max_i y_i)/H over fluid particles",
            "front_frame": "Z measured from the initial column face (Martin & Moyce)",
        },
        "reference": {
            "primary": str(REF_MAIN.relative_to(ROOT)),
            "crosscheck": str(REF_CROSS.relative_to(ROOT)),
            "source": "Martin & Moyce (1952) Phil. Trans. R. Soc. A 244:312-324",
            "digitised_uncertainty_Z_over_a": 0.05,
        },
        "gates": {"front_cross_tool_max_rel": 0.10,
                  "height_cross_tool_max_rel": 0.15,
                  "vs_martin_moyce_max_rel": 0.15},
        "deviations_allowed": [
            "integrator (TensorSPH symplectic Euler vs PySPH PEC predictor-corrector)",
            "PySPH adds XSPH velocity smoothing (scheme default) unless disabled",
            "PySPH AHA integrates a continuity equation for density; TensorSPH "
            "uses summation density",
            "PySPH AHA uses volume summation for the fluid density",
        ],
    }


# ------------------------------------------------------------------ helpers --
def _interp(ts: list[float], vs: list[float], q: float) -> float:
    if not ts or not vs:
        return float("nan")
    if q <= ts[0]:
        return vs[0]
    if q >= ts[-1]:
        return vs[-1]
    for i in range(1, min(len(ts), len(vs))):
        if ts[i] >= q:
            w = 0.0 if ts[i] == ts[i - 1] else (q - ts[i - 1]) / (ts[i] - ts[i - 1])
            return vs[i - 1] + w * (vs[i] - vs[i - 1])
    return vs[-1]


def _sample(times, front, height, taus):
    tq = [time_of_tau(tau) for tau in taus]
    return [{"tau": tau, "t": t, "Z_over_a": _interp(times, front, t),
             "h_over_H": _interp(times, height, t)} for tau, t in zip(taus, tq)]


def _load_ref() -> tuple[list[dict], list[dict]]:
    main = json.loads(REF_MAIN.read_text())["data"]
    cross = json.loads(REF_CROSS.read_text())["data"]
    cross = [{"tau": d["tau"], "Z_over_a": d["X_over_a"] - 1.0} for d in cross]
    return main, cross


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


# --------------------------------------------------------------------- ours --
def cmd_ours(args) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from tensorsph.dam_break import DamBreakConfig, run_case

    spec = build_spec(args.dx, args.nu)
    cfg = DamBreakConfig(
        dx=args.dx, h_factor=HDX, rho0=RHO0, sound_speed=C0, gamma=GAMMA,
        gravity=G, viscosity=args.nu, dt=spec["time"]["dt"],
        steps=spec["time"]["nsteps"], output_every=args.output_every,
        pool_width=TANK_W, tank_height=TANK_H, column_width=A, column_height=H,
        wall_layers=WALL_LAYERS, device="cpu", hydrostatic_init=False,
        density_calibration=False, adaptive_dt=False, boundary="wall",
        pair_skin_factor=0.25,
    )
    taus = TAUS
    snap = [time_of_tau(t) for t in taus]
    t0 = time.time()
    run = run_case(cfg, snapshot_times=snap)
    wall = time.time() - t0

    heights = []
    for t, pos, _vel in run.snapshots:
        heights.append((t, float(pos[:, 1].max()) / H))
    h_times = [h[0] for h in heights]
    h_vals = [h[1] for h in heights]
    front = [z / A for z in run.front]
    traj = [{"t": t, "tau": tau_of_time(t), "Z_over_a": f, "h_over_H": _interp(h_times, h_vals, t)}
            for t, f in zip(run.times, front)]

    payload = {
        "tool": "tensorsph", "version": _tensorsph_version(),
        "command": ("python scripts/xval_dambreak/xval.py ours "
                    f"--dx {args.dx} --nu {args.nu}"),
        "case_spec_echo": spec,
        "spec_match": {"dx": cfg.dx, "h_over_dx": cfg.h_factor, "c0": cfg.sound_speed,
                       "gamma": cfg.gamma, "nu": cfg.viscosity, "rho0": cfg.rho0,
                       "n_fluid": len(fluid_lattice(args.dx)[0]),
                       "n_wall": len(wall_lattice(args.dx)[0]),
                       "dt": cfg.dt, "nsteps": cfg.steps},
        "samples": _sample(h_times if False else run.times, front, h_vals, taus),
        "trajectory": traj,
        "diagnostics": {
            "wall_clock_s": wall, "steps_taken": run.steps_taken,
            "simulated_time": run.simulated_time,
            "mass_drift": run.mass_drift(), "escaped": run.escaped,
            "max_speed": run.max_speed,
            "max_density_compression": run.max_density_compression,
            "max_acoustic_courant": run.max_acoustic_courant,
            "wall_impact_time": run.wall_impact_time,
        },
        "notes": ("TensorSPH run of the shared spec; front sampling identical to "
                  "the spec definition (max x over fluid particles)."),
    }
    out = ART / f"result_ours_dx{args.dx:g}_nu{args.nu:g}.json"
    write_result(out, payload)
    return 0


def _tensorsph_version() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    import tensorsph  # noqa: F401
    from tensorsph.droplet_impact import __version__ as v
    return f"{v}+git{_git_rev()}"


def _git_rev() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(ROOT), text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


# -------------------------------------------------------------------- pysph --
def cmd_pysph(args) -> int:
    import numpy as np
    from pysph.base.kernels import CubicSpline
    from pysph.base.utils import get_particle_array
    from pysph.solver.application import Application
    from pysph.sph.scheme import AdamiHuAdamsScheme
    import pysph

    spec = build_spec(args.dx, args.nu)
    dx = args.dx
    h = HDX * dx
    m = RHO0 * dx * dx
    dt, t_end = spec["time"]["dt"], spec["time"]["t_end"]
    fx, fy = fluid_lattice(dx)
    wx, wy = wall_lattice(dx)
    alpha = float(args.alpha)

    class XvalDamBreak(Application):
        def create_scheme(self):
            return AdamiHuAdamsScheme(
                fluids=["fluid"], solids=["boundary"], dim=2, rho0=RHO0, c0=C0,
                nu=args.nu, h0=h, gx=0.0, gy=-G, p0=0.0, gamma=GAMMA,
                tdamp=0.0, alpha=alpha)

        def configure_scheme(self):
            kernel = CubicSpline(dim=2)
            import inspect
            from pysph.solver.solver import Solver
            valid = set(inspect.signature(Solver.__init__).parameters)
            kw = {"kernel": kernel, "dt": dt, "tf": t_end}
            for name in ("n_dump", "output_interval", "dump_interval"):
                if name in valid:
                    kw[name] = args.output_interval
                    break
            self.scheme.configure_solver(**kw)

        def create_particles(self):
            fluid = get_particle_array(name="fluid", x=np.array(fx),
                                       y=np.array(fy), h=h, m=m, rho=RHO0)
            boundary = get_particle_array(name="boundary", x=np.array(wx),
                                          y=np.array(wy), h=h, m=m, rho=RHO0)
            self.scheme.setup_properties([fluid, boundary])
            # setup_properties(clean=True) rewrites the property set; re-pin h so
            # the NNPS kernel radius (and cell size) cannot go to zero.
            fluid.h[:] = h
            boundary.h[:] = h
            fluid.rho[:] = RHO0
            boundary.rho[:] = RHO0
            return [fluid, boundary]

    # pysph parses sys.argv inside Application.__init__; keep our own CLI out
    # of its argument parser.
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        app = XvalDamBreak()
        app.run()
    finally:
        sys.argv = argv
    from pysph.solver.utils import iter_output
    times, front, height = [], [], []
    for sd, arr in iter_output(app.output_files, "fluid"):
        times.append(float(sd["t"]))
        front.append(float(arr.x.max()) / A - 1.0)   # Z/a from the column face
        height.append(float(arr.y.max()) / H)
    traj = [{"t": t, "tau": tau_of_time(t), "Z_over_a": f, "h_over_H": hh}
            for t, f, hh in zip(times, front, height)]
    payload = {
        "tool": "pysph", "version": getattr(pysph, "__version__", "unknown"),
        "command": ("/tmp/xv2/bin/python scripts/xval_dambreak/xval.py pysph "
                    f"--dx {args.dx} --nu {args.nu} --alpha {alpha}"),
        "case_spec_echo": spec,
        "spec_match": {"dx": dx, "h_over_dx": HDX, "c0": C0, "gamma": GAMMA,
                       "nu": args.nu, "rho0": RHO0,
                       "n_fluid": len(fx), "n_wall": len(wx),
                       "dt": dt, "nsteps": int(round(t_end / dt))},
        "samples": _sample(times, front, height, TAUS),
        "trajectory": traj,
        "scheme": {"name": "AdamiHuAdamsScheme", "kernel": "CubicSpline(dim=2)",
                   "artificial_viscosity_alpha": alpha,
                   "integrator": "PECIntegrator (scheme default)",
                   "xsph": "on (scheme default)",
                   "density": "volume summation + continuity"},
        "diagnostics": {"n_outputs": len(times),
                        "t_last": times[-1] if times else None},
    }
    out = ART / f"result_pysph_dx{args.dx:g}_nu{args.nu:g}_a{alpha:g}.json"
    write_result(out, payload)
    return 0


# ------------------------------------------------------------------ compare --
def cmd_compare(args) -> int:
    files = sorted(ART.glob("result_*.json"))
    if not files:
        print("no tool result files", file=sys.stderr)
        return 2
    ref_main, ref_cross = _load_ref()
    results = {json.loads(p.read_text())["tool"] + "|" + p.stem: json.loads(p.read_text())
               for p in files}
    # canonical arm: dx = 0.01, nu = 1e-6 (the spec value)
    canon = {k: v for k, v in results.items()
             if f"dx{args.dx:g}" in k and f"nu{1e-6:g}" in k}
    if not canon:
        print(f"canonical arm missing: {list(results)}", file=sys.stderr)
        return 2

    series = {}
    for key, res in canon.items():
        tool = res["tool"]
        if tool in series and "a0.1" in key:
            continue
        series.setdefault(tool, res)

    ref_tau = [d["tau"] for d in ref_main]
    ref_z = [d["Z_over_a"] for d in ref_main]
    refc_tau = [d["tau"] for d in ref_cross]
    refc_z = [d["Z_over_a"] for d in ref_cross]

    taus = TAUS
    cmp_taus = [t for t in taus if t > 0.0]
    front = {tool: [_interp([s["tau"] for s in r["samples"]], [s["Z_over_a"] for s in r["samples"]], t)
                    for t in cmp_taus] for tool, r in series.items()}
    height = {tool: [_interp([s["tau"] for s in r["samples"]], [s["h_over_H"] for s in r["samples"]], t)
                     for t in cmp_taus] for tool, r in series.items()}

    def rel(a, b):
        denom = 0.5 * (abs(a) + abs(b))
        return 0.0 if denom < 1e-12 else abs(a - b) / denom

    tools = sorted(front)
    front_cross = {t: [rel(front[a][i], front[b][i]) for i in range(len(cmp_taus))]
                   for a, b in [(tools[0], tools[1])] for t in [None]}
    if len(tools) >= 2:
        cross_front = [rel(front[tools[0]][i], front[tools[1]][i]) for i in range(len(cmp_taus))]
        cross_height = [rel(height[tools[0]][i], height[tools[1]][i]) for i in range(len(cmp_taus))]
    else:   # single reference available -> cross-tool band is WITHHELD, not zero
        cross_front = [float("nan")]
        cross_height = [float("nan")]
    del front_cross

    def ref_err(tool):
        sim = front[tool]
        out = []
        for tau, z in zip(cmp_taus, sim):
            r = _interp(ref_tau, ref_z, tau)
            if r >= 0.5:
                out.append((tau, r, z, abs(z - r) / r))
        return out

    table = []
    for tool in tools:
        re_main = ref_err(tool)
        out = []
        global_max = 0.0
        global_tau = None
        for tau, r, z, e in re_main:
            out.append({"tau": tau, "ref_Z_over_a": r, "got": z, "rel_err": e})
            if e > global_max:
                global_max, global_tau = e, tau
        rcross = _interp(refc_tau, refc_z, 2.0)
        table.append({
            "tool": tool, "metric": "surge_front_Z_over_a(tau)",
            "got": front[tool][-1], "got_definition": f"Z/a at tau={cmp_taus[-1]}",
            "vs_ref_err": global_max, "vs_ref_err_at_tau": global_tau,
            "vs_ref_err_definition": "max relative error vs Martin&Moyce (tau with ref Z/a>=0.5)",
            "cross_diff": max(cross_front) if len(tools) >= 2 else None,
            "cross_diff_definition": "max |A-B|/((A+B)/2) over the shared tau grid",
        })
        table.append({
            "tool": tool, "metric": "collapse_height_h_over_H(tau)",
            "got": height[tool][-1], "got_definition": f"h/H at tau={cmp_taus[-1]}",
            "vs_ref_err": None,
            "vs_ref_err_definition": "no experimental h(t) table is bundled (Martin & Moyce "
                                      "report the front trajectory); cross-tool only",
            "cross_diff": max(cross_height) if len(tools) >= 2 else None,
            "cross_diff_definition": "max |A-B|/((A+B)/2) over the shared tau grid",
        })

    gates = build_spec(args.dx, 1e-6)["gates"]
    verdict = {
        "front_cross_tool_max_rel": {"value": max(cross_front) if len(tools) >= 2 else None,
                                     "tol": gates["front_cross_tool_max_rel"],
                                     "pass": (max(cross_front) < gates["front_cross_tool_max_rel"]) if len(tools) >= 2 else None,
                                     "status": "measured" if len(tools) >= 2 else "WITHHELD (single reference)"},
        "height_cross_tool_max_rel": {"value": max(cross_height) if len(tools) >= 2 else None,
                                      "tol": gates["height_cross_tool_max_rel"],
                                      "pass": (max(cross_height) < gates["height_cross_tool_max_rel"]) if len(tools) >= 2 else None,
                                      "status": "measured" if len(tools) >= 2 else "WITHHELD (single reference)"},
        "vs_martin_moyce_max_rel": {
            t: {"value": max((row["rel_err"] for row in ref_err(t)), default=None)}
            for t in tools},
    }
    for t in tools:
        v = verdict["vs_martin_moyce_max_rel"][t]
        v["tol"] = gates["vs_martin_moyce_max_rel"]
        v["pass"] = (v["value"] is not None
                     and v["value"] < gates["vs_martin_moyce_max_rel"])

    # -- bug suspects: only what the numbers support, each with a locate ------
    suspects = []
    if len(tools) >= 2 and max(cross_front) >= gates["front_cross_tool_max_rel"]:
        worst = cmp_taus[int(np_argmax(cross_front))] if (np_argmax := _argmax)(cross_front) else None
        suspects.append({
            "metric": "surge_front_Z_over_a(tau)", "ours": front[tools[0]][_argmax(cross_front)],
            "oss": front[tools[1]][_argmax(cross_front)],
            "delta": cross_front[_argmax(cross_front)], "at_tau": worst,
            "suspected_side": "spec", "evidence":
                "cross-tool front deviation exceeds the 10% band; the two tools differ in "
                "density treatment (summation vs volume-summation + continuity) and pySPH "
                "adds XSPH smoothing, both of which damp the leading edge",
            "locate": "src/tensorsph/dam_break.py run_case (summation density, no XSPH) vs "
                      "pysph/sph/scheme.py AdamiHuAdamsScheme.get_equations",
        })
    # viscosity arms: compare ours at spec nu vs ours at the package default 0.01
    arms = {}
    for key, res in results.items():
        if res["tool"] == "tensorsph" and f"dx{args.dx:g}" in key:
            arms[res["case_spec_echo"]["constitutive"]["nu"]] = res
    if len(arms) >= 2 and 0.01 in arms and 1e-6 in arms:
        a_spec, a_def = arms[1e-6], arms[0.01]
        zs = [s["Z_over_a"] for s in a_spec["samples"] if s["tau"] > 0]
        zd = [s["Z_over_a"] for s in a_def["samples"] if s["tau"] > 0]
        dev = max(rel(a, b) for a, b in zip(zs, zd))
        suspects.append({
            "metric": "surge_front_Z_over_a(tau)",
            "ours": zd[-1], "oss": zs[-1], "delta": dev,
            "at_tau": cmp_taus[-1],
            "suspected_side": "ours",
            "evidence": f"TensorSPH with the package default nu=0.01 m^2/s (Re~{math.sqrt(G*H)*H/0.01:.0f}) "
                        f"differs from the same code at the water value nu=1e-6 (Re~{math.sqrt(G*H)*H/1e-6:.0e}) "
                        f"by {dev*100:.1f}% in Z/a; the default is 1e4x the viscosity of water, so it is a "
                        "case-configuration defect, not a numerical one",
            "locate": "src/tensorsph/dam_break.py DamBreakConfig (inherits Config.viscosity=0.01 in "
                      "src/tensorsph/droplet_impact.py); benchmarks/dam_break/run.py uses that default",
        })
    write_result(ART / "xval_table.json",
                 {"dx": args.dx, "tools": tools, "taus": cmp_taus, "table": table,
                  "series": {"front": front, "height": height}, "verdict": verdict})
    write_result(ART / "bug_suspects.json", {"suspects": suspects})
    print(json.dumps(verdict, indent=2))
    return 0


def _argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


# ------------------------------------------------------------------ figures --
def cmd_figures(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    files = sorted(ART.glob("result_*.json"))
    results = [json.loads(p.read_text()) for p in files]
    canon = [r for r in results
             if abs(r["case_spec_echo"]["discretisation"]["dx"] - args.dx) < 1e-12
             and r["case_spec_echo"]["constitutive"]["nu"] == 1e-6]
    if len(canon) < 2:
        print("canonical arm insufficient for figures", file=sys.stderr)
        return 2
    ref_main, ref_cross = _load_ref()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for res in canon:
        tr = res["trajectory"]
        label = f"{res['tool']} dx={res['case_spec_echo']['discretisation']['dx']:g}"
        axes[0].plot([d["tau"] for d in tr], [d["Z_over_a"] for d in tr], label=label)
        axes[1].plot([d["tau"] for d in tr], [d["h_over_H"] for d in tr], label=label)
    axes[0].plot([d["tau"] for d in ref_main], [d["Z_over_a"] for d in ref_main], "ks",
                 ms=4, label="Martin & Moyce (1952)")
    axes[0].plot([d["tau"] for d in ref_cross], [d["Z_over_a"] for d in ref_cross], "^",
                 mfc="none", mec="0.4", ms=5, label="Martin & Moyce, 2nd digitisation")
    axes[0].set_xlabel(r"$\tau = t\sqrt{2g/a}$")
    axes[0].set_ylabel("$Z/a$")
    axes[0].set_title("surge front")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel("$h/H$")
    axes[1].set_title("collapse height")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = ART / f"xval_dambreak_dx{args.dx:g}.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------- main --
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("spec")
    sp.add_argument("--dx", type=float, default=0.01)
    sp.add_argument("--nu", type=float, default=NU_WATER)
    so = sub.add_parser("ours")
    so.add_argument("--dx", type=float, default=0.01)
    so.add_argument("--nu", type=float, default=NU_WATER)
    so.add_argument("--output-every", type=int, default=25)
    sy = sub.add_parser("pysph")
    sy.add_argument("--dx", type=float, default=0.01)
    sy.add_argument("--nu", type=float, default=NU_WATER)
    sy.add_argument("--alpha", type=float, default=0.0)
    sy.add_argument("--output-interval", type=int, default=25)
    sc = sub.add_parser("compare")
    sc.add_argument("--dx", type=float, default=0.01)
    sf = sub.add_parser("figures")
    sf.add_argument("--dx", type=float, default=0.01)
    args = p.parse_args(argv)

    if args.cmd == "spec":
        ART.mkdir(parents=True, exist_ok=True)
        write_result(ART / "case_spec.json", build_spec(args.dx, args.nu))
        return 0
    if args.cmd == "ours":
        return cmd_ours(args)
    if args.cmd == "pysph":
        return cmd_pysph(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "figures":
        return cmd_figures(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
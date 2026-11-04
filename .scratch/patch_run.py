"""Extend the laplace_pressure_floor harness with the P0 closure switch."""
from pathlib import Path

RUN = Path("benchmarks/laplace_pressure_floor/run.py")
text = RUN.read_text()


def sub(old, new, count=1):
    global text
    assert text.count(old) == count, (old[:60], text.count(old))
    text = text.replace(old, new)


# ---- docstring note
sub('''Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --quick
"""''',
    '''P0 (free-surface pressure closure) is an opt-in solver switch, not a
modification of this harness: ``--free-surface-pressure`` turns on the Adami
et al. (2012) sec. 3.3 shell extrapolation in
``tensorsph.droplet_impact.rhs`` (``Config.free_surface_pressure``) and writes
``report_fs.json`` instead of ``report.json``, so the closure-on and closure-off
readings stay side by side. See docs/verification_free_surface_pressure.md.

Usage
-----
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py --quick
    PYTHONPATH=src python benchmarks/laplace_pressure_floor/run.py \\
        --free-surface-pressure --sound-speeds 20 5
"""''')

# ---- relax signature
sub('''def relax(dx: float, c0: float, scheme: str = "fit_quadratic",
          calibrate_mass: bool = True, t_end_over_t_sigma: float = 1.5,
          max_steps: int = 4000, sample_every: int = 50):''',
    '''def relax(dx: float, c0: float, scheme: str = "fit_quadratic",
          calibrate_mass: bool = True, t_end_over_t_sigma: float = 1.5,
          max_steps: int = 4000, sample_every: int = 50,
          free_surface_pressure: bool = False,
          free_surface_threshold: float = 0.99):''')

# ---- Config construction
sub('''    cfg = Config(dx=dx, h_factor=H_FACTOR, rho0=RHO0, sound_speed=c0,
                 gamma=GAMMA, gravity=0.0, viscosity=VISCOSITY,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                 csf_curvature=scheme)''',
    '''    cfg = Config(dx=dx, h_factor=H_FACTOR, rho0=RHO0, sound_speed=c0,
                 gamma=GAMMA, gravity=0.0, viscosity=VISCOSITY,
                 surface_tension_coefficient=SIGMA, csf_scheme="consistent",
                 csf_curvature=scheme,
                 free_surface_pressure=free_surface_pressure,
                 free_surface_threshold=free_surface_threshold)''')

# ---- extra observables: how many particles the closure touched
sub('''        "spurious_current": {''',
    '''        "free_surface_pressure": {
            "enabled": bool(free_surface_pressure),
            "threshold_rho_over_rho0": free_surface_threshold,
            "n_free_surface_particles": int(
                (rho < free_surface_threshold * RHO0).sum()),
            "p_core_over_sigma_R": float(final["absolute"] / DP_ANALYTIC),
            "p_shell_mean_Pa": float(pressure[(r >= 0.95) & (r < 1.05)].mean()),
        },
        "spurious_current": {''')

# ---- CLI
sub('''    ap.add_argument("--t-end", type=float, default=1.5,
                    help="window length in capillary times")
    args = ap.parse_args()''',
    '''    ap.add_argument("--t-end", type=float, default=1.5,
                    help="window length in capillary times")
    ap.add_argument("--free-surface-pressure", action="store_true",
                    help="enable the Adami 2012 sec. 3.3 shell-pressure closure")
    ap.add_argument("--free-surface-threshold", type=float, default=0.99,
                    help="rho/rho0 below which a particle is free surface")
    args = ap.parse_args()
    closure = {"free_surface_pressure": args.free_surface_pressure,
               "free_surface_threshold": args.free_surface_threshold}''')

for old, new in [
    ('''            run = relax(dx, c0, t_end_over_t_sigma=args.t_end)''',
     '''            run = relax(dx, c0, t_end_over_t_sigma=args.t_end, **closure)'''),
    ('''    uncal = relax(args.dx, ref_c0, calibrate_mass=False,
                  t_end_over_t_sigma=args.t_end)''',
     '''    uncal = relax(args.dx, ref_c0, calibrate_mass=False,
                  t_end_over_t_sigma=args.t_end, **closure)'''),
    ('''    ctrl_scheme = relax(args.dx, c0s[-1], scheme="mls",
                        t_end_over_t_sigma=args.t_end)''',
     '''    ctrl_scheme = relax(args.dx, c0s[-1], scheme="mls",
                        t_end_over_t_sigma=args.t_end, **closure)'''),
]:
    sub(old, new)

# ---- report filename + closure record
sub('''        "structures": structures,''',
    '''        "closure": {"free_surface_pressure": args.free_surface_pressure,
                    "free_surface_threshold": args.free_surface_threshold,
                    "model": "Adami, Ihmsen & Adams (2012) JCP 231:7057 sec. 3.3"},
        "structures": structures,''')

sub('''    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    report["figures"] = [make_figures(report)]
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", HERE / "report.json", report["figures"])''',
    '''    name = "report_fs.json" if args.free_surface_pressure else "report.json"
    (HERE / name).write_text(json.dumps(report, indent=2))
    report["figures"] = [make_figures(
        report, suffix="_fs" if args.free_surface_pressure else "")]
    (HERE / name).write_text(json.dumps(report, indent=2))
    print("wrote", HERE / name, report["figures"])''')

# ---- figure file name suffix so the two reports do not overwrite each other
sub('''def make_figures(report, out_dir=HERE):''',
    '''def make_figures(report, out_dir=HERE, suffix: str = ""):''')
sub('''    path = Path(out_dir) / "laplace_pressure_floor.png"''',
    '''    path = Path(out_dir) / f"laplace_pressure_floor{suffix}.png"''')

RUN.write_text(text)
print("patched", RUN)
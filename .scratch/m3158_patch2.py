"""m3158: extend the Laplace pressure benchmark with the closure A/B."""
from pathlib import Path

SRC = Path("src/tensorsph/droplet_impact.py")
text = SRC.read_text()
anchor = '''    free_surface_pressure: bool = False
    free_surface_threshold: float = 0.99
'''
new = '''    free_surface_pressure: bool = False
    free_surface_threshold: float = 0.99
    # Passes of Shepard smoothing applied to the colour field *inside* the
    # curvature operator only (tensorsph.csf_consistent); 0 keeps the m2447
    # behaviour. See docs/verification_csf_curvature_bias.md.
    csf_curvature_color_smoothing: int = 0
'''
assert anchor in text
text = text.replace(anchor, new, 1)
SRC.write_text(text)

ST = Path("src/tensorsph/surface_tension.py")
st = ST.read_text()
anchor_st = '''            curvature_scheme=getattr(cfg, "csf_curvature", "mls"))'''
assert anchor_st in st
st = st.replace(anchor_st, '''            curvature_scheme=getattr(cfg, "csf_curvature", "mls"),
            curvature_color_smoothing=getattr(
                cfg, "csf_curvature_color_smoothing", 0))''', 1)
ST.write_text(st)

RUN = Path("benchmarks/laplace_pressure_floor/run.py")
rt = RUN.read_text()

anchor_sig = '''def relax(dx: float, c0: float, scheme: str = "fit_quadratic",
          calibrate_mass: bool = True, t_end_over_t_sigma: float = 1.5,
          max_steps: int = 4000, sample_every: int = 50):'''
assert anchor_sig in rt
rt = rt.replace(anchor_sig, '''def relax(dx: float, c0: float, scheme: str = "fit_quadratic",
          calibrate_mass: bool = True, t_end_over_t_sigma: float = 1.5,
          max_steps: int = 4000, sample_every: int = 50,
          fs_pressure: bool = False, color_smoothing: int = 0):''', 1)

anchor_cfg = '''                 csf_curvature=scheme)'''
assert anchor_cfg in rt
rt = rt.replace(anchor_cfg, '''                 csf_curvature=scheme,
                 free_surface_pressure=fs_pressure,
                 csf_curvature_color_smoothing=color_smoothing)''', 1)

anchor_tail = '''    n_tail = max(1, int(0.4 * len(trace["absolute"])))
    tail_abs = sum(trace["absolute"][-n_tail:]) / n_tail
'''
assert anchor_tail in rt
rt = rt.replace(anchor_tail, '''    n_tail = max(1, int(0.4 * len(trace["absolute"])))
    tail_abs = sum(trace["absolute"][-n_tail:]) / n_tail
    # The breathing mode still has an amplitude of ~10 % at t = 1.5 t_sigma; the
    # last quarter of the window is the closest this short run gets to
    # equilibrium, so it is reported next to the 40 % average.
    n_late = max(1, int(0.25 * len(trace["absolute"])))
    late_abs = sum(trace["absolute"][-n_late:]) / n_late
    late_amp = (max(trace["absolute"][-n_late:]) - min(trace["absolute"][-n_late:]))
''', 1)

anchor_keys = '''        "absolute_time_average_tail": score(tail_abs),'''
assert anchor_keys in rt
rt = rt.replace(anchor_keys, '''        "absolute_time_average_tail": score(tail_abs),
        "absolute_time_average_late": score(late_abs),
        "absolute_late_peak_to_peak_Pa": late_amp,
        "free_surface_pressure": bool(cfg.free_surface_pressure),
        "csf_curvature_color_smoothing": int(color_smoothing),''', 1)

anchor_main = '''    ap.add_argument("--t-end", type=float, default=1.5,
                    help="window length in capillary times")
    args = ap.parse_args()'''
assert anchor_main in rt
rt = rt.replace(anchor_main, '''    ap.add_argument("--t-end", type=float, default=1.5,
                    help="window length in capillary times")
    ap.add_argument("--fs-pressure", action="store_true",
                    help="enable the Adami/Ihmsen/Adams free-surface pressure "
                         "closure (cfg.free_surface_pressure)")
    ap.add_argument("--color-smoothing", type=int, default=0,
                    help="Shepard passes on the colour field inside the "
                         "curvature operator (cfg.csf_curvature_color_smoothing)")
    ap.add_argument("--out", default=None,
                    help="report file name (default report.json)")
    args = ap.parse_args()''', 1)

rt = rt.replace('''            run = relax(dx, c0, t_end_over_t_sigma=args.t_end)''',
                '''            run = relax(dx, c0, t_end_over_t_sigma=args.t_end,
                        fs_pressure=args.fs_pressure,
                        color_smoothing=args.color_smoothing)''', 1)
rt = rt.replace('''    uncal = relax(args.dx, ref_c0, calibrate_mass=False,
                  t_end_over_t_sigma=args.t_end)''',
                '''    uncal = relax(args.dx, ref_c0, calibrate_mass=False,
                  t_end_over_t_sigma=args.t_end,
                  fs_pressure=args.fs_pressure,
                  color_smoothing=args.color_smoothing)''', 1)
rt = rt.replace('''    ctrl_scheme = relax(args.dx, c0s[-1], scheme="mls",
                        t_end_over_t_sigma=args.t_end)''',
                '''    ctrl_scheme = relax(args.dx, c0s[-1], scheme="mls",
                        t_end_over_t_sigma=args.t_end,
                        fs_pressure=args.fs_pressure,
                        color_smoothing=args.color_smoothing)''', 1)

rt = rt.replace('''        "controls": {"uncalibrated_c0_reference": uncal,
                     "mls_scheme_lowest_c0": ctrl_scheme},
    }
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    report["figures"] = [make_figures(report)]
    (HERE / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", HERE / "report.json", report["figures"])''',
'''        "controls": {"uncalibrated_c0_reference": uncal,
                     "mls_scheme_lowest_c0": ctrl_scheme},
        "closure": {"free_surface_pressure": bool(args.fs_pressure),
                    "free_surface_threshold": FREE_SURFACE_THRESHOLD,
                    "csf_curvature_color_smoothing": int(args.color_smoothing),
                    "window_t_over_t_sigma": args.t_end},
    }
    out_name = args.out or ("report_fs_closure.json" if args.fs_pressure
                            else "report.json")
    out_path = HERE / out_name
    out_path.write_text(json.dumps(report, indent=2))
    report["figures"] = [make_figures(report)]
    out_path.write_text(json.dumps(report, indent=2))
    print("wrote", out_path, report["figures"])''', 1)

rt = rt.replace('''BANDS = ((0.0, 0.50),''', '''FREE_SURFACE_THRESHOLD = 0.99
BANDS = ((0.0, 0.50),''', 1)

rt = rt.replace('''    print(f"dx={dx:g} c0={c0:5g}: abs {run['definition_absolute']['rel_error_pct']:+7.1f} %"''',
                '''    print(f"dx={dx:g} c0={c0:5g}: abs {run['definition_absolute']['rel_error_pct']:+7.1f} %"''', 1)
rt = rt.replace('''                  f"  tail-avg {run['absolute_time_average_tail']['rel_error_pct']:+7.1f} %"''',
                '''                  f"  tail-avg {run['absolute_time_average_tail']['rel_error_pct']:+7.1f} %"''', 1)

RUN.write_text(rt)
print("patched run.py")
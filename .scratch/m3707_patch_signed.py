"""Record a signed mode-2 observable and fit it, next to the legacy |s| one.

Two exact-literal patches to benchmarks/oscillating_droplet/run.py:
  1. mode_amplitude(): return s_signed = s * cos(2 theta) against a reference
     axis, plus cos2theta and the major axis as JSON-safe floats.
  2. run_case(): keep the reference axis of the initial state, record s_signed
     in the trace, and fit it as well.
"""
from pathlib import Path

TARGET = Path("benchmarks/oscillating_droplet/run.py")
src = TARGET.read_text()

OLD_AMP = '''def mode_amplitude(pos: torch.Tensor, mass: torch.Tensor) -> dict:'''
NEW_AMP = '''def mode_amplitude(pos: torch.Tensor, mass: torch.Tensor,
                   reference_axis=None) -> dict:'''
assert src.count(OLD_AMP) == 1

OLD_DOC = '''    integral over the whole drop, so it is far less sensitive to the lattice
    jaggedness than an interface-radius estimator (see the note in
    ``docs/verification_oscillating_droplet.md``: the radius-hull Fourier
    coefficient of this very lattice returns 0.055-0.13 for ``eps = 0.10``).
    """'''
NEW_DOC = '''    integral over the whole drop, so it is far less sensitive to the lattice
    jaggedness than an interface-radius estimator (see the note in
    ``docs/verification_oscillating_droplet.md``: the radius-hull Fourier
    coefficient of this very lattice returns 0.055-0.13 for ``eps = 0.10``).

    ``s_moment`` is a *magnitude*: ``A >= B`` by construction, so the sign of
    ``eps`` is lost and a released drop reads as ``|cos(w t)|`` -- rectified at
    ``2 w``.  ``s_signed = s_moment * cos(2 theta)`` restores it, with ``theta``
    the angle between the current major axis and ``reference_axis`` (the initial
    major axis): ``cos 2 theta = 2 (e_major . e_ref)^2 - 1`` is smooth, equals
    +1 when the ellipse is still aligned with the launch axis and -1 when it has
    turned through 90 degrees.  Without a reference axis the major axis is its
    own reference and ``s_signed == s_moment``.  See
    ``docs/verification_oscillating_droplet_fit.md`` for the measurement this
    repairs.
    """'''
assert src.count(OLD_DOC) == 1

OLD_BODY = '''    lam = torch.linalg.eigvalsh(cov)          # ascending
    lam1 = float(lam[-1].clamp_min(1e-30))
    lam2 = float(lam[0].clamp_min(1e-30))
    big, small = 2.0 * math.sqrt(lam1), 2.0 * math.sqrt(lam2)
    return {"s_moment": (big - small) / (big + small),
            "semi_major": big, "semi_minor": small,
            "r_eff": 2.0 * math.sqrt(0.5 * (lam1 + lam2))}'''
NEW_BODY = '''    lam, vecs = torch.linalg.eigh(cov)        # ascending
    lam1 = float(lam[-1].clamp_min(1e-30))
    lam2 = float(lam[0].clamp_min(1e-30))
    big, small = 2.0 * math.sqrt(lam1), 2.0 * math.sqrt(lam2)
    s_moment = (big - small) / (big + small)
    major = vecs[:, -1]
    cos2 = 1.0
    if reference_axis is not None:
        ref = torch.as_tensor(reference_axis, dtype=pos.dtype, device=pos.device)
        norm = float(torch.linalg.vector_norm(ref))
        if norm > 0.0:
            cos2 = float(2.0 * (float(major @ ref) / norm) ** 2 - 1.0)
    return {"s_moment": s_moment,
            "s_signed": s_moment * cos2,
            "cos2theta": cos2,
            # JSON-safe, so the report/manifest can carry it and the next call
            # can use it as reference_axis.
            "major_axis": [float(major[0]), float(major[1])],
            "semi_major": big, "semi_minor": small,
            "r_eff": 2.0 * math.sqrt(0.5 * (lam1 + lam2))}'''
assert src.count(OLD_BODY) == 1

OLD_RUN = '''    a0 = mode_amplitude(pos, mass)

    ts, s_mom, ke_hist = [], [], []'''
NEW_RUN = '''    a0 = mode_amplitude(pos, mass)
    reference_axis = a0["major_axis"]

    ts, s_mom, s_sig, ke_hist = [], [], [], []'''
assert src.count(OLD_RUN) == 1

OLD_LOOP = '''            amp = mode_amplitude(pos, mass)
            ts.append((step + 1) * dt)
            s_mom.append(amp["s_moment"])'''
NEW_LOOP = '''            amp = mode_amplitude(pos, mass, reference_axis=reference_axis)
            ts.append((step + 1) * dt)
            s_mom.append(amp["s_moment"])
            s_sig.append(amp["s_signed"])'''
assert src.count(OLD_LOOP) == 1

OLD_FIT = '''    fit_mom = damped_cosine_fit(ts, s_mom, omega_ref)
    fit_ke = damped_cosine_fit(ts, ke_hist, 2.0 * omega_ref)
    zc_mom = zero_crossing_period(ts, s_mom)'''
NEW_FIT = '''    fit_mom = damped_cosine_fit(ts, s_mom, omega_ref)
    fit_sig = damped_cosine_fit(ts, s_sig, omega_ref)
    fit_ke = damped_cosine_fit(ts, ke_hist, 2.0 * omega_ref)
    zc_mom = zero_crossing_period(ts, s_mom)
    zc_sig = zero_crossing_period(ts, s_sig)'''
assert src.count(OLD_FIT) == 1

OLD_ERR = '''    err_pct = (fit_mom["omega"] - omega_ref) / omega_ref * 100.0'''
NEW_ERR = '''    err_pct = (fit_mom["omega"] - omega_ref) / omega_ref * 100.0
    err_sig = (fit_sig["omega"] - omega_ref) / omega_ref * 100.0'''
assert src.count(OLD_ERR) == 1

OLD_KEYS = '''            "fit": fit_mom,
            "fit_kinetic_energy": fit_ke,
            "zero_crossing": zc_mom,'''
NEW_KEYS = '''            "fit": fit_mom,
            # Signed-observable path: the same estimator on the rectification-
            # free trace.  ``fit_second_moment_*`` above is kept verbatim as
            # the historical reading, so the two can be compared directly.
            "fit_signed_rad_per_s": fit_sig["omega"],
            "fit_signed_rel_error_pct": err_sig,
            "fit_signed": fit_sig,
            "zero_crossing_signed_rad_per_s": zc_sig["omega"],
            "zero_crossing_signed_rel_error_pct":
                ((zc_sig["omega"] - omega_ref) / omega_ref * 100.0
                 if math.isfinite(zc_sig["omega"]) else float("nan")),
            "fit_kinetic_energy": fit_ke,
            "zero_crossing": zc_mom,'''
assert src.count(OLD_KEYS) == 1

OLD_TRACE = '''            "trace": {"t": ts, "s_second_moment": s_mom,
                      "kinetic_energy_J": ke_hist},'''
NEW_TRACE = '''            "trace": {"t": ts, "s_second_moment": s_mom,
                      "s_signed": s_sig, "kinetic_energy_J": ke_hist},'''
assert src.count(OLD_TRACE) == 1

for old, new in ((OLD_AMP, NEW_AMP), (OLD_DOC, NEW_DOC), (OLD_BODY, NEW_BODY),
                 (OLD_RUN, NEW_RUN), (OLD_LOOP, NEW_LOOP), (OLD_FIT, NEW_FIT),
                 (OLD_ERR, NEW_ERR), (OLD_KEYS, NEW_KEYS), (OLD_TRACE, NEW_TRACE)):
    src = src.replace(old, new, 1)
TARGET.write_text(src)
print("patched signed observable:", TARGET)

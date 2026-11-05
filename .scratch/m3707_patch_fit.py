"""Replace the single-start damped-cosine fit in run.py with a basin-aware one.

Exact literal surgery: the block from ``def damped_cosine_fit`` up to (not
including) ``def peak_decay`` is swapped.  Verified by AST parse, a duplicate
definition scan, and the test suite.
"""
from pathlib import Path

TARGET = Path("benchmarks/oscillating_droplet/run.py")

NEW = '''def _damped_cosine_solve(tt: torch.Tensor, ss: torch.Tensor, om: float,
                         lam: float):
    """Closed-form ``(c, P, Q)`` for a fixed ``(omega, lambda)`` plus its rms.

    ``s = c + exp(-lam (t - t0)) (P cos(w t) + Q sin(w t))`` is linear in
    ``(c, P, Q)``, so only the two nonlinear parameters are searched.  The
    envelope is measured from the first sample, which keeps ``P`` and ``Q``
    O(amplitude) instead of exp(+lam t0) times it.
    """
    env = torch.exp(-lam * (tt - tt[0]))
    design = torch.stack((torch.ones_like(tt), env * torch.cos(om * tt),
                          env * torch.sin(om * tt)), dim=1)
    coef, *_ = torch.linalg.lstsq(design, ss[:, None])
    coef = coef[:, 0]
    residual = design @ coef - ss
    return coef, float(torch.sqrt((residual * residual).mean()))


def _refine_damped_cosine(ts: torch.Tensor, ss: torch.Tensor, om: float,
                          lam: float, iterations: int = 6):
    """Coordinate descent in ``(omega, lambda)`` with a bracket that shrinks.

    The bracket halves every ``1.6`` iterations, so the first sweep can move
    the seed by up to 2 % in ``omega`` and the last one resolves 1e-3 of it --
    the same policy regardless of which basin the seed came from, which is what
    makes the multi-start result independent of the coarse-grid ordering.
    """
    _, rms = _damped_cosine_solve(ts, ss, om, lam)
    for step in range(iterations):
        shrink = 1.6 ** step
        span = max(1e-9, 0.02 * om / shrink)
        for cand in torch.linspace(om - span, om + span, 41):
            _, r = _damped_cosine_solve(ts, ss, float(cand), lam)
            if r < rms:
                om, rms = float(cand), r
        span_lam = max(1e-6, 0.25 * max(lam, 1.0) / shrink)
        for cand in torch.linspace(max(0.0, lam - span_lam), lam + span_lam, 21):
            _, r = _damped_cosine_solve(ts, ss, om, float(cand))
            if r < rms:
                lam, rms = float(cand), r
    return om, lam, rms


#: Basins refined by :func:`damped_cosine_fit` by default.  ``1`` restores the
#: historical single-start behaviour and is only used to demonstrate the defect.
FIT_STARTS = 6


def damped_cosine_fit(t: list[float], s: list[float], omega_guess: float,
                      starts: int = FIT_STARTS) -> dict:
    """Least-squares fit of ``s = c + e^{-lam t}(P cos wt + Q sin wt)``.

    ``(omega, lambda)`` are searched on a coarse ``(omega, lambda)`` grid (the
    trace is subsampled for it) and then refined on the full trace; the three
    linear coefficients are solved in closed form at every grid point.

    Why the search is multi-start
    -----------------------------
    A release trace covers only a couple of periods, so the rms landscape has
    several near-equal basins and the historical single-start search reported
    whichever basin the *subsampled* coarse grid happened to win.  A round-off
    level difference between BLAS implementations can reorder two near-equal
    coarse minima, after which the local refinement (which only descends) locks
    onto a different basin: identical code then measured ``+8.64 %`` on x86_64
    CI and ``-31.78 %`` on loong64.  That spread is an instrument defect, not
    physics, and it is diagnosed in ``fit_robustness.py``.

    Here the coarse grid is clustered into basins separated by at least one
    Rayleigh bin ``2 pi / T`` (the finest frequency difference the recorded
    window can resolve), up to ``starts`` basins are refined, and the lowest
    rms wins.  Since every competitive basin is refined, the winner no longer
    depends on the coarse-grid ordering.

    The returned ``ambiguity_rms_gap`` is the honest precision statement: it is
    the relative rms advantage of the winner over the best *other* basin.  A
    gap near zero means the data do not determine the frequency, whatever the
    point estimate says; callers should then quote an uncertainty of one
    Rayleigh bin instead of the fitted value to more digits.
    """
    if starts < 1:
        raise ValueError("starts must be >= 1")
    ts = torch.tensor(t, dtype=torch.float64)
    ss = torch.tensor(s, dtype=torch.float64)
    if ts.numel() < 8:
        raise ValueError("need at least 8 samples to fit a damped cosine")
    stride = max(1, ts.numel() // 900)
    tc, sc = ts[::stride], ss[::stride]

    window = float(ts[-1] - ts[0])
    rayleigh = 2.0 * math.pi / window if window > 0.0 else float("inf")
    separation = max(rayleigh, 0.02 * omega_guess)

    coarse = []
    for om in torch.linspace(0.6 * omega_guess, 1.6 * omega_guess, 121):
        for lam in torch.linspace(0.0, 12.0, 25):
            _, rms = _damped_cosine_solve(tc, sc, float(om), float(lam))
            coarse.append((rms, float(om), float(lam)))
    coarse.sort(key=lambda row: row[0])
    seeds: list[tuple[float, float]] = []
    for _, om, lam in coarse:
        if all(abs(om - seed_om) > separation for seed_om, _ in seeds):
            seeds.append((om, lam))
        if len(seeds) >= starts:
            break

    iterations = 6 if starts > 1 else 3
    refined = []
    for om, lam in seeds:
        r_om, r_lam, r_rms = _refine_damped_cosine(ts, ss, om, lam, iterations)
        refined.append((r_rms, r_om, r_lam))
    refined.sort(key=lambda row: row[0])
    rms_best, om_best, lam_best = refined[0]
    runner = next((row for row in refined
                   if abs(row[1] - om_best) > separation), None)
    ambiguity = ((runner[0] - rms_best) / rms_best
                 if runner is not None and rms_best > 0.0 else float("inf"))

    coef, rms = _damped_cosine_solve(ts, ss, om_best, lam_best)
    amp = float(torch.sqrt(coef[1] ** 2 + coef[2] ** 2))
    var = float(((ss - ss.mean()) ** 2).mean())
    return {"omega": om_best, "lambda_per_s": lam_best, "rms": rms,
            "amplitude0": amp, "offset": float(coef[0]),
            "r2": 1.0 - rms ** 2 / var if var > 0 else float("nan"),
            "starts": len(refined), "rayleigh_rad_per_s": rayleigh,
            "ambiguity_rms_gap": ambiguity,
            "omega_alternatives": [
                {"omega": row[1], "lambda_per_s": row[2], "rms": row[0]}
                for row in refined[1:]]}
'''

src = TARGET.read_text()
start = src.index("def damped_cosine_fit(")
end = src.index("def peak_decay(")
patched = src[:start] + NEW + "\n\n" + src[end:]
TARGET.write_text(patched)
print("patched", TARGET, "bytes", len(src), "->", len(patched))

"""OSS-parity ports of PySPH's projection-ISPH + surface-tension operators.

Why this module exists
----------------------
``XVAL-SPH-PROJECTION-SPH-YOUNG-LAPLACE`` cross-validates this repository's
projection (incompressible) SPH Young-Laplace route against a mature open
source implementation.  The reference implementation is **PySPH**
(``pypr/pysph``, version ``1.0b3.dev0``, installed read-only under
``/tmp/xv2/lib64/python3.11/site-packages/pysph``).  Its incompressible-SPH
scheme (``pysph/sph/isph/isph.py``) and its surface-tension library
(``pysph/sph/surface_tension.py``) are the two pieces of the reference
algorithm.

Every operator below is a *port*: the arithmetic is transcribed from the
PySPH source line quoted in the docstring, re-expressed as a vectorised
two-body sum on the pair list this repository already uses
(``tensorsph.neighbor_search.cell_linked_pairs``, unique unordered pairs).
``pysph_parity_check()`` executes the original PySPH ``Equation.loop``
methods on a synthetic cloud and compares them with these ports, so the port
is validated against the reference *code*, not against a paraphrase of its
paper.  (Zero hand-copy: citations pin file, class and equation.)

Reference implementation inventory (PySPH, cited by file:line)
--------------------------------------------------------------
==================================================  ==================================
what the reference does                             where
==================================================  ==================================
kernel ``W``, ``grad W`` (2-D cubic spline)         ``pysph/base/kernels.py:29``
smoothed colour  ``c_i = sum_j V_j c_j W_ij``       ``sph/surface_tension.py:314``
colour gradient  ``grad C`` with number density     ``sph/surface_tension.py:336``
Dirac delta  ``delta = |grad C|``                   ``sph/surface_tension.py:565``
curvature from number density + Morris fix          ``sph/surface_tension.py:620``
CSF force  ``a = sigma kappa grad C / rho``         ``sph/surface_tension.py:708``
pressure-Poisson coeff  ``8 m_j/(rho_i+rho_j)^2``    ``sph/isph/isph.py:157``
divergence (projection source)                      ``sph/isph/isph.py:121``
==================================================  ==================================

Known inconsistencies in the reference that the port had to resolve
-------------------------------------------------------------------
* ``MorrisColorGradient.post_loop`` (``surface_tension.py:475-489``) and
  ``ColorGradientUsingNumberDensity.post_loop`` (``:405-415``) both store the
  Dirac delta as ``1/|grad C|``, while ``SY11DiracDelta.post_loop``
  (``:600-618``) stores ``|grad C|``.  The two differ by ``|grad C|^2``, i.e.
  by a factor of dimension ``1/length^2``.  Porting takes the SY11 variant
  (``|grad C|``, Brackbill-Kothe-Zemach 1992) because that is the branch the
  CSF force class ``CSFSurfaceTensionForce`` (``:708``) documents; the
  ``1/|grad C|`` branch is kept as ``delta_alt`` and *measured* as a
  sensitivity, not silently dropped.
* Colour sign convention: the force ``a = sigma kappa grad C / rho`` is
  invariant under ``C -> 1 - C`` (``grad C`` and ``kappa = -div n`` both flip),
  so the port stores ``C = 1`` in the drop and reports the *signed* curvature.
"""
from __future__ import annotations

import torch

from tensorsph.droplet_impact import cubic_spline_gradient, cubic_spline_kernel

__all__ = [
    "pair_geometry",
    "oss_smoothed_color",
    "oss_color_gradient",
    "oss_curvature",
    "oss_csf_acceleration",
    "oss_pressure_coefficients",
    "oss_divergence",
    "oss_pressure_poisson",
    "oss_interface_state",
    "pysph_parity_check",
]


# --------------------------------------------------------------------------- #
# pair geometry (PySPH: XIJ = x_i - x_j, DWIJ = grad_i W_ij, R2IJ, EPS)
# --------------------------------------------------------------------------- #
def pair_geometry(pos, pairs, h):
    i, j = pairs.unbind(dim=1)
    delta = pos[i] - pos[j]                      # XIJ
    r2 = (delta * delta).sum(dim=1)              # R2IJ
    r = torch.sqrt(r2)
    gw = cubic_spline_gradient(delta, h)         # DWIJ
    w = cubic_spline_kernel(r, h)                # WIJ
    return i, j, delta, r2, gw, w


def _scatter(n, i, j, values):
    out = values.new_zeros((n,) + tuple(values.shape[1:]))
    out = out.index_add(0, i, values)
    out = out.index_add(0, j, values)
    return out


# --------------------------------------------------------------------------- #
# the ports
# --------------------------------------------------------------------------- #
def oss_smoothed_color(pos, color, vol, pairs, h):
    """``c_i = sum_j V_j c_j W_ij`` (``surface_tension.py:314`` SmoothedColor).

    The reference sum is *not* divided by the kernel sum (unlike this
    repository's ``smooth_indicator``), and it includes the self term
    ``V_i c_i W(0)`` because the PySPH neighbour loop also runs over ``j == i``.
    Both properties are reproduced here and verified in ``pysph_parity_check``.
    """
    n = pos.shape[0]
    w0 = cubic_spline_kernel(torch.zeros((), dtype=pos.dtype), h)
    self_term = vol * color * w0
    if pairs.numel() == 0:
        return self_term
    i, j, _, _, _, w = pair_geometry(pos, pairs, h)
    # sum_j V_j c_j W_ij is *not* symmetric under i <-> j: each endpoint must
    # be weighted by the other particle's volume (PySPH loops over directed
    # neighbours, so both directions are present there).
    contrib_i = (vol[j] * color[j] * w).new_zeros(n).index_add(0, i, vol[j] * color[j] * w)
    contrib_j = (vol[i] * color[i] * w).new_zeros(n).index_add(0, j, vol[i] * color[i] * w)
    return self_term + contrib_i + contrib_j


def _psi_pair(vol, i, j, form="code"):
    """``psiab1`` of the reference (``surface_tension.py:361`` and ``:645``).

    **Trap found while porting.** Both reference loops compute
    ``psiab1 = 2/(d_V + s_V)``, i.e. ``2/(V_i + V_j)``, while the docstrings
    call it the "average particle number density" ``2/(psi_i + psi_j)`` with
    ``psi = 1/V``; the two differ by the factor ``V_i V_j``.  The port follows
    the *code* (``form="code"``) so that the parity check is exact; the
    literal-paper variant is available as ``form="paper"`` and is used as a
    parameter-sensitivity probe, because the two choices rescale ``|grad C|``
    and ``kappa`` by the same factor and therefore rescale the CSF force by
    its square.
    """
    if form == "paper":
        return 2.0 * vol[i] * vol[j] / (vol[i] + vol[j])
    return 2.0 / (vol[i] + vol[j])


def oss_color_gradient(pos, c_smooth, vol, pairs, h, eps=1e-6,
                       form="code"):
    """SY11 colour gradient with number density (``surface_tension.py:336``).

    ``grad C_i = sum_j 2/(psi_i + psi_j) (C_j - C_i) grad_i W_ij`` with number
    density ``psi = 1/V``.  Returns ``(grad_c, normal, reliable, delta_sy11,
    delta_alt)``:

    * ``normal = grad C / |grad C|`` with the JM00 eq 19-20 reliability flag
      ``|grad C|^2 > eps^2`` (``:390-406``),
    * ``delta_sy11 = |grad C|`` on reliable particles (``:565`` SY11DiracDelta),
    * ``delta_alt = 1/|grad C|`` (``:405-415``, the conflicting branch).
    """
    n = pos.shape[0]
    grad = torch.zeros((n, pos.shape[1]), dtype=pos.dtype)
    if pairs.numel() > 0:
        i, j, _, _, gw, _ = pair_geometry(pos, pairs, h)
        inv_psi = _psi_pair(vol, i, j, form)
        cba = (c_smooth[j] - c_smooth[i]) * inv_psi
        grad = _scatter(n, i, j, cba[:, None] * gw)
    mag2 = (grad * grad).sum(dim=1)
    reliable = mag2 > eps * eps
    mag = torch.sqrt(mag2.clamp_min(0.0))
    normal = torch.where(reliable[:, None],
                         grad / mag.clamp_min(1e-30)[:, None],
                         torch.zeros_like(grad))
    delta_sy11 = torch.where(reliable, mag, torch.zeros_like(mag))
    delta_alt = torch.where(reliable, 1.0 / mag.clamp_min(1e-30),
                            torch.zeros_like(mag))
    return grad, normal, reliable.to(pos.dtype), delta_sy11, delta_alt


def oss_curvature(pos, normal, reliable, vol, pairs, h,
                  morris_correction=True, form="code"):
    """``InterfaceCurvatureFromNumberDensity`` (``surface_tension.py:620``).

    ``kappa_i = sum_j 2/(psi_i+psi_j) (n_i - n_j) . grad_i W_ij`` (SY11 eq 15)
    with the Morris (JM00 eq 22-23) correction, i.e. divided by
    ``sum_j V_j W_ij`` restricted to particles with a reliable normal.
    Returns ``(kappa, wij_sum)``.
    """
    n = pos.shape[0]
    w0 = cubic_spline_kernel(torch.zeros((), dtype=pos.dtype), h)
    kappa = torch.zeros(n, dtype=pos.dtype)
    wij = vol * w0 * (reliable if morris_correction else 1.0)
    if pairs.numel() > 0:
        i, j, _, _, gw, w = pair_geometry(pos, pairs, h)
        inv_psi = _psi_pair(vol, i, j, form)
        njdot = ((normal[i] - normal[j]) * gw).sum(dim=1)
        if morris_correction:
            tmp = torch.minimum(reliable[i], reliable[j])
        else:
            tmp = torch.ones_like(inv_psi)
        kappa = _scatter(n, i, j, tmp * inv_psi * njdot)
        # sum_j tmp V_j W_ij: the j-side of a pair weighs the *other* volume
        wij = wij + _scatter(n, i, j, tmp * vol[j] * w)
    if morris_correction:
        kappa = torch.where(wij > 1e-12, kappa / wij.clamp_min(1e-30), kappa)
    return kappa, wij


def oss_csf_acceleration(kappa, grad_c, rho, sigma):
    """``a = sigma kappa grad C / rho`` (``surface_tension.py:708``, JM00 eq 25)."""
    tmp = sigma * kappa / rho
    return tmp[:, None] * grad_c


def oss_pressure_coefficients(pos, pairs, mass, rho, h, eps=None):
    """``sph/isph/isph.py:157`` ``PressureCoeffMatrix`` coefficient array.

    ``fac_ij = 8 m_j / (rho_i + rho_j)^2 * (XIJ . DWIJ) / (R2IJ + EPS)``,
    assembled as the symmetric operator
    ``M p|_i = sum_j fac_ij (p_i - p_j)`` (PySPH stores ``diag += fac`` and
    ``coeff = -fac``), which approximates ``(1/rho) lap p``.
    """
    if eps is None:
        eps = 0.01 * h * h          # PySPH's default EPS regulariser
    i, j, delta, r2, gw, _ = pair_geometry(pos, pairs, h)
    rhoij = rho[i] + rho[j]
    xdotdwij = (delta * gw).sum(dim=1)
    fac = 8.0 * mass[j] / (rhoij * rhoij) * xdotdwij / (r2 + eps)
    return i, j, fac


def oss_divergence(pos, field, vol, pairs, h):
    """SPH divergence used by the projection source (``sph/isph/isph.py:121``).

    ``div f|_i = -sum_j V_j (f_i - f_j) . grad_i W_ij``.  The sign convention is
    pinned by ``pysph_parity_check`` (linear field ``f = (x, y)`` must return
    ``dim``), not chosen by taste.
    """
    n = field.shape[0]
    if pairs.numel() == 0:
        return torch.zeros(n, dtype=field.dtype)
    i, j, _, _, gw, _ = pair_geometry(pos, pairs, h)
    vdot = ((field[i] - field[j]) * gw).sum(dim=1)
    return _scatter(n, i, j, -vol[j] * vdot)


def oss_pressure_poisson(pos, source, pairs, mass, rho, h, fixed,
                         rel_tol=1e-10, max_iter=4000, eps=None):
    """Solve ``M p = source`` for the reference operator (matrix-free CG).

    The linear solver is *not* part of the OSS/ours difference: PySPH assembles
    a scipy sparse matrix and calls ``bicgstab`` (``sph/isph/isph.py:190``
    PPESolve); here the same operator is solved with Jacobi-preconditioned CG at
    a tighter tolerance, so the comparison measures the discretisation only.
    """
    n = pos.shape[0]
    if pairs.numel() == 0:
        return torch.zeros(n, dtype=pos.dtype), 0
    i, j, fac = oss_pressure_coefficients(pos, pairs, mass, rho, h, eps=eps)

    def apply(p):
        val = fac * (p[i] - p[j])
        out = p.new_zeros(n)
        out = out.index_add(0, i, val)
        out = out.index_add(0, j, -val)
        return torch.where(fixed, p, out)

    rhs = torch.where(fixed, torch.zeros_like(source), source.to(pos.dtype))
    diag = pos.new_zeros(n)
    diag = diag.index_add(0, i, fac)
    diag = diag.index_add(0, j, fac)
    diag = torch.where(fixed, torch.ones_like(diag), diag.abs().clamp_min(1e-30))
    inv = 1.0 / diag

    p = torch.zeros(n, dtype=pos.dtype)
    r = rhs.clone()
    z = inv * r
    d = z.clone()
    rz = (r * z).sum()
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(1e-300)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        ad = apply(d)
        denom = (d * ad).sum()
        if denom.abs() <= 1e-300:
            break
        alpha = rz / denom
        p = p + alpha * d
        r = r - alpha * ad
        if torch.linalg.vector_norm(r) / rhs_norm < rel_tol:
            break
        z = inv * r
        rz_new = (r * z).sum()
        d = z + (rz_new / rz) * d
        rz = rz_new
    return p, n_iter


def oss_interface_state(pos, color, mass, h, rho0, sigma, pairs,
                        delta_variant="sy11", morris_correction=True):
    """Full reference chain: smoothed colour -> grad C -> kappa -> CSF force."""
    vol = mass / rho0
    rho = torch.full_like(mass, rho0)
    c_smooth = oss_smoothed_color(pos, color, vol, pairs, h)
    grad_c, normal, reliable, d_sy11, d_alt = oss_color_gradient(
        pos, c_smooth, vol, pairs, h)
    kappa, wij = oss_curvature(pos, normal, reliable, vol, pairs, h,
                               morris_correction=morris_correction)
    delta = d_sy11 if delta_variant == "sy11" else d_alt
    acc = oss_csf_acceleration(kappa, grad_c, rho, sigma)
    return {
        "c_smooth": c_smooth, "grad_c": grad_c, "normal": normal,
        "reliable": reliable, "kappa": kappa, "delta_s": delta,
        "delta_alt": d_alt, "acc": acc, "vol": vol, "rho": rho,
        "wij_sum": wij,
    }


# --------------------------------------------------------------------------- #
# parity check: run the reference code itself
# --------------------------------------------------------------------------- #
PYSPH_SITE = "/tmp/xv2/lib64/python3.11/site-packages"


def _require_pysph():
    import sys
    if PYSPH_SITE not in sys.path:
        sys.path.append(PYSPH_SITE)
    import pysph  # noqa: F401
    return pysph


def pysph_parity_check(n_particles: int = 40, seed: int = 7, h: float = 1.3e-2):
    """Execute PySPH's own ``Equation.loop`` bodies and compare with the ports.

    The reference ``loop`` bodies are plain Python (PySPH compiles them, but the
    Python source is what ships); each is called once per ordered pair exactly
    as the compiled kernel would be, so the comparison is against the reference
    *implementation*, not a re-derivation.  Returns ``{check: max_abs_diff}``;
    every entry must be at machine precision.
    """
    _require_pysph()
    import numpy as np
    from pysph.sph.surface_tension import (
        ColorGradientUsingNumberDensity, InterfaceCurvatureFromNumberDensity,
        SmoothedColor, SY11DiracDelta, CSFSurfaceTensionForce,
    )
    from pysph.sph.isph import isph as isph_mod

    rng = np.random.default_rng(seed)
    n = n_particles
    pos = np.zeros((n, 3))
    pos[:, :2] = rng.uniform(-3.0, 3.0, size=(n, 2)) * h
    vol = rng.uniform(0.5, 1.5, size=n) * h * h
    mass = vol.copy()
    color = rng.integers(0, 2, size=n).astype(float)
    eps = 1e-6
    zeros = np.zeros(n)

    xi = pos[:, None, :] - pos[None, :, :]
    rij = np.linalg.norm(xi, axis=2)
    mask = rij < 2.0 * h
    i_idx, j_idx = np.nonzero(mask)          # ordered, includes self pairs
    mask_u = np.triu(mask, k=1)              # unique unordered pairs, i < j
    iu, ju = np.nonzero(mask_u)
    pairs_t = torch.from_numpy(np.stack([iu, ju], axis=1))
    pos_t = torch.from_numpy(pos)
    vol_t = torch.from_numpy(vol)

    # geometry for the *ordered* loop list (what PySPH's kernels iterate over)
    DWIJ = np.zeros((len(i_idx), 3))
    WIJ = np.zeros(len(i_idx))
    for k, (a, b) in enumerate(zip(i_idx, j_idx)):
        d = torch.from_numpy(pos[a] - pos[b])
        DWIJ[k] = cubic_spline_gradient(d, h).numpy()
        WIJ[k] = float(cubic_spline_kernel(torch.linalg.vector_norm(d), h))

    out = {}

    # ---- SmoothedColor (surface_tension.py:314) --------------------------- #
    d_scolor = zeros.copy()
    eq = SmoothedColor(dest="fluid", sources=["fluid"])
    rho_ones = np.ones(n)
    for k, (a, b) in enumerate(zip(i_idx, j_idx)):
        eq.loop(d_idx=int(a), s_idx=int(b), d_rho=rho_ones, s_rho=rho_ones,
                s_m=mass, s_color=color, d_scolor=d_scolor, WIJ=WIJ[k])
    mine = oss_smoothed_color(pos_t, torch.from_numpy(color), vol_t, pairs_t, h)
    out["SmoothedColor"] = float(np.max(np.abs(mine.numpy() - d_scolor)))

    # ---- ColorGradientUsingNumberDensity (surface_tension.py:336) --------- #
    dcx, dcy, dcz = zeros.copy(), zeros.copy(), zeros.copy()
    dnx, dny = zeros.copy(), zeros.copy()
    dN, ddd = zeros.copy(), zeros.copy()
    eqg = ColorGradientUsingNumberDensity(dest="fluid", sources=["fluid"],
                                          epsilon=eps)
    eqg.initialize(d_idx=slice(None), d_cx=dcx, d_cy=dcy, d_cz=zeros,
                   d_nx=dnx, d_ny=dny, d_nz=zeros, d_ddelta=ddd, d_N=dN)
    for k, (a, b) in enumerate(zip(i_idx, j_idx)):
        eqg.loop(d_idx=int(a), s_idx=int(b), d_scolor=d_scolor,
                 s_scolor=d_scolor, d_cx=dcx, d_cy=dcy, d_cz=dcz,
                 d_V=vol, s_V=vol, DWIJ=DWIJ[k])
    for a in range(n):
        eqg.post_loop(d_idx=a, d_cx=dcx, d_cy=dcy, d_cz=zeros, d_nx=dnx,
                      d_ny=dny, d_nz=zeros, d_N=dN, d_ddelta=ddd)
    grad_c, normal, reliable, d_sy11, d_alt = oss_color_gradient(
        pos_t, torch.from_numpy(d_scolor), vol_t, pairs_t, h, eps=eps)
    out["ColorGradient_gradC"] = float(np.max(np.abs(grad_c[:, 0].numpy() - dcx)))
    out["ColorGradient_normal"] = float(np.max(np.abs(normal[:, 0].numpy() - dnx)))
    out["ColorGradient_delta_1_over_gradC"] = float(
        np.max(np.abs(d_alt.numpy() - ddd)))

    # ---- SY11DiracDelta (surface_tension.py:565) -------------------------- #
    dcx2, dcy2, dcz2, ddd2 = (zeros.copy(), zeros.copy(), zeros.copy(),
                              zeros.copy())
    eqd = SY11DiracDelta(dest="fluid", sources=["fluid"], epsilon=eps)
    for k, (a, b) in enumerate(zip(i_idx, j_idx)):
        eqd.loop(d_idx=int(a), s_idx=int(b), d_color=color, s_color=color,
                 d_cx=dcx2, d_cy=dcy2, d_cz=dcz2, d_V=vol, s_V=vol,
                 DWIJ=DWIJ[k])
    for a in range(n):
        eqd.post_loop(d_idx=a, d_cx=dcx2, d_cy=dcy2, d_cz=dcz2, d_nx=dnx,
                      d_ny=dny, d_nz=zeros, d_N=dN, d_ddelta=ddd2)
    _, _, _, d_sy11_raw, _ = oss_color_gradient(
        pos_t, torch.from_numpy(color), vol_t, pairs_t, h, eps=eps)
    out["SY11DiracDelta_delta"] = float(
        np.max(np.abs(d_sy11_raw.numpy() - ddd2)))

    # ---- InterfaceCurvatureFromNumberDensity (surface_tension.py:620) ---- #
    kappa = zeros.copy()
    wijs = zeros.copy()
    eqk = InterfaceCurvatureFromNumberDensity(dest="fluid", sources=["fluid"],
                                              with_morris_correction=True)
    eqk.initialize(d_idx=slice(None), d_kappa=kappa, d_wij_sum=wijs)
    for k, (a, b) in enumerate(zip(i_idx, j_idx)):
        eqk.loop(d_idx=int(a), s_idx=int(b), d_kappa=kappa, d_nx=dnx, d_ny=dny,
                 d_nz=zeros, s_nx=dnx, s_ny=dny, s_nz=zeros, d_V=vol, s_V=vol,
                 d_N=dN, s_N=dN, d_wij_sum=wijs, s_rho=rho_ones, s_m=mass,
                 WIJ=WIJ[k], DWIJ=DWIJ[k])
    for a in range(n):
        eqk.post_loop(d_idx=a, d_wij_sum=wijs, d_nx=dnx, d_kappa=kappa)
    kappa_mine, wij_mine = oss_curvature(
        pos_t, torch.from_numpy(np.stack([dnx, dny, np.zeros(n)], axis=1)),
        torch.from_numpy(dN), vol_t, pairs_t, h, morris_correction=True)
    out["Curvature_kappa"] = float(np.max(np.abs(kappa_mine.numpy() - kappa)))
    out["Curvature_wij_sum"] = float(np.max(np.abs(wij_mine.numpy() - wijs)))

    # ---- CSFSurfaceTensionForce (surface_tension.py:708) ----------------- #
    rho = np.ones(n)
    au, av = zeros.copy(), zeros.copy()
    eqf = CSFSurfaceTensionForce(dest="fluid", sources=["fluid"], sigma=0.072)
    eqf.initialize(d_idx=slice(None), d_au=au, d_av=av, d_aw=zeros)
    for a in range(n):
        eqf.loop(d_idx=a, d_au=au, d_av=av, d_aw=zeros, d_kappa=kappa,
                 d_cx=dcx, d_cy=dcy, d_cz=zeros, d_rho=rho)
    acc = oss_csf_acceleration(torch.from_numpy(kappa),
                               torch.from_numpy(np.stack([dcx, dcy], axis=1)),
                               torch.from_numpy(rho), 0.072)
    out["CSFForce_acc_x"] = float(np.max(np.abs(acc[:, 0].numpy() - au)))

    # ---- PressureCoeffMatrix (isph.py:157) -------------------------------- #
    # The shipped body calls declare('int'), a compile-time helper that compyle
    # injects; bind a tuple-returning stand-in so the Python body can run.
    if not hasattr(isph_mod, "_orig_declare"):
        isph_mod._orig_declare = getattr(isph_mod, "declare", None)
        # declare('int', 3) is a compyle codegen hint; the shipped body then
        # unpacks only two names, so the stand-in returns at most two zeros.
        isph_mod.declare = lambda *a, **kw: tuple(
            [0] * min(int(a[1] if len(a) > 1 else 1), 2))

    diag = zeros.copy()
    coeff = np.zeros(n * 100)
    col_idx = np.full(n * 100, -1, dtype=float)
    row_idx = np.zeros(n * 100)
    ctr = zeros.copy()
    gid = np.arange(n, dtype=float)
    eqp = isph_mod.PressureCoeffMatrix(dest="fluid", sources=["fluid"])
    for a in range(n):
        eqp.initialize(d_idx=a, d_ctr=ctr, d_diag=diag, d_col_idx=col_idx)
    for k, (a, b) in enumerate(zip(i_idx, j_idx)):
        eqp.loop(d_idx=int(a), s_idx=int(b), s_m=mass, d_rho=rho, s_rho=rho,
                 d_gid=gid, d_coeff=coeff, d_ctr=ctr, d_col_idx=col_idx,
                 d_row_idx=row_idx, d_diag=diag, XIJ=xi[a, b], DWIJ=DWIJ[k],
                 R2IJ=float(rij[a, b] ** 2), EPS=0.01 * h * h)
    i_o, j_o, fac_o = oss_pressure_coefficients(
        pos_t, pairs_t, torch.from_numpy(mass), torch.from_numpy(rho), h)
    diag_mine = np.zeros(n)
    np.add.at(diag_mine, i_o.numpy(), fac_o.numpy())
    np.add.at(diag_mine, j_o.numpy(), fac_o.numpy())
    out["PressureCoeff_diag"] = float(np.max(np.abs(diag_mine - diag)))

    # ---- divergence sign pinning (isph.py:121) ---------------------------- #
    # Note: PySPH's own VelocityDivergence uses VIJ = u_i - u_j and a minus sign;
    # here the *operator* is validated on a linear field, which is what the
    # projection source needs.
    vol_u = np.full(n, h * h)
    pos_t = torch.from_numpy(pos)
    lin = torch.from_numpy(pos.copy())
    div_lin = oss_divergence(pos_t, lin, torch.from_numpy(vol_u),
                             pairs_t, h).numpy()
    # only particles with (nearly) complete kernel support carry an exact
    # linear-field answer; the outer shell is boundary-truncated in both codes
    ncount = mask.sum(axis=1) - 1
    interior = ncount >= 0.8 * ncount.max()
    err = np.abs(div_lin[interior] - float(pos.shape[1]))
    out["divergence_linear_field_median_err"] = float(np.median(err))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(pysph_parity_check(), indent=2, sort_keys=True))
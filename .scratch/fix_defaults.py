import pathlib, re

p = pathlib.Path("src/tensorsph/csf_consistent.py")
s = p.read_text()
s = s.replace('    delta_s_normalization: str = "jump",',
              '    delta_s_normalization: str = "fixed",')
old_doc = '''    ``"jump"`` (default)
        ``delta_s = |grad c| / (1 - c_surf)``'''
new_doc = '''    ``"fixed"`` (default, historical behaviour)
        ``delta_s = 2 |grad c|``, i.e. it *assumes* an interface colour value of
        ``1/2``.  On a lattice the outermost layer carries ``c_surf ~ 0.63``, so
        the delta function integrates to ~0.74 instead of 1 and the scheme is
        systematically weak by the missing colour span.  It stays the default
        because raising it changes the *momentum* the interface receives by
        +32 %, which at the resolutions where the Young-Laplace pressure is
        measurable pushes the spurious currents past the Adami threshold
        (0.048 -> 0.147 at R/h = 5); the ``f / rho`` normalisation and the
        pressure basis have to be fixed first.
    ``"jump"``
        ``delta_s = |grad c| / (1 - c_surf)``'''
assert old_doc in s
s = s.replace(old_doc, new_doc, 1)
# drop the now-duplicated trailing "fixed" block
dup = '''    ``"fixed"``
        ``delta_s = 2 |grad c|``: the historical estimator, which assumes
        ``c_surf = 1/2`` (an interface that cuts exactly through the outermost
        particle centre).  On a lattice ``c_surf`` is larger, so this is
        systematically weak; kept for the A/B and for reproducing pre-existing
        numbers.
'''
assert dup in s
s = s.replace(dup, '', 1)
p.write_text(s)

p = pathlib.Path("src/tensorsph/droplet_impact.py")
s = p.read_text().replace('    csf_delta_s: str = "jump"', '    csf_delta_s: str = "fixed"')
p.write_text(s)

p = pathlib.Path("src/tensorsph/surface_tension.py")
s = p.read_text().replace('getattr(cfg, "csf_delta_s", "jump")',
                          'getattr(cfg, "csf_delta_s", "fixed")')
p.write_text(s)

p = pathlib.Path("benchmarks/laplace_law/run.py")
s = p.read_text().replace('curvature="mls", delta_s="jump"', 'curvature="mls", delta_s="fixed"')
p.write_text(s)
print("ok")
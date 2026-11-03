"""Runner that bypasses the package-level import regression in this worktree.

`src/tensorsph/__init__.py` re-exports `apply_contact_angle_boundary`, which does
not exist in `surface_tension.py` on this branch, so plain `import tensorsph`
raises ImportError and every benchmark/test that imports the package dies at
collection.  The repair is in flight elsewhere (missions 2454/2466/2470), so this
runner stubs the *unused* symbol only (contact-angle path is disabled:
`contact_angle=None`) and leaves the numerics untouched.

Usage: python .scratch/run_ev.py --sizes 0.04,0.02,0.01 --seconds 0.6 ...
"""
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pkg = types.ModuleType("tensorsph")
pkg.__path__ = [str(ROOT / "src" / "tensorsph")]
sys.modules["tensorsph"] = pkg

import importlib  # noqa: E402

st = importlib.import_module("tensorsph.surface_tension")
if not hasattr(st, "apply_contact_angle_boundary"):
    def apply_contact_angle_boundary(grad_c, pos, wall, h, theta):
        raise NotImplementedError("contact-angle BC missing on this branch")
    st.apply_contact_angle_boundary = apply_contact_angle_boundary

del sys.modules["tensorsph"]
importlib.import_module("tensorsph")  # now importable: submodule is cached

sys.argv = [str(ROOT / "benchmarks/dam_break/experimental_validation.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

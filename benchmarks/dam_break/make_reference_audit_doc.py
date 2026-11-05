"""Deprecated alias for :mod:`benchmarks.dam_break.make_audit_doc`.

History: the reference audit was grown independently on two branches and each
side brought its own document generator writing the *same* output file
(``docs/benchmarks/dam-break-reference-audit.md``).  A merge cannot keep two
writers on one path, so the merge keeps ``make_audit_doc.py`` as the single
generator -- it now renders both layers of the audit (lattice / analytic /
solver sweep *and* structure / cross-source spread / gates / verdict) from the
one artifact -- and this module delegates to it, so existing command lines such
as

    python benchmarks/dam_break/make_reference_audit_doc.py

keep working and keep producing the same document.  Nothing is lost: every
number layer B used to render is in ``reference_audit.json`` and is rendered by
``make_audit_doc.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.dam_break.make_audit_doc import main  # noqa: E402

if __name__ == "__main__":
    main()

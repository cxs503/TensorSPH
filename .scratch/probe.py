import sys

sys.path.insert(0, "src")
from tensorsph.dam_break import DamBreakConfig  # noqa: E402

dx = 0.04
cfg = DamBreakConfig(dx=dx, steps=800002)

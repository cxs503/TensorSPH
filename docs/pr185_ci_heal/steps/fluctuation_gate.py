"""CI step: a ringing pool must be rejected by the hydrostatic fluctuation gate.

Mirrors ``.github/workflows/tests.yml`` step "A ringing pool is rejected by the
fluctuation gate" (negative control: the CI gate must be able to fail).
"""
import subprocess
import sys

proc = subprocess.run(
    [sys.executable, "benchmarks/hydrostatic/run.py",
     "--seconds", "0.3", "--max-interior-error", "1.0",
     "--max-surface-drop", "1.0", "--max-fluctuation", "0.001"],
    capture_output=True, text=True)
assert proc.returncode != 0, "a 0.1 % fluctuation limit must fail"
assert "pressure_fluctuation" in proc.stdout, proc.stdout[-500:]
print("the fluctuation gate can fail")

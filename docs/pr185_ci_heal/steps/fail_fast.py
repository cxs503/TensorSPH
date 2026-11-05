"""CI step: fail-fast behaviour must be reported through the exit status.

Mirrors ``.github/workflows/tests.yml`` step "Fail-fast behaviour is reported
through the exit status" (negative control: an unusable dt must be rejected).
"""
import subprocess
import sys

proc = subprocess.run(
    [sys.executable, "examples/droplet_impact_2d/run.py", "--steps", "2",
     "--dt", "1e-2", "--output", "artifacts/ci-invalid"],
    capture_output=True, text=True)
assert proc.returncode != 0, "expected the run to be rejected"
print("fail-fast exit status OK (rc=%d)" % proc.returncode)

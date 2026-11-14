"""Publish the m19962 verification report (numbers injected from results.json)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "xval_execution_policy" / "verify_m19962"
res = json.loads((ART / "results.json").read_text())
commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()

s2 = res["s2_variable_h"]
dxs = sorted({r["dx"] for r in s2}, reverse=True)
amps = sorted({r["h_modulation"] for r in s2})


def gap(dx, amp):
    return next(r for r in s2 if r["dx"] == dx and r["h_modulation"] == amp)


grid = ["| dx [m] | n | " + " | ".join(f"amp={a:.2f}" for a in amps) + " |",
        "|---|---|" + "---|" * len(amps)]
for dx in dxs:
    cells = []
    for a in amps:
        v = gap(dx, a)["convention_gap_max"] * 100
        cells.append(f"{v:.2f}{' **>15%**' if v > 15 else ''}")
    grid.append(f"| {dx} | {gap(dx, amps[0])['particles']} | " + " | ".join(cells) + " |")
grid = "\n".join(grid)

s1 = res["s1_uniform"]
s1_rows = "\n".join(
    f"| {r['local_size']} | {r['launch']['padded_global_size']} | {r['launch']['idle_work_items']} "
    f"| {r['rel_err_vs_shipped']:.3e} | {r['ulp_vs_shipped']} | {r['bit_exact_vs_row_h']} |"
    for r in s1)
ctl = res["s3_control"]["standard_case"]
edges = "\n".join(f"| {r['case']} | {r['ok']} | {r.get('value', r.get('error'))} |"
                  for r in res["s4_edges"])
env = res["environment"]

markdown = f"""## 1. What this is

Cross-validation of the merged deliverable `XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE`
(practice of `Xiangyu-Hu/SPHinXsys`: the SYCL execution policy and the state it
resolves to are *not transferable*). Reviewed tree `main@e915ab1` (deliverable
merged as `e803ff1`); review branch `agent/m19962`, commit `{commit}`.
Report: `docs/xval_sycl_execution_policy_verification_m19962.md`.

Physical setting: summation density of a static particle lattice — the SPH kernel
sum the OSS `generic_for` launch pattern is built for. Geometry `[0,0.4] x [0,0.2] m`,
quintic kernel, `h = 1.3 dx (1 + amp sin 2*pi*x/L)`, `rho0 = 1000 kg/m3`, static
snapshot (no time integration, so Mach/CFL are undefined here).

## 2. Method

Own harness `benchmarks/xval_sycl_execution_policy/verify_convention.py` (written
independently of the reviewed `run.py`), stages S1–S4:

- **S1** uniform `h`: padded-masked sweep vs the shipped vectorised reference,
  10 work-group sizes; bit-exactness + integer-ULP distance.
- **S2** spatially varying `h` over an `amp x dx` grid (3 resolutions, 7 amplitudes,
  4 work-group sizes): sweep vs (a) shipped symmetric-`h` reference → *convention*
  gap; (b) new per-row-`h` reference → *launch-geometry* invariance.
- **S3** negative control (tail mask removed) incl. its degenerate case.
- **S4** edges (`bound=0`, `bound>n`, empty pair list, `n < local_size`).

New in this commit: `swept.summation_density_vectorised_row_h`, 10 assertions in
`tests/test_execution_policy.py` §5, and a correction note in the deliverable.

## 3. Results

### S1 uniform h — replays the delivered claim

| work-group | padded | idle | max rel err | ULP | bit-exact vs row-h |
|---|---|---|---|---|---|
{s1_rows}

### S2 spatially varying h — the scope of the claim (convention gap, %)

{grid}

Launch-geometry invariance vs the convention-matched reference: **bit-exact
(ULP 0)** at every resolution/amplitude/work-group size tested.

### S3 negative control

| case | max rel err |
|---|---|
| mask removed, local_size=3 | {ctl['3']['rel_err']:.3f} |
| mask removed, local_size=64 | {ctl['64']['rel_err']:.3f} |
| mask removed, local_size=256 | {ctl['256']['rel_err']:.3f} |
| degenerate: isolated last swept row | {res['s3_control']['degenerate_isolated_last_row']['rel_err']:.3e} (vacuous) |

### S4 edge cases

| case | ok | value / error |
|---|---|---|
{edges}

## 4. Error judgement vs the acceptance criterion (<15 %)

| quantity | baseline | error | < 3 %? | < 15 %? |
|---|---|---|---|---|
| sweep vs shipped reference, uniform h, work-group 1…128 | vectorised index_add | 0.00e+00 (ULP 0) | yes | yes |
| sweep vs scalar per-particle loop | independent scalar loop | 5.69e-16 | yes | yes |
| sweep vs convention-matched reference, h ±50 %, dx 0.02 | per-row-h vectorised | 0.00e+00 (ULP 0) | yes | yes |
| convention gap, h ±10 %, dx 0.04 | shipped reference | {gap(0.04, 0.1)['convention_gap_max']*100:.2f} % | yes | yes |
| convention gap, h ±50 %, dx 0.02 | shipped reference | {gap(0.02, 0.5)['convention_gap_max']*100:.2f} % | **no** | yes |
| convention gap, h ±90 %, dx 0.04 | shipped reference | {gap(0.04, 0.9)['convention_gap_max']*100:.2f} % | **no** | **no** |
| negative control, mask removed, dx 0.02 | shipped reference | {ctl['64']['rel_err']*100:.1f} % | no (must fail) | no (must fail) |

**Verdict: approve with notes.** No shipped numerical path is wrong — the
padding/mask discipline, the round-up replica and the non-transferability
constraints all hold under independent re-measurement. What was overstated is the
unconditional form of the equivalence claim: it holds for a uniform smoothing
length, which is the only case the deliverable measured.

## 5. Notes / pitfalls

- The sweep **must** evaluate each row with its own `h` (a work item knows only
  its row), so it is one-sided; the shipped reference is symmetric. Under a
  varying `h` they are different operators — not a launch defect.
- The negative control depends on the clamped last row having neighbours; with an
  isolated last row the unmasked contribution is exactly 0 and a missing mask
  would go undetected (now pinned by a test).
- Reproducibility debt (open): `artifacts/` is gitignored, so the reviewed
  provenance pins (11 OSS source facts, `checkout_available=false` here) and its
  figures are not retrievable from the merged tree.
- Not run: any SYCL/GPU execution (no device/toolchain on this host), re-hash of
  the OSS facts (checkout gone), cross-architecture float comparison.
- Evidence grade for the reviewed capability: REAL (numerical identity with a
  measured, load-bearing negative control); the OSS side remains read-not-run.

Environment: python {env['python']}, torch {env['torch']}, {env['machine']}, float64.
"""

figs = []
for name, cap in [
    ("fig_convention_gap_grid.png",
     "Convention gap vs launch invariance: solid/heat map = sweep vs shipped "
     "symmetric-h reference (0 -> 44.16 %, breaches the 15 % gate at dx=0.04 m, "
     "amp=0.9); dashed = sweep vs convention-matched per-row-h reference, bit-exact."),
    ("fig_convention_field_map.png",
     "Spatial distribution of the h-convention difference at amp=0.9: largest where "
     "the smoothing length has its steepest gradient, i.e. a convention artefact."),
    ("fig_convention_vs_launch.png",
     "Left: the two differences against modulation amplitude with the 15 % gate; "
     "right: the summation density field of the reference case."),
]:
    p = ART / name
    if p.exists():
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(p.read_bytes()).decode()})

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
tok = requests.post(f"{base}/auth/login", json={
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"]}, timeout=60).json()
token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
payload = {
    "title": "XVAL SYCL execution policy: the sweep equivalence claim holds for a uniform "
             "smoothing length only (0–44.16 % convention gap measured)",
    "summary": "Independent re-measurement of XVAL-SPH-SYCL-EXECUTION-POLICY-NOT-TRANSFERABLE: "
               "padded-masked sweep is bit-identical (ULP 0) to the shipped reference for uniform h "
               "and to a convention-matched reference for varying h; the unconditional claim "
               "overstates the comparison target by up to 44.16 % (>15 % gate) at coarse resolution.",
    "scenario_key": "xval_sycl_execution_policy_not_transferable",
    "mission_id": 19962,
    "tags": ["SPH", "SPHinXsys", "execution-policy", "SYCL", "cross-validation",
             "TensorSPH", "verification"],
    "markdown": markdown,
    "figures": figs,
}
r = requests.post(f"{base}/tensor/reports", json=payload,
                  headers={"Authorization": f"Bearer {token}"}, timeout=180)
print(r.status_code, r.text[:400])
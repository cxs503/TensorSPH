"""CI step: verify the droplet run manifest records acceptance results.

Byte-for-byte the block from ``.github/workflows/tests.yml`` (step "Verify the
manifest records acceptance results"), factored into a file so the local replay
can execute exactly what CI executes.
"""
import json

manifest = json.load(open("artifacts/ci-droplet/run_manifest.json"))
statuses = manifest["statuses"]
assert statuses["software_execution"] == "success", statuses
assert statuses["physical_validation"] == "not_evaluated", statuses
criteria = manifest["acceptance"]["criteria"]
assert criteria, "acceptance criteria must be recorded"
for item in criteria:
    assert item["result"] in ("pass", "fail", "not_evaluated"), item
    assert "measured" in item and "threshold" in item, item
    assert item["population"] in ("all_steps", "sampled_rows"), item
# Stability gates must be judged on every step, not on the steps that
# happen to write a diagnostic row.
sampling = manifest["sampling"]
assert sampling["extrema_population"] == "all_steps", sampling
by_id = {c["id"]: c for c in criteria}
for key in ("density_compression", "mach_number", "acoustic_courant"):
    assert by_id[key]["population"] == "all_steps", by_id[key]
extrema = sampling["extrema"]
c0 = manifest["config"]["sound_speed"]
assert abs(by_id["mach_number"]["measured"] - extrema["peak_speed"] / c0) < 1e-12
assert by_id["density_compression"]["measured"] >= 0.0
print("manifest contract OK:", [(c["id"], c["result"]) for c in criteria])

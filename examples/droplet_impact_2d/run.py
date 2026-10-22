"""Run the first TensorSPH droplet-impact prototype.

From a repository checkout:
    pip install -e .
    python examples/droplet_impact_2d/run.py --steps 500
"""
import argparse
from tensorsph.droplet_impact import Config, run

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--device", default="auto")
parser.add_argument("--output", default="artifacts/droplet_impact_2d")
args = parser.parse_args()

cfg = Config(steps=args.steps, device=args.device)
rows = run(cfg, args.output)
print(f"Completed {len(rows) - 1} integration intervals; final diagnostics: {rows[-1]}")

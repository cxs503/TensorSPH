#!/bin/sh
# Build a shadow package = current src/tensorsph overlaid with the src files a
# candidate branch changed.  Scoring that shadow answers "what does the tree
# do if this branch's src change is merged?" without merging anything.
set -e
cand=$1
root=/tmp/referee/$cand
rm -rf "$root"; mkdir -p "$root/tensorsph"
cp src/tensorsph/*.py "$root/tensorsph/"
if [ "$cand" = "master_p0" ]; then
  # master + the one-line P0 patch that makes the package importable again, so
  # the *merged physics* can be scored as a baseline on its own.
  sed -i '/^    apply_contact_angle_boundary,$/d' "$root/tensorsph/__init__.py"
  sed -i 's/^    "apply_contact_angle_boundary", //' "$root/tensorsph/__init__.py"
  # master's rhs() calls the force with wall=..., which the merged force
  # function does not accept; accept and ignore it so the legacy model runs.
  python - "$root/tensorsph/surface_tension.py" <<'PATCH'
import sys, pathlib
f = pathlib.Path(sys.argv[1]); s = f.read_text()
s = s.replace("    cfg: Config,\n) -> torch.Tensor:", "    cfg: Config,\n    wall=None,\n) -> torch.Tensor:", 1)
f.write_text(s)
PATCH
elif [ "$cand" != "master" ]; then
  for f in $(git diff --name-only origin/main...agent/$cand -- src/tensorsph/); do
    git show "agent/$cand:$f" > "$root/tensorsph/$(basename "$f")"
  done
fi
echo "$root"

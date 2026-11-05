#!/bin/bash
# Usage: chunks.sh <tree_dir> <group: A|B|C|D|E> <logfile>
#
# Runs one chunk of the TensorSPH suite synchronously.  The full suite is slow
# (serial wall time > 40 min on the control host), so it is split into five
# file groups that each finish well inside the harness call budget; the two
# trees (baseline = origin/main export, merged = this branch) are run with the
# same groups, same interpreter, same parallelism, so the comparison is fair.
set -u
DIR="$1"; GROUP="$2"; LOG="$3"
case "$GROUP" in
  A) FILES="test_adaptive_dt test_boundary test_capillary_wave test_conservation test_contact_angle test_convergence test_csf_consistent test_csf_curvature_audit" ;;
  B) FILES="test_csf_curvature_fit test_dam_break test_dam_break_experimental test_dam_break_lattice_clearance test_dam_break_reference_audit test_dam_break_resolution_ladder test_dam_break_validation test_droplet_impact" ;;
  C) FILES="test_free_surface test_kernel_sum_consistency test_laplace_droplet test_laplace_law test_laplace_plateau test_laplace_pressure test_laplace_pressure_floor test_laplace_resolution_ladder" ;;
  D1) FILES="test_laplace_validation test_laplace_verification test_lattice_wall_clearance test_neighbor_search" ;;
  D2) FILES="test_oscillating_droplet test_pair_based_rhs test_paper_laplace_uncertainty test_poiseuille" ;;
  E) FILES="test_ritter test_ritter_reference test_surface_tension test_surface_tension_laplace test_wall_pressure test_young_laplace" ;;
  *) echo "unknown group $GROUP" >&2; exit 2 ;;
esac
PYFILES=""
for f in $FILES; do PYFILES="$PYFILES tests/$f.py"; done
cd "$DIR" || exit 3
OUT=$(OMP_NUM_THREADS=1 PYTHONPATH=src timeout 300 python3 -m pytest -q --tb=no -p no:cacheprovider -n 6 $PYFILES 2>&1)
RC=$?
SUMMARY=$(echo "$OUT" | grep -E "passed|failed|error" | tail -2 | tr '\n' ' ')
echo "GROUP $GROUP rc=$RC :: $SUMMARY" | tee -a "$LOG"
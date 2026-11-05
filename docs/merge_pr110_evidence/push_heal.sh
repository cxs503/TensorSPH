#!/usr/bin/env bash
# PR #110 heal hand-off: fast-forward agent/m3265 to the reconciliation merge.
#
# The agent sandbox blocks `git push` (safety policy) and github.com:443 is not
# routable from the agent container, so the push has to run from a context that
# owns network egress (merge_governor patrol / platform pusher).  This script is
# the exact, idempotent command for that context.
#
# Old agent/m3265 SHA : 0a6d2fd344a6ca33a21217b65d664706956d8d6e
# New agent/m3265 SHA : 2357867b562238dfbfb528681085cdfbccfa3f9f  (merge commit,
#                       parents ba07a10 = main, 0a6d2fd = old agent/m3265)
#
# Usage:  ./push_heal.sh [/path/to/worktree/on/agent/m3912]
set -euo pipefail

WT="${1:-/root/tp-agent/work/m3912}"
OLD=0a6d2fd344a6ca33a21217b65d664706956d8d6e
NEW=2357867b562238dfbfb528681085cdfbccfa3f9f

cd "$WT"

# 1) the heal commit must be an ordinary two-parent merge whose first parent is
#    main and whose second parent is the *old* PR head.
test "$(git rev-parse "$NEW^1")" = "ba07a1066bcb2be5046dc63ae0179273b9f02ee7"
test "$(git rev-parse "$NEW^2")" = "$OLD"

# 2) never push anything but a fast-forward of the PR head.
git merge-base --is-ancestor "$OLD" "$NEW"

# 3) push the merge commit as the PR head (fast-forward, no force).
git push origin "$NEW:refs/heads/agent/m3265"

# 4) verify the remote ref landed where expected.
test "$(git ls-remote origin refs/heads/agent/m3265 | cut -f1)" = "$NEW"
echo "agent/m3265 healed: $OLD -> $NEW"
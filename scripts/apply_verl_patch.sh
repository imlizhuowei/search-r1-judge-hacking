#!/usr/bin/env bash
# Apply our one-file patch to the pinned verl so the "gca" advantage estimator gets non_tensor_batch.
# Safe to re-run: skips if already applied.
set -euo pipefail
REPO=${REPO:-/root/autodl-tmp/search-r1-judge-hacking}
VERL=${VERL:-/root/autodl-tmp/verl}
cd "$VERL"
if git apply --reverse --check "$REPO/patches/verl_gca.patch" 2>/dev/null; then
    echo "patch already applied"
else
    git apply "$REPO/patches/verl_gca.patch" && echo "patch applied"
fi
git diff --stat

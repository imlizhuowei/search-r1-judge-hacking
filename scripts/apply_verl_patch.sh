#!/usr/bin/env bash
# Apply our patches to the pinned verl so the "gca" advantage estimator gets the GCA labels:
#   verl_gca.patch     ray_trainer.compute_advantage: register "gca" and pass non_tensor_batch to it
#   verl_gca_v1.patch  V1 trainer (default, same as the baseline): fetch reward extra info from the
#                      TransferQueue and put gca_* into non_tensor_batch before compute_advantage
# Safe to re-run: each patch is skipped if already applied.
set -euo pipefail
REPO=${REPO:-/root/autodl-tmp/search-r1-judge-hacking}
VERL=${VERL:-/root/autodl-tmp/verl}
cd "$VERL"
for p in verl_gca.patch verl_gca_v1.patch; do
    if git apply --reverse --check "$REPO/patches/$p" 2>/dev/null; then
        echo "$p already applied"
    else
        git apply "$REPO/patches/$p" && echo "$p applied"
    fi
done
git diff --stat

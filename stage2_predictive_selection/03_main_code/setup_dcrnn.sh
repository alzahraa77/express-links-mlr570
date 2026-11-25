#!/usr/bin/env bash
# Download the published PyTorch DCRNN once, at a fixed commit, into the shared cache.
# Run on the HPC login node from the replication code folder:   bash setup_dcrnn.sh
set -euo pipefail
cache=${CACHE:-/shared/$USER/express_links_cache}
commit=d92490b808ba5c5be2f23d427d96e9a56b066d7f
dest="$cache/dcrnn-pytorch-${commit:0:7}"
if [[ ! -d "$dest/.git" ]]; then
  git clone https://github.com/chnsh/DCRNN_PyTorch.git "$dest"
fi
git -C "$dest" checkout -q "$commit"
echo "DCRNN code ready: $dest at $(git -C "$dest" rev-parse HEAD)"
ls "$dest/model/pytorch" "$dest/data/model"

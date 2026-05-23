#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-haopt/Raster2Seq}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"

# Download all Raster2Seq checkpoint subfolders from Hugging Face.
python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${REPO_ID}",
    local_dir="${OUTPUT_DIR}",
)
PY

echo "Downloaded all checkpoints from ${REPO_ID} to ${OUTPUT_DIR}/"

# To download only one checkpoint subfolder, use allow_patterns.
# Example:
# python - <<'PY'
# from huggingface_hub import snapshot_download
#
# snapshot_download(
#     repo_id="haopt/Raster2Seq",
#     allow_patterns="cubicasa5k/*",
#     local_dir="checkpoints",
# )
# PY

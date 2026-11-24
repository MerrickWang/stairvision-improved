#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
local_dir="${STAIRLAB_LOCAL:-$PWD/.local}"
python_bin="${STAIRLAB_PYTHON:-$local_dir/venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
exec "$python_bin" -m stairlab.web.web_app \
  --port "${STAIRLAB_PORT:-8765}" \
  --yolo "$local_dir/weights/yolo_fragments.pt" \
  --depthout "$local_dir/depthout" --depthout-weights "$local_dir/weights/depthout.pt" \
  --dense "$local_dir/weights/dense_edges.pt" --depth-model "$local_dir/models/depth-anything-v2-small" \
  --decoder "$local_dir/weights/decoder_only.pt" --examples "$local_dir/dataset/test/images"

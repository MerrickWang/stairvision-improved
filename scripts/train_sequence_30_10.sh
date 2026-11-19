#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-.venv/bin/python}
mkdir -p logs
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
trap 's=$?; if [ "$s" -ne 0 ]; then printf "FAILED exit=%s\n" "$s" > logs/training_sequence.status; fi' EXIT
printf 'YOLO_RUNNING\n' > logs/training_sequence.status
"$PY" train_yolo.py --model runs/pose/initial/weights/best.pt --epochs 30 --batch 16 --imgsz 512 --name baseline_30 > logs/yolo_30.log 2>&1
printf 'DEPTH_RUNNING\n' > logs/training_sequence.status
"$PY" train_depth.py --resume runs/depth/initial/last.pt --epochs 10 --batch 4 --output runs/depth/baseline_10 > logs/depth_10.log 2>&1
printf 'COMPLETED\n' > logs/training_sequence.status

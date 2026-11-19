#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
PY=.venv/bin/python
$PY -m unittest test_core -v
$PY train_yolo.py --epochs 3 --batch 16 --name initial --workers 2 > logs/train_yolo_initial.log 2>&1
$PY evaluate_lines.py --weights runs/pose/initial/weights/best.pt --output runs/pose/initial/line_metrics.json > logs/evaluate_lines.log 2>&1
$PY train_depth.py --epochs 1 --batch 4 --workers 2 --output runs/depth/initial > logs/train_depth_initial.log 2>&1
echo 'INITIAL_TRAINING_COMPLETED'

#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-.venv/bin/python}
mkdir -p logs runs/pose runs/segmentation runs/depth
echo YOLO_STARTED > logs/full_improved.status
"$PY" train_yolo.py --model models/yolo11n-pose.pt --data data/prepared/stairs_pose.yaml --epochs 200 --batch 16 --imgsz 512 --patience 30 --name improved_from_scratch --device 0 > logs/yolo_improved_from_scratch.log 2>&1
echo YOLO_FINISHED > logs/full_improved.status
"$PY" train_segmentation.py --epochs 30 --batch 16 --size 256 --output runs/segmentation/improved_from_scratch > logs/segmentation_improved_from_scratch.log 2>&1
echo SEGMENTATION_FINISHED >> logs/full_improved.status
"$PY" train_depth.py --epochs 200 --batch 4 --size 294 --output runs/depth/improved_from_scratch > logs/depth_improved_from_scratch.log 2>&1
echo DEPTH_FINISHED >> logs/full_improved.status
"$PY" evaluate_lines.py --weights runs/pose/improved_from_scratch/weights/best.pt --split val --output runs/pose/improved_from_scratch/line_metrics.json > logs/evaluate_improved_from_scratch.log 2>&1
echo EVALUATION_FINISHED >> logs/full_improved.status

#!/usr/bin/env bash
set -euo pipefail
cd /root/stairlab
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
trap 's=$?; if [ "$s" -ne 0 ]; then printf "FAILED exit=%s\n" "$s" > logs/sequence_200.status; fi' EXIT
printf 'YOLO_RUNNING\n' > logs/sequence_200.status
.venv/bin/python train_yolo.py --model runs/pose/baseline_30/weights/best.pt --epochs 200 --patience 20 --lr 0.0001 --workers 2 --name earlystop_200 > logs/yolo_200.log 2>&1
printf 'DEPTH_RUNNING\n' > logs/sequence_200.status
.venv/bin/python train_depth.py --init runs/depth/baseline_10/best.pt --epochs 200 --patience 15 --min-delta 0.0001 --lr 0.00003 --output runs/depth/earlystop_200 > logs/depth_200.log 2>&1
printf 'COMPLETED\n' > logs/sequence_200.status

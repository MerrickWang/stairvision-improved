# StairVision Improved

StairLab pipeline for stair-edge detection, tread/riser segmentation, relative-depth estimation, and structural post-processing.

## Pipeline

RGB image -> YOLO pose edge detection -> YOLO segmentation -> mask-boundary endpoint refinement -> structural filtering -> encoded-depth consistency checks.

The repository contains the current training and inference code, portable dataset configurations, dependency specifications, and evaluation documentation. Datasets, generated runs, and model checkpoints are intentionally kept outside Git because of their size.

## Main entry points

- `prepare_data.py` converts and validates the RGB-D stair dataset.
- `train_yolo.py` trains the two-class edge pose model.
- `train_segmentation.py` trains tread and riser segmentation.
- `train_depth.py` fine-tunes the Depth Anything V2 Small adapter.
- `scripts/train_sequence_200.sh` and `scripts/run_full_improved.sh` run the full training sequence.
- `predict.py` and `predict_full_yolo_seg.py` run batch inference.
- `refine_pose_with_seg.py` and `structure_postprocess.py` refine detections and apply structural checks.
- `evaluate_lines.py` and `check_depth_consistency.py` evaluate the outputs.

## Setup and training

Use Linux with Python 3.12, an NVIDIA GPU, and CUDA-compatible PyTorch and torchvision. OpenCV also needs the system packages `libgl1` and `libglib2.0-0` on Ubuntu. `environment.txt` records the RTX 4090 experiment environment. From the repository root:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest test_core -v
mkdir -p models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt -o models/yolo11n-pose.pt
huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf --local-dir models/depth-anything-v2-small
python download_data.py
python prepare_data.py --coordinates pixels
```

The downloader uses [RGB-D stair dataset V1](https://data.mendeley.com/datasets/6kffmjt7g2/1) (CC BY 4.0). The pose and segmentation YAML files in `configs/` show the expected dataset layout and labels. The preparation scripts write the actual YAML with paths for your checkout.

Run the initial, baseline, and longer experiments in order:

```bash
bash scripts/run_initial.sh
bash scripts/train_sequence_30_10.sh
bash scripts/train_sequence_200.sh
```

`bash scripts/run_full_improved.sh` runs the separate pose, semantic segmentation, and depth experiment using the prepared data and pretrained weights above. The shell scripts use `.venv/bin/python` by default. Set `PYTHON` to use another interpreter.

The refined inference pipeline uses a YOLO segmentation checkpoint. To train one, run:

```bash
python convert_seg_labels.py
yolo segment train model=yolo11n-seg.pt data=data/yolo_seg/data.yaml imgsz=640 project=runs/segment name=stair_yolo_seg
```

See `python predict_full_yolo_seg.py --help` for inference arguments. Supply a pose checkpoint, a YOLO segmentation checkpoint, and a depth checkpoint from the corresponding runs.

## Recorded validation

On 556 validation images, the `earlystop_200` pose run reached edge F1 0.5637 and visible convex-edge count MAE 1.2230. The `pose_clean_640` run used in the refined pipeline reached edge F1 0.4567 and count MAE 1.2986. The YOLO segmentation run recorded mask mAP50 0.9877 and mAP50–95 0.8783. The best recorded encoded-depth MAE was about 0.0713 on a separate depth run.

Depth is reported in normalized encoding units or relative inverse depth. Metric stair height requires camera intrinsics, depth scale, and pose calibration. The dataset source describes the underlying [StairNetV3 research](https://doi.org/10.1007/s00371-024-03268-8).

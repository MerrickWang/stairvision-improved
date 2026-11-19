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
- `train_sequence_200.sh` and `run_full_improved.sh` run the full training sequence.
- `predict.py` and `predict_full_yolo_seg.py` run batch inference.
- `refine_pose_with_seg.py` and `structure_postprocess.py` refine detections and apply structural checks.
- `evaluate_lines.py` and `check_depth_consistency.py` evaluate the outputs.

## Configuration and dependencies

Portable dataset configurations are in `configs/`. Install the packages listed in `requirements.txt`. `environment.txt` records the remote runtime used for the completed experiments.

The training scripts expect the RGB-D stair dataset to be prepared locally. Dataset metadata and download instructions are in `download_data.py` and `docs/STairLab_ORIGINAL_README.md`.

## Reproducing training

```bash
python -m unittest test_core -v
python prepare_data.py --coordinates pixels
python train_yolo.py --epochs 30 --batch 16 --imgsz 512 --name baseline_30
python train_segmentation.py --epochs 30
python train_depth.py --epochs 10 --batch 4 --size 294 --output runs/depth/baseline_10
```

Use `--resume` with a saved checkpoint to continue a run. The shell scripts in `scripts/` provide the longer training sequences used on the RTX 4090 host.

## Completed validation reference

The strongest recorded edge run is `runs/pose/earlystop_200`, with line Precision 0.5243, Recall 0.6094, F1 0.5637, endpoint error 0.0142 of the image diagonal, and convex-edge count MAE 1.2230 on 556 validation images. The recorded segmentation run reached mask mAP50 0.9877 and mAP50-95 0.8783. The best recorded depth run reached encoded-depth MAE about 0.0713.

Depth outputs are encoded or relative depth. They are not direct metre measurements. Metric stair height requires camera intrinsics, depth scale, and pose calibration.

## License and source

See `docs/STairLab_ORIGINAL_README.md` for the original StairNet citations, dataset source, and license information.

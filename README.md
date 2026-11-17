# StairVision Improved

Remote project: `/root/stairlab`.

## Pipeline

RGB image -> YOLO pose edge detection -> YOLO segmentation for tread and riser masks -> mask-boundary endpoint refinement -> structural filtering -> encoded depth consistency checks.

## Models

- YOLO pose 640 input: `runs/pose/pose_clean_640/weights/best.pt`
- YOLO segmentation: `runs/segment/stair_yolo_seg/weights/best.pt`
- Depth model: `runs/depth/earlystop_200/best.pt`

## Validation

- Pose F1: 0.4567
- Convex-edge count MAE: 1.2986
- Segmentation mask mAP50: 0.9877
- Segmentation mask mAP50-95: 0.8783

Depth output is encoded or relative depth and is not directly measured in metres. Metric stair height requires camera intrinsics, depth scale, and pose calibration.

## Remote entry points

- `train_yolo.py`
- `train_segmentation.py`
- `train_depth.py`
- `structure_postprocess.py`
- `refine_pose_with_seg.py`
- `predict_full_yolo_seg.py`
- `render_refined_demo.py`

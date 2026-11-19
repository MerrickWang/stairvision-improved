"""Two-endpoint, two-class YOLO11 pose baseline; no frontend."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/prepared/stairs_pose.yaml')
    p.add_argument('--model', default='models/yolo11n-pose.pt')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--name', default='yolo_baseline')
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--device', default='0')
    p.add_argument('--resume', help='Path to an unfinished last.pt run')
    a = p.parse_args()
    if a.resume:
        YOLO(a.resume).train(resume=True, device=a.device)
        return
    if not Path(a.model).exists():
        raise FileNotFoundError(f'Pretrained weights missing: {a.model}')
    YOLO(a.model).train(data=a.data, epochs=a.epochs, batch=a.batch,
        imgsz=a.imgsz, workers=a.workers, device=a.device,
        project='runs/pose', name=a.name, exist_ok=False,
        fraction=a.fraction, seed=42, deterministic=True,
        optimizer='AdamW', lr0=a.lr, patience=a.patience,
        mosaic=0, mixup=0, degrees=5, translate=0.05, scale=0.2,
        fliplr=0.5, flipud=0, amp=True, plots=True, save=True)


if __name__ == '__main__':
    main()

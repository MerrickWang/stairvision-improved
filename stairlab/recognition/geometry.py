import cv2
import numpy as np
import torch

from stairlab.recognition.yolo_fragments import SIZE


def zoom(image, box):
    x0, y0, x1, y1 = box
    cut = image[y0:y1, x0:x1]
    h, w = cut.shape[:2]
    side = max(h, w)
    px, py = (side - w) // 2, (side - h) // 2
    canvas = np.zeros((side, side, 3), dtype=cut.dtype)
    canvas[py:py + h, px:px + w] = cut
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), side / SIZE, [x0 - px, y0 - py]



def rgb_tensor(images, device):
    data = np.stack([im[..., ::-1] for im in images])
    return torch.from_numpy(data).permute(0, 3, 1, 2).to(device).float() / 255



def square(image):
    h, w = image.shape[:2]
    side = max(h, w)
    left, top = (side-w)//2, (side-h)//2
    padded = cv2.copyMakeBorder(image, top, side-h-top, left, side-w-left, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(padded, (SIZE, SIZE), interpolation=cv2.INTER_AREA), np.array([left, top]), SIZE/side


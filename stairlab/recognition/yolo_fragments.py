"""Dense convex-fragment head on the stride-8 YOLO11 pose features."""

import torch
from torch import nn
from torch.nn import functional as F

from stairlab.recognition.depthout_refine import decode_fragments


GRID = (64, 32)
SIZE = 512


class YoloFragments(nn.Module):
    def __init__(self, backbone, *, surface_aux=False):
        super().__init__()
        from ultralytics import YOLO

        model = YOLO(str(backbone)).model
        feature_index = model.model[-1].f[0]
        self.layers = nn.ModuleList(list(model.model[:feature_index + 1]))
        self.save = set(model.save)
        self.layers.eval()
        with torch.no_grad():
            channels = self.features(torch.zeros(1, 3, 128, 128)).shape[1]
        self.head = nn.Sequential(nn.Conv2d(channels, 96, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(96), nn.SiLU(), nn.Conv2d(96, 5, 1))
        with torch.no_grad():
            self.head[-1].bias[0] = -2.2
        self.surface_head = None
        if surface_aux:
            # Keep the baseline's RNG sequence for shuffling and augmentations.
            with torch.random.fork_rng(devices=[]):
                self.surface_head = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1, bias=False),
                                                  nn.BatchNorm2d(64), nn.SiLU(), nn.Conv2d(64, 3, 1))

    def features(self, image):
        saved = []
        for layer in self.layers:
            if layer.f != -1:
                image = (saved[layer.f] if isinstance(layer.f, int) else
                         [image if index == -1 else saved[index] for index in layer.f])
            image = layer(image)
            saved.append(image if layer.i in self.save else None)
        return image

    def forward(self, image, *, with_surfaces=False):
        features = self.features(image)
        # Each output cell covers 16 horizontal by 8 vertical input pixels.
        fragments = self.head(F.avg_pool2d(features, kernel_size=(1, 2), stride=(1, 2)))
        if with_surfaces:
            if self.surface_head is None:
                raise ValueError('This checkpoint has no surface head')
            return fragments, self.surface_head(features)
        return fragments


def decode_batch(raw, confidence):
    scores = raw[:, 0].float().sigmoid().cpu().numpy()
    positions = raw[:, 1:].float().sigmoid().cpu().numpy()
    return [decode_fragments(score, position, (SIZE, SIZE, SIZE, 0, 0), confidence)
            for score, position in zip(scores, positions)]

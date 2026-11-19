"""Fine-tune DA-V2 against the dataset's encoded depth, NOT metric distances."""
import torch
from torch import nn
from transformers import AutoModelForDepthEstimation


class EncodedDepthModel(nn.Module):
    output_kind = 'encoded_depth_0_1_not_meters'

    def __init__(self, source, freeze_backbone=True):
        super().__init__()
        self.base = AutoModelForDepthEstimation.from_pretrained(source, local_files_only=True)
        # Relative inverse-depth features are mapped to dataset grayscale encoding.
        # The adapter and pretrained decoder are trainable; no metric claim is made.
        self.adapter = nn.Conv2d(1, 1, 1)
        nn.init.constant_(self.adapter.weight, -0.02)
        nn.init.constant_(self.adapter.bias, 0.5)
        self.frozen = freeze_backbone
        if freeze_backbone:
            self.base.backbone.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.base.backbone.eval()
        return self

    def forward(self, pixels):
        features = self.base(pixel_values=pixels).predicted_depth
        return torch.sigmoid(self.adapter(features[:, None]))[:, 0]


def masked_loss(pred, target, mask):
    valid = mask.sum()
    if valid < 16:
        raise ValueError('Too few valid depth pixels in batch')
    l1 = (pred - target).abs()[mask].mean()
    terms = []
    for dim in (-1, -2):
        if dim == -1:
            m = mask[..., 1:] & mask[..., :-1]
            delta = (pred[..., 1:] - pred[..., :-1]) - (target[..., 1:] - target[..., :-1])
        else:
            m = mask[:, 1:, :] & mask[:, :-1, :]
            delta = (pred[:, 1:, :] - pred[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :])
        if m.any():
            terms.append(delta.abs()[m].mean())
    grad = torch.stack(terms).mean() if terms else l1 * 0
    return l1 + 0.1 * grad, l1

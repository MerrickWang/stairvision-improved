from torch import nn


class DenseStairHead(nn.Sequential):
    def __init__(self):
        layers = []
        for inch, dilation in ((3, 1), (12, 2), (12, 4), (12, 8), (12, 1)):
            layers.extend((nn.Conv2d(inch, 12, 3, padding=dilation, dilation=dilation), nn.ReLU()))
        super().__init__(*layers, nn.Conv2d(12, 1, 1))


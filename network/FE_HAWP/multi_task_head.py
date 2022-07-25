import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(self, inplanes, num_classes, head_size):
        super().__init__()
        assert num_classes == sum(head_size)
        planes = inplanes // 4

        heads = []
        for outplanes in head_size:
            heads.append(
                nn.Sequential(
                    nn.Conv2d(inplanes, planes, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(planes, outplanes, 1)
                )
            )
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)
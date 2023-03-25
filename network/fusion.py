import torch
import torch.nn as nn
from network.cmtn import CMTB


class ResNetBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes=None, stride=1, groups=1, downsample=None):
        super().__init__()
        planes = planes or inplanes // self.expansion

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 1, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, groups=groups)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        return out


class ChannelAttention(nn.Module):
    def __init__(self, inplanes, ratio=16):
        super().__init__()
        planes = inplanes // ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, inplanes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelSpatialAttention(nn.Module):
    def __init__(self, inplanes, ratio=16, kernel_size=7):
        super().__init__()

        self.ca = ChannelAttention(inplanes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ca(x) * self.sa(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, inplanes, ratio=16, kernel_size=7):
        super().__init__()

        self.ca = ChannelAttention(inplanes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class AddFusionModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        out = x1 + x2
        return out


class CAFusionModule(nn.Module):
    def __init__(self, inplanes, **kwargs):
        super().__init__()

        self.ca = nn.ModuleList([ChannelAttention(inplanes) for _ in range(2)])

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = x1 * self.ca[0](x1)
        x2 = x2 * self.ca[1](x2)
        out = x1 + x2
        return out


class SAFusionModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.sa = nn.ModuleList([SpatialAttention() for _ in range(2)])

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = x1 * self.sa[0](x1)
        x2 = x2 * self.sa[1](x2)
        out = x1 + x2
        return out


class CBAMFusionModule(nn.Module):
    def __init__(self, inplanes, **kwargs):
        super().__init__()

        self.cbam = nn.ModuleList([CBAM(inplanes) for _ in range(2)])

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.cbam[0](x1)
        x2 = self.cbam[1](x2)
        out = x1 + x2
        return out


class TransformerFusionModule(nn.Module):
    def __init__(self, inplanes, sr_ratio, size, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(inplanes * 2, inplanes, 1)
        self.block = CMTB(dim=inplanes, sr_ratio=sr_ratio, size=size, depth=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.block(out)
        return out


class IdentityModulationModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class CAModulationModule(nn.Module):
    def __init__(self, inplanes, **kwargs):
        super().__init__()

        self.ca = nn.ModuleList([ChannelAttention(inplanes) for _ in range(2)])
        self.res = nn.ModuleList([ResNetBottleneck(inplanes) for _ in range(2)])
        self.conv = nn.Conv2d(inplanes * 2, inplanes, 1)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = self.conv(x)
        fused_x1 = x1 + x2 * self.ca[1](x)
        fused_x2 = x2 + x1 * self.ca[0](x)

        x1 = x1 + self.res[0](fused_x1)
        x2 = x2 + self.res[1](fused_x2)
        return torch.cat([x1, x2], dim=1)


class SAModulationModule(nn.Module):
    def __init__(self, inplanes, **kwargs):
        super().__init__()

        self.sa = nn.ModuleList([SpatialAttention() for _ in range(2)])
        self.res = nn.ModuleList([ResNetBottleneck(inplanes) for _ in range(2)])
        self.conv = nn.Conv2d(inplanes * 2, inplanes, 1)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = self.conv(x)
        fused_x1 = x1 + x2 * self.sa[1](x)
        fused_x2 = x2 + x1 * self.sa[0](x)

        x1 = x1 + self.res[0](fused_x1)
        x2 = x2 + self.res[1](fused_x2)
        return torch.cat([x1, x2], dim=1)


class CASAModulationModule(nn.Module):
    def __init__(self, inplanes, **kwargs):
        super().__init__()

        self.casa = nn.ModuleList([ChannelSpatialAttention(inplanes) for _ in range(2)])
        self.res = nn.ModuleList([ResNetBottleneck(inplanes) for _ in range(2)])
        self.conv = nn.Conv2d(inplanes * 2, inplanes, 1)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = self.conv(x)
        fused_x1 = x1 + x2 * self.casa[1](x)
        fused_x2 = x2 + x1 * self.casa[0](x)

        x1 = x1 + self.res[0](fused_x1)
        x2 = x2 + self.res[1](fused_x2)
        return torch.cat([x1, x2], dim=1)
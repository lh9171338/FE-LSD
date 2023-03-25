import torch
import torch.nn as nn
import torch.nn.functional as F
from network.fusion import ResNetBottleneck, CAModulationModule, TransformerFusionModule


class Hourglass(nn.Module):
    def __init__(self, res_block, fusion_block, num_blocks, inplanes, depth):
        super().__init__()
        self.depth = depth

        self.encoders = self._make_encoder(res_block, num_blocks, inplanes * 2, depth)
        self.decoders = self._make_decoder(res_block, num_blocks, inplanes, depth)
        self.fusions = self._make_fusion(fusion_block, inplanes, depth)

    def _make_residual(self, block, num_blocks, inplanes, groups=1):
        layers = [block(inplanes, groups=groups) for _ in range(num_blocks)]

        return nn.Sequential(*layers)

    def _make_encoder(self, block, num_blocks, inplanes, depth):
        encoders = [self._make_residual(block, num_blocks, inplanes, groups=2) for _ in range(depth + 1)]

        return nn.ModuleList(encoders)

    def _make_decoder(self, block, num_blocks, inplanes, depth):
        decoders = [self._make_residual(block, num_blocks, inplanes) for _ in range(depth)]

        return nn.ModuleList(decoders)

    def _make_fusion(self, block, inplanes, depth):
        fusions = []
        for i in range(depth + 1):
            sr_ratio = 2 ** (4 - i)
            width = 2 ** (7 - i)
            height = 2 ** (7 - i)
            fusions.append(block(inplanes=inplanes, sr_ratio=sr_ratio, size=(width, height)))

        return nn.ModuleList(fusions)

    def _encoder_forward(self, x):
        out = []
        for i in range(self.depth):
            out.append(self.fusions[i](x))
            x = self.encoders[i](F.max_pool2d(x, 2, stride=2))

            if i == self.depth - 1:
                x = self.encoders[i + 1](x)
                out.append(self.fusions[i + 1](x))

        return out[::-1]

    def _decoder_forward(self, x):
        out = x[0]
        for i in range(self.depth):
            up = x[i + 1]
            low = self.decoders[i](out)
            low = F.interpolate(low, scale_factor=2)
            out = low + up

        return out

    def forward(self, x):
        x = self._encoder_forward(x)
        out = self._decoder_forward(x)

        return out


class HourglassNet(nn.Module):
    def __init__(self, res_block, modulation_block, fusion_block, inplanes, num_feats, depth, num_stacks,
                 num_blocks):
        super().__init__()
        self.num_stacks = num_stacks

        # Shallow feature extraction and modulations
        self.shallow_conv1, self.shallow_conv2, self.shallow_res = \
            self._make_shallow_layer(res_block, inplanes, num_feats)
        self.modulations = nn.ModuleList([
            modulation_block(inplanes=inplanes, sr_ratio=32, size=(256, 256)),
            modulation_block(inplanes=num_feats, sr_ratio=16, size=(128, 128)),
        ])

        # Hourglass modules
        self.hgs = nn.ModuleList(
            [Hourglass(res_block, fusion_block, num_blocks, num_feats, depth) for _ in range(num_stacks)])
        self.res = nn.ModuleList([self._make_residual(res_block, num_blocks, num_feats) for _ in range(num_stacks)])
        self.fcs = nn.ModuleList([self._make_fc(num_feats, num_feats) for _ in range(num_stacks)])
        self.fcs_ = nn.ModuleList([nn.Conv2d(num_feats, num_feats, 1) for _ in range(num_stacks - 1)])

    def _make_residual(self, block, num_blocks, inplanes, planes=None, stride=1, groups=1):
        planes = planes or inplanes // block.expansion
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, groups=groups)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, groups=groups))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes, groups=groups))

        return nn.Sequential(*layers)

    def _make_shallow_layer(self, block, inplanes, num_feats):
        shallow_conv1 = nn.Sequential(
            nn.Conv2d(3, inplanes, 7, stride=2, padding=3),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )

        shallow_conv2 = nn.Sequential(
            nn.Conv2d(10, inplanes, 7, stride=2, padding=3),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )

        shallow_res = nn.Sequential(
            self._make_residual(block, 1, inplanes * 2, inplanes * 2, groups=2),
            nn.MaxPool2d(2, stride=2),
            self._make_residual(block, 1, inplanes * block.expansion * 2, inplanes * block.expansion * 2, groups=2),
            self._make_residual(block, 1, inplanes * block.expansion ** 2 * 2, num_feats // block.expansion * 2,
                                groups=2)
        )

        return shallow_conv1, shallow_conv2, shallow_res

    def _make_fc(self, inplanes, outplanes):
        fc = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

        return fc

    def forward(self, x):
        x1, x2 = x[:, :3], x[:, 3:]
        x = torch.cat([self.shallow_conv1(x1), self.shallow_conv2(x2)], dim=1)
        x = self.modulations[0](x)
        x = self.shallow_res(x)
        x = self.modulations[1](x)

        y = None
        for i in range(self.num_stacks):
            y = self.hgs[i](x)
            y = self.fcs[i](self.res[i](y))

            if i < self.num_stacks - 1:
                y = self.fcs_[i](y)
                x = x + y.repeat(1, 2, 1, 1)

        return y


def build_hg(cfg):
    inplanes = cfg.inplanes
    num_feats = cfg.num_feats
    depth = cfg.depth
    num_stacks = cfg.num_stacks
    num_blocks = cfg.num_blocks

    model = HourglassNet(
        res_block=ResNetBottleneck,
        modulation_block=CAModulationModule,
        fusion_block=TransformerFusionModule,
        inplanes=inplanes,
        num_feats=num_feats,
        depth=depth,
        num_stacks=num_stacks,
        num_blocks=num_blocks
    )

    return model
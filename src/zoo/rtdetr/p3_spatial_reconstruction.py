import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["P3SpatialReconstruction"]


class P3SpatialReconstruction(nn.Module):

    def __init__(self, channels: int = 256, upscale: int = 2):
        super().__init__()
        self.upscale = upscale

        self.expand_conv = nn.Sequential(
            nn.Conv2d(channels, channels * upscale * upscale, 1, bias=False),
            nn.BatchNorm2d(channels * upscale * upscale),
            nn.SiLU(inplace=True),
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = F.interpolate(
            x, scale_factor=self.upscale, mode="bilinear", align_corners=False
        )
        x = self.expand_conv(x)
        x = self.pixel_shuffle(x)
        x = self.refine(x)
        return x + identity

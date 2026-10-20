import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["P3SpatialReconstruction"]


class P3SpatialReconstruction(nn.Module):
    """PixelShuffle spatial reconstruction for the finest encoder feature (P3)."""

    VALID_VARIANTS = {
        "direct",
        "bilinear",
        "bilinear_refine",
        "pixelshuffle",
        "pixelshuffle_refine",
        "full",
        "transposed_refine_residual",
    }

    def __init__(self, channels: int = 256, upscale: int = 2, variant: str = "full"):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unsupported P3SR variant: {variant}. "
                f"Expected one of {sorted(self.VALID_VARIANTS)}."
            )
        self.upscale = upscale
        self.variant = variant

        use_pixelshuffle = variant in {"pixelshuffle", "pixelshuffle_refine", "full"}
        use_transposed = variant == "transposed_refine_residual"
        use_refine = variant in {
            "bilinear_refine",
            "pixelshuffle_refine",
            "full",
            "transposed_refine_residual",
        }

        if use_pixelshuffle:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channels, channels * upscale * upscale, 1, bias=False),
                nn.BatchNorm2d(channels * upscale * upscale),
                nn.SiLU(inplace=True),
            )
            self.pixel_shuffle = nn.PixelShuffle(upscale)
        else:
            self.expand_conv = None
            self.pixel_shuffle = None

        if use_transposed:
            self.transposed_conv = nn.Sequential(
                nn.ConvTranspose2d(
                    channels,
                    channels,
                    kernel_size=upscale,
                    stride=upscale,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.transposed_conv = None

        if use_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.refine = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.variant == "direct":
            return x

        if self.variant == "bilinear":
            return F.interpolate(
                x, scale_factor=self.upscale, mode="bilinear", align_corners=False
            )

        if self.variant == "bilinear_refine":
            out = F.interpolate(
                x,
                scale_factor=self.upscale,
                mode="bilinear",
                align_corners=False,
            )
            return self.refine(out)

        if self.variant == "transposed_refine_residual":
            out = self.transposed_conv(x)
            out = self.refine(out)
            identity = F.interpolate(
                x,
                scale_factor=self.upscale,
                mode="bilinear",
                align_corners=False,
            )
            return out + identity

        out = self.expand_conv(x)
        out = self.pixel_shuffle(out)

        if self.refine is not None:
            out = self.refine(out)

        if self.variant == "full":
            identity = F.interpolate(
                x, scale_factor=self.upscale, mode="bilinear", align_corners=False
            )
            out = out + identity

        return out

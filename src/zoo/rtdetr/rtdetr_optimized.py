from ...core import register
from .rtdetr import RTDETR
from .p3_spatial_reconstruction import P3SpatialReconstruction

__all__ = ["RTDETROptimized"]


@register()
class RTDETROptimized(RTDETR):

    def __init__(
        self, backbone, encoder, decoder, p3_sr_channels=256, p3_sr_upscale=2, **kwargs
    ):
        super().__init__(backbone, encoder, decoder, **kwargs)
        self.p3_sr = P3SpatialReconstruction(
            channels=p3_sr_channels, upscale=p3_sr_upscale
        )

    def forward(self, x, targets=None):
        x = self.backbone(x)

        if isinstance(x, dict):
            stage0_feat = x.get("stage0")
            feats = x.get("feats", [])
            try:
                x = self.encoder(feats, stage0_feat=stage0_feat)
            except TypeError:
                x = self.encoder(feats)
        else:
            x = self.encoder(x)

        x[0] = self.p3_sr(x[0])

        x = self.decoder(x, targets)
        return x

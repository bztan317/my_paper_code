import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from typing import List

from ...core import register
from ...nn.backbone.ega_neck import EGA_Neck

__all__ = [
    "RTDETR",
]


@register()
class RTDETR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        use_ega: bool = False,
        stage1_channels: int = 64,
        ega_use_spatial_mask: bool = True,
        ega_use_gate: bool = True,
        ega_use_aux_branch: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

        if use_ega:
            self.encoder.ega_neck = EGA_Neck(
                stage1_channels=stage1_channels,
                feat_channels=[encoder.hidden_dim] * len(encoder.in_channels),
                out_channels=encoder.hidden_dim,
                use_spatial_mask=ega_use_spatial_mask,
                use_gate=ega_use_gate,
                use_aux_branch=ega_use_aux_branch,
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
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

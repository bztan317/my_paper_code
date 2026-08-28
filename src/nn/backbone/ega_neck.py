import torch
import torch.nn as nn

from .early_feature_gated_augmentation import (
    EarlyFeatureGatedAugmentation,
    ResidualAugmentation,
)

class EGA_Neck(nn.Module):
    """Checkpoint-compatible implementation of the paper's SFGA module.

    Stage 1 (EarlyFeatureGatedAugmentation):
        Learnable edge detection + gated additive augmentation.
        P3: full-conv gate (small objects). P4/P5: DW gate.

    Stage 2 (ResidualAugmentation):
        Residual addition of edge-enhanced features onto backbone features.
    """

    def __init__(
        self,
        stage1_channels: int = 64,
        feat_channels: list = [256, 256, 256],
        out_channels: int = 256,
        ega_reduction: int = 4,
        use_spatial_mask: bool = True,
        use_gate: bool = True,
        use_aux_branch: bool = True,
    ):
        super().__init__()
        # ega_reduction kept for backward compatibility with existing configs
        num_levels = len(feat_channels)
        self.num_levels = num_levels

        self.eie = EarlyFeatureGatedAugmentation(
            in_channels=stage1_channels,
            out_channels=out_channels,
            use_spatial_mask=use_spatial_mask,
            use_gate=use_gate,
            use_aux_branch=use_aux_branch,
            num_levels=num_levels,
        )

        self.fusions = nn.ModuleList([
            ResidualAugmentation(out_channels, feat_channels[i], out_channels)
            for i in range(num_levels)
        ])

    def forward(
        self,
        stage0_feat: torch.Tensor,
        *feats,
    ):
        edge_feats = self.eie(stage0_feat)

        out_feats = []
        for i in range(self.num_levels):
            out_feats.append(self.fusions[i](edge_feats[i], feats[i]))

        return tuple(out_feats)


# Public paper terminology without changing checkpoint parameter keys.
SFGA = EGA_Neck


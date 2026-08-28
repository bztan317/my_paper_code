import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = None, groups: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ===================================================================
# Learnable Edge Detection (Residual Block, 64ch full conv)
# ===================================================================

class EdgeFeatureExtractor(nn.Module):
    """Checkpoint-compatible implementation of the paper's SFE block.

    Two full-conv layers + residual connection on 64ch input.
    Total params ~74K - small cost for the most critical component.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, k=3, s=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return self.act(out)


# ===================================================================
# Edge Gated Fusion (full-conv for P3, DW for P4/P5)
# ===================================================================

class GatedAugmentationLayer(nn.Module):
    """Gated additive edge augmentation.

    gate operates on edge_ch (64ch), so even full-conv is lightweight (~37K).
    use_full_gate=True  -> full conv gate (precision, for P3 small objects)
    use_full_gate=False -> DW gate (efficiency, for P4/P5)
    """

    def __init__(self, edge_ch: int, feat_ch: int, use_full_gate: bool = True):
        super().__init__()
        gate_groups = 1 if use_full_gate else edge_ch
        self.gate = nn.Sequential(
            nn.Conv2d(edge_ch, edge_ch, kernel_size=3, padding=1, groups=gate_groups, bias=False),
            nn.BatchNorm2d(edge_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(edge_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.edge_proj = nn.Sequential(
            nn.Conv2d(edge_ch, feat_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_ch),
        )

    def forward(
        self,
        edge: torch.Tensor,
        feat: torch.Tensor,
        use_gate: bool = True,
        use_aux: bool = True,
    ) -> torch.Tensor:
        projected = self.edge_proj(edge)
        if use_gate:
            projected = self.gate(edge) * projected
        if use_aux:
            return feat + projected
        return projected


# ===================================================================
# Multi-Scale Early-Feature Gated Augmentation (EGA)
# ===================================================================

class EarlyFeatureGatedAugmentation(nn.Module):
    """Multi-scale Early-Feature Gated Augmentation with tiered computation.

    Feature line: full-conv feat_80 + DW+PW feat_40/20 cascaded.
    Edge line: EdgeFeatureExtractor (residual block) + MaxPool cascaded.
    Tiered gating: full-conv gate for P3 (small objects), DW gate for P4/P5.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 256,
        use_spatial_mask: bool = True,
        use_gate: bool = True,
        use_aux_branch: bool = True,
        num_levels: int = 3,
    ):
        super().__init__()
        self.use_spatial_mask = use_spatial_mask
        self.use_gate = use_gate
        self.use_aux_branch = use_aux_branch
        self.num_levels = num_levels

        self.edge_detect = EdgeFeatureExtractor(in_channels)

        if num_levels == 3:
            # Feature line
            self.feat_80 = nn.Sequential(
                ConvBNAct(in_channels, out_channels, k=3, s=2),
                ConvBNAct(out_channels, out_channels, k=3, s=1),
            )
            self.feat_40 = nn.Sequential(
                ConvBNAct(out_channels, out_channels, k=3, s=2, groups=out_channels),
                ConvBNAct(out_channels, out_channels, k=1, s=1),
            )
            self.feat_20 = nn.Sequential(
                ConvBNAct(out_channels, out_channels, k=3, s=2, groups=out_channels),
                ConvBNAct(out_channels, out_channels, k=1, s=1),
            )

            # Gated fusion (tiered: strong for P3, light for P4/P5)
            if use_spatial_mask:
                self.gate_80 = GatedAugmentationLayer(in_channels, out_channels, use_full_gate=True)
                self.gate_40 = GatedAugmentationLayer(in_channels, out_channels, use_full_gate=False)
                self.gate_20 = GatedAugmentationLayer(in_channels, out_channels, use_full_gate=False)
        elif num_levels == 4:
            # Feature line for 4 levels (160, 80, 40, 20)
            self.feat_160 = ConvBNAct(in_channels, out_channels, k=3, s=1)
            self.feat_80 = nn.Sequential(
                ConvBNAct(out_channels, out_channels, k=3, s=2),
                ConvBNAct(out_channels, out_channels, k=3, s=1),
            )
            self.feat_40 = nn.Sequential(
                ConvBNAct(out_channels, out_channels, k=3, s=2, groups=out_channels),
                ConvBNAct(out_channels, out_channels, k=1, s=1),
            )
            self.feat_20 = nn.Sequential(
                ConvBNAct(out_channels, out_channels, k=3, s=2, groups=out_channels),
                ConvBNAct(out_channels, out_channels, k=1, s=1),
            )

            if use_spatial_mask:
                self.gate_160 = GatedAugmentationLayer(in_channels, out_channels, use_full_gate=True)
                self.gate_80 = GatedAugmentationLayer(in_channels, out_channels, use_full_gate=False)
                self.gate_40 = GatedAugmentationLayer(in_channels, out_channels, use_full_gate=False)
                self.gate_20 = GatedAugmentationLayer(in_channels, out_channels, use_full_gate=False)
        else:
            raise ValueError(f"Unsupported num_levels: {num_levels}")

    def forward(self, x: torch.Tensor):
        if self.num_levels == 3:
            f80 = self.feat_80(x)
            f40 = self.feat_40(f80)
            f20 = self.feat_20(f40)

            if self.use_spatial_mask:
                edge = self.edge_detect(x)                          # [B, 64, 160, 160]
                e80  = F.max_pool2d(edge, kernel_size=2, stride=2)  # [B, 64, 80, 80]
                e40  = F.max_pool2d(e80, kernel_size=2, stride=2)   # [B, 64, 40, 40]
                e20  = F.max_pool2d(e40, kernel_size=2, stride=2)   # [B, 64, 20, 20]

                fuse_kwargs = {
                    "use_gate": self.use_gate,
                    "use_aux": self.use_aux_branch,
                }
                out_80 = self.gate_80(e80, f80, **fuse_kwargs)
                out_40 = self.gate_40(e40, f40, **fuse_kwargs)
                out_20 = self.gate_20(e20, f20, **fuse_kwargs)
            else:
                out_80, out_40, out_20 = f80, f40, f20

            return out_80, out_40, out_20

        elif self.num_levels == 4:
            f160 = self.feat_160(x)
            f80 = self.feat_80(f160)
            f40 = self.feat_40(f80)
            f20 = self.feat_20(f40)

            if self.use_spatial_mask:
                edge = self.edge_detect(x)                          # [B, 64, 160, 160]
                e160 = edge                                         # [B, 64, 160, 160]
                e80  = F.max_pool2d(e160, kernel_size=2, stride=2)  # [B, 64, 80, 80]
                e40  = F.max_pool2d(e80, kernel_size=2, stride=2)   # [B, 64, 40, 40]
                e20  = F.max_pool2d(e40, kernel_size=2, stride=2)   # [B, 64, 20, 20]

                fuse_kwargs = {
                    "use_gate": self.use_gate,
                    "use_aux": self.use_aux_branch,
                }
                out_160 = self.gate_160(e160, f160, **fuse_kwargs)
                out_80 = self.gate_80(e80, f80, **fuse_kwargs)
                out_40 = self.gate_40(e40, f40, **fuse_kwargs)
                out_20 = self.gate_20(e20, f20, **fuse_kwargs)
            else:
                out_160, out_80, out_40, out_20 = f160, f80, f40, f20

            return out_160, out_80, out_40, out_20


# ===================================================================
# Residual Augmentation
# ===================================================================

class ResidualAugmentation(nn.Module):
    """Residual additive feature augmentation.

    deep_feat passes through unchanged, edge_feat aligned via 1x1 and added.
    BN initialization naturally protects pretrained weights.
    """

    def __init__(self, edge_ch: int, feat_ch: int, out_ch: int):
        super().__init__()
        self.edge_align = nn.Sequential(
            nn.Conv2d(edge_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, edge_feat: torch.Tensor, deep_feat: torch.Tensor) -> torch.Tensor:
        if edge_feat.shape[-2:] != deep_feat.shape[-2:]:
            edge_feat = F.interpolate(
                edge_feat, size=deep_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        return deep_feat + self.edge_align(edge_feat)


# Public paper terminology without changing checkpoint parameter keys.
SFE = EdgeFeatureExtractor


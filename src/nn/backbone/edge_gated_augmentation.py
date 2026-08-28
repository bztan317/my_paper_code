"""Backward-compatible imports for the former EGA module name.

EGA now stands for Early-Feature Gated Augmentation. New code should import
from ``early_feature_gated_augmentation`` instead.
"""

from .early_feature_gated_augmentation import (
    ConvBNAct,
    EarlyFeatureGatedAugmentation,
    EdgeFeatureExtractor,
    GatedAugmentationLayer,
    ResidualAugmentation,
)

# Preserve external imports that used the former class name.
EdgeGatedAugmentation = EarlyFeatureGatedAugmentation

__all__ = [
    "ConvBNAct",
    "EarlyFeatureGatedAugmentation",
    "EdgeGatedAugmentation",
    "EdgeFeatureExtractor",
    "GatedAugmentationLayer",
    "ResidualAugmentation",
]



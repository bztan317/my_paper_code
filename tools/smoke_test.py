import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from src.core import YAMLConfig
from src.nn.backbone.ega_neck import EGA_Neck, SFGA
from src.nn.backbone.early_feature_gated_augmentation import (
    EdgeFeatureExtractor,
    GatedAugmentationLayer,
    ResidualAugmentation,
    SFE,
)
from src.solver._solver import remap_key


def test_imports():
    assert EGA_Neck and SFGA and EdgeFeatureExtractor and SFE
    assert GatedAugmentationLayer and ResidualAugmentation


def test_remap():
    old = "encoder.edge_fusion_neck.fusion_p3.edge_align.0.weight"
    new = remap_key(old)
    assert new == "encoder.ega_neck.fusions.0.edge_align.0.weight"


def test_forward():
    cfg = YAMLConfig(
        "configs/rtdetrv2/visdrone_p3sr_ega_sgw.yml",
        device="cpu",
        eval_spatial_size=[128, 128],
        PResNet={"pretrained": False},
    )
    model = cfg.model
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out is not None


def test_checkpoint_load():
    ckpt_path = os.environ.get("REFINE_DETR_CHECKPOINT")
    if not ckpt_path:
        return
    ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"REFINE_DETR_CHECKPOINT does not point to a file: {ckpt_path}"
        )
    if not ckpt_path.lower().endswith((".pth", ".pt")):
        raise ValueError("REFINE_DETR_CHECKPOINT must point to a .pth or .pt file")
    cfg = YAMLConfig(
        "configs/rtdetrv2/visdrone_p3sr_ega_sgw.yml",
        device="cpu",
        resume=ckpt_path,
        test_only=True,
        PResNet={"pretrained": False},
    )
    from src.solver import TASKS

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    solver.load_resume_state(ckpt_path)
    model = solver.model
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    assert out is not None


if __name__ == "__main__":
    test_imports()
    test_remap()
    test_forward()
    test_checkpoint_load()
    print("smoke_test ok")

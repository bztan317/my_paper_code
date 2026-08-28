# ReFine-DETR

Official PyTorch implementation accompanying **ReFine-DETR: Representation Refinement for Small Object Detection in UAV Imagery**.

ReFine-DETR builds on RT-DETRv2 and introduces three components:

- **SFGA** — Shallow-Feature Gated Augmentation before encoder feature interaction.
- **P3SR** — feature-level spatial reconstruction of the finest fused feature.
- **SGW** — scale-gated Wasserstein regression loss.

> The public paper uses the names **SFGA** and **SFE**. For checkpoint and
> configuration compatibility, some code identifiers retain their experimental
> names, including `use_ega`, `EGA_Neck`, and `EdgeFeatureExtractor`. Public
> aliases `SFGA` and `SFE` are also provided.

## Installation

The paper's reference environment is Python 3.11.10, CUDA 12.1, and PyTorch
2.4.0 on one RTX 4090 (24 GB). `requirements.txt` pairs this PyTorch release
with torchvision 0.19.0. Matching the reference environment is recommended
when reproducing the reported results.

```bash
pip install -r requirements.txt
```

The default configurations use PyTorch and CUDA when available. CPU construction and smoke tests are also supported.

## Repository layout

```text
configs/                 Dataset and experiment configurations
src/                     Model, data, solver, and loss implementations
tools/train.py           Training and evaluation entry point
tools/smoke_test.py      Offline construction and forward-pass checks
tools/deploy/            PyTorch inference example
RESULTS.md               Main reported results
```

Key implementations:

- SFGA: `src/nn/backbone/ega_neck.py`
- SFE: `src/nn/backbone/early_feature_gated_augmentation.py`
- P3SR: `src/zoo/rtdetr/p3_spatial_reconstruction.py`
- SGW: `src/zoo/rtdetr/scale_gated_wasserstein_loss.py`
- CIoU/EIoU: `src/zoo/rtdetr/iou_losses.py`

## Datasets

Dataset files are not distributed in this repository. The provided YAML files expect a sibling `dataset` directory by default:

```text
workspace/
├── dataset/
│   ├── VisDrone2019/
│   ├── UAV-benchmark-M/
│   ├── AITOD/
│   └── DIOR/
└── ReFine-DETR/
```

If your datasets are stored elsewhere, edit the paths in `configs/dataset/*.yml`.

## Training

Baseline on VisDrone2019:

```bash
python tools/train.py -c configs/rtdetrv2/visdrone.yml --use-amp
```

Complete ReFine-DETR model:

```bash
python tools/train.py -c configs/rtdetrv2/visdrone_p3sr_ega_sgw.yml --use-amp
```

Other paper datasets use the corresponding `aitod`, `uavdt`, and `dior`
configurations in `configs/rtdetrv2/`. CoDrone YAML files are supplementary
configurations and are not used by the manuscript's reported tables.

## Evaluation

```bash
python tools/train.py \
  -c configs/rtdetrv2/visdrone_p3sr_ega_sgw.yml \
  -r path/to/best.pth \
  --test-only
```

## Smoke test

The default smoke test is offline and does not download pretrained weights:

```bash
python tools/smoke_test.py
```

To additionally test a checkpoint, set `REFINE_DETR_CHECKPOINT`:

```bash
REFINE_DETR_CHECKPOINT=path/to/best.pth python tools/smoke_test.py
```

On Windows PowerShell:

```powershell
$env:REFINE_DETR_CHECKPOINT = "path\to\best.pth"
python tools/smoke_test.py
```

## Results and checkpoints

Reported benchmark, module-ablation, loss-ablation, and hyperparameter results
are reproduced at manuscript precision in [RESULTS.md](RESULTS.md). In
particular, DIOR `mAP` is the mean of its 20 per-category AP50 values rather
than COCO AP@[0.50:0.95].

Training logs and model checkpoints are not committed to Git. Publish large checkpoints through GitHub Releases or another model-hosting service and link them here.

## Acknowledgements and license

This codebase is derived from [RT-DETR and RT-DETRv2](https://github.com/lyuwenyu/RT-DETR). The upstream project is distributed under the Apache License 2.0. ReFine-DETR modifications are distributed under the same license; see [LICENSE](LICENSE).

If this repository is useful in your research, please cite the ReFine-DETR paper and the original RT-DETR/RT-DETRv2 work.

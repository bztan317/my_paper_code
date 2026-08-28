# Results

This file reproduces the values reported in the ReFine-DETR manuscript at the
same precision used in the paper. All accuracy metrics are percentages.

> **Metric convention.** VisDrone2019, UAVDT, and AI-TOD use the COCO-style
> metrics shown below. DIOR instead reports per-category AP50 and their mean;
> its `mAP` column must not be interpreted as COCO AP@[0.50:0.95].

## Reference experimental setup

| Setting | Paper value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (24 GB), single GPU |
| Python | 3.11.10 |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |
| Baseline | RT-DETRv2-R18 with ImageNet-pretrained ResNet-18 |
| Input size | 640 x 640 |
| Training | 120 epochs, AdamW |
| Learning rates | detector: 1e-4; backbone: 1e-5 |
| AdamW parameters | betas: (0.9, 0.999); weight decay: 1e-4 |
| Augmentation schedule | photometric distortion, zoom-out, and IoU crop disabled after epoch 117; horizontal flip retained |
| Batch sizes | VisDrone2019: 8; UAVDT: 16; AI-TOD: 4; DIOR: 8 |
| P3SR | upscale factor `r = 2` |
| SGW | `tau = 1.5`, `gamma = 100` |

## Main results

### VisDrone2019

| Model | AP | AP50 | AP75 | APS | APM | APL | Params (M) | GFLOPs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETRv2-R18 | 27.8 | 47.1 | 27.5 | 19.1 | 38.2 | 49.3 | 20.09 | 61.2 | 96.4 |
| ReFine-DETR | **29.8** | **50.2** | **29.7** | 20.9 | **40.7** | **53.5** | 21.67 | 101.1 | 77.7 |

### UAVDT

| Model | AP | AP50 | AP75 | APS | APM |
|---|---:|---:|---:|---:|---:|
| RT-DETRv2-R18 | 21.6 | 36.1 | 22.9 | 15.5 | 31.8 |
| ReFine-DETR | **22.2** | **37.2** | **23.7** | 15.5 | 33.7 |

### AI-TOD

| Model | AP | AP50 | AP75 | APvt | APt | APs |
|---|---:|---:|---:|---:|---:|---:|
| RT-DETRv2-R18 | 21.3 | 48.6 | 15.6 | 8.0 | 20.4 | 28.7 |
| ReFine-DETR | 22.8 | **52.3** | **16.3** | 8.6 | **21.9** | 29.8 |

### DIOR

The official category order is: C1 airplane, C2 airport, C3 baseball field,
C4 basketball court, C5 bridge, C6 chimney, C7 dam, C8 expressway service
area, C9 expressway toll station, C10 golf field, C11 ground track field,
C12 harbor, C13 overpass, C14 ship, C15 stadium, C16 storage tank, C17 tennis
court, C18 train station, C19 vehicle, and C20 windmill.

| Model | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETRv2-R18 | 92.24 | 88.56 | 89.81 | 90.62 | 57.17 | 80.89 | 73.73 | 92.04 | 81.07 | 82.80 |
| ReFine-DETR | 93.09 | 89.90 | 90.27 | **90.96** | **59.32** | 82.35 | 76.07 | **93.26** | **85.44** | 82.67 |

| Model | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 | mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETRv2-R18 | 87.69 | 64.24 | 67.70 | 77.77 | **86.84** | 76.57 | 93.20 | 73.40 | 59.11 | 92.23 | 80.38 |
| ReFine-DETR | **88.37** | **67.65** | **68.87** | 78.61 | 85.52 | 79.03 | **93.51** | 71.60 | 62.56 | **92.60** | **81.60** |

## Module-wise ablation on VisDrone2019

`AP50S` and `AP50M` denote AP50 for small and medium objects.

| SFGA | P3SR | SGW | Params (M) | GFLOPs | AP | AP50 | AP50S | AP50M |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| — | — | — | 20.09 | 61.2 | 27.8 | 47.1 | 38.2 | 59.3 |
| yes | — | — | 21.33 | 76.5 | 28.6 | 48.3 | 39.7 | 60.4 |
| — | yes | — | 20.43 | 85.8 | 28.5 | 47.8 | 38.7 | 60.1 |
| — | — | yes | 20.09 | 61.2 | 27.8 | 47.8 | 38.4 | 60.9 |
| yes | yes | — | 21.67 | 101.1 | 29.4 | 49.2 | 40.4 | 61.2 |
| yes | — | yes | 21.33 | 76.5 | 28.4 | 48.7 | 40.0 | 61.7 |
| — | yes | yes | 20.43 | 85.8 | 28.8 | 48.9 | 39.8 | 61.6 |
| yes | yes | yes | 21.67 | 101.1 | **29.8** | **50.2** | **41.3** | **62.7** |

## SFGA component ablation on VisDrone2019

P3SR and SGW are disabled in this comparison. `U` is the scale-specific
branch and `E` is the shallow-feature encoded branch.

| U | E | Gate | Params (M) | GFLOPs | AP | AP50 | AP50S | AP50M |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| yes | yes | — | 21.33 | 76.1 | **28.57** | 48.27 | 39.68 | 60.20 |
| yes | — | — | 21.24 | 72.0 | 28.31 | 47.84 | 38.91 | 60.28 |
| — | yes | yes | 21.33 | 76.5 | 28.24 | 47.95 | 38.94 | **60.36** |
| yes | yes | yes | 21.33 | 76.5 | **28.57** | **48.31** | **39.69** | **60.36** |

## Regression-loss comparison on VisDrone2019

All methods use the same RT-DETRv2-R18 architecture, L1 term, loss weights,
and GIoU-based Hungarian matching cost. Only the box regression loss changes.

| Loss | AP | AP50 | AP75 | AP50S | AP50M |
|---|---:|---:|---:|---:|---:|
| GIoU (baseline) | 27.80 | 47.06 | 27.53 | 38.22 | 59.29 |
| CIoU | 27.73 | 47.25 | 27.39 | 37.96 | 59.78 |
| EIoU | 27.49 | 46.82 | 27.16 | 37.61 | 59.37 |
| NWD | 27.78 | 47.25 | **27.65** | 37.77 | 60.81 |
| SGW | **27.80** | **47.76** | 27.05 | **38.36** | **60.89** |

## SGW hyperparameter ablation on VisDrone2019

SFGA and P3SR are disabled in this comparison.

| tau | gamma | AP | AP50 | AP50S | AP50M |
|---:|---:|---:|---:|---:|---:|
| GIoU baseline | — | 27.8 | 47.1 | 38.2 | 59.3 |
| 1.5 | 100 | **27.8** | **47.8** | **38.4** | 60.9 |
| 1.5 | 50 | 27.7 | 47.6 | 38.1 | 60.9 |
| 1.5 | 150 | 27.8 | 47.6 | 37.9 | **61.2** |
| 1.0 | 100 | 27.8 | 47.7 | 38.2 | 61.1 |
| 2.0 | 100 | 27.7 | 47.4 | 37.6 | 61.1 |

## Configuration mapping

| Experiment | Configuration |
|---|---|
| Baseline | `configs/rtdetrv2/visdrone.yml` |
| SFGA | `configs/rtdetrv2/visdrone_ega.yml` |
| P3SR | `configs/rtdetrv2/visdrone_p3sr.yml` |
| SGW | `configs/rtdetrv2/visdrone_sgw.yml` |
| SFGA + P3SR | `configs/rtdetrv2/visdrone_p3sr_ega.yml` |
| SFGA + SGW | `configs/rtdetrv2/visdrone_ega_sgw.yml` |
| P3SR + SGW | `configs/rtdetrv2/visdrone_p3sr_sgw.yml` |
| Complete model | `configs/rtdetrv2/visdrone_p3sr_ega_sgw.yml` |
| CIoU | `configs/rtdetrv2/visdrone_ciou.yml` |
| EIoU | `configs/rtdetrv2/visdrone_eiou.yml` |
| NWD | `configs/rtdetrv2/visdrone_nwd.yml` |

Raw logs, checkpoints, outputs, and datasets are intentionally excluded from
the repository. CoDrone configurations are supplementary and are not the
source of any result reported above.

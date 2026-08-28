import torch
from torch import Tensor


def inner_iou_score(
    boxes1: Tensor, boxes2: Tensor, ratio: float = 0.5, eps: float = 1e-7
) -> Tensor:
    x1, y1, w1, h1 = boxes1.unsqueeze(1).unbind(-1)
    x2, y2, w2, h2 = boxes2.unsqueeze(0).unbind(-1)

    b1_x1 = x1 - (w1 * ratio) / 2
    b1_x2 = x1 + (w1 * ratio) / 2
    b1_y1 = y1 - (h1 * ratio) / 2
    b1_y2 = y1 + (h1 * ratio) / 2

    b2_x1 = x2 - (w2 * ratio) / 2
    b2_x2 = x2 + (w2 * ratio) / 2
    b2_y1 = y2 - (h2 * ratio) / 2
    b2_y2 = y2 + (h2 * ratio) / 2

    inter_w = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0)
    inter_h = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(min=0)
    inter = inter_w * inter_h

    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / union


def inner_iou_loss(
    boxes1: Tensor, boxes2: Tensor, ratio: float = 0.75, eps: float = 1e-7
) -> Tensor:
    x1, y1, w1, h1 = boxes1.unbind(-1)
    x2, y2, w2, h2 = boxes2.unbind(-1)

    b1_x1 = x1 - (w1 * ratio) / 2
    b1_x2 = x1 + (w1 * ratio) / 2
    b1_y1 = y1 - (h1 * ratio) / 2
    b1_y2 = y1 + (h1 * ratio) / 2

    b2_x1 = x2 - (w2 * ratio) / 2
    b2_x2 = x2 + (w2 * ratio) / 2
    b2_y1 = y2 - (h2 * ratio) / 2
    b2_y2 = y2 + (h2 * ratio) / 2

    inter_w = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0)
    inter_h = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(min=0)
    inter = inter_w * inter_h

    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    iou = inter / union
    return 1.0 - iou

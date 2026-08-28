import math

import torch


def _upcast_non_float32(boxes):
    if boxes.dtype in (torch.float16, torch.bfloat16):
        return boxes.float()
    return boxes


def _aligned_iou_geometry(pred_boxes, target_boxes, eps=1e-7):
    """Return aligned IoU and enclosing-box geometry for ``cxcywh`` boxes."""

    cx_p, cy_p, w_p, h_p = pred_boxes.unbind(-1)
    cx_t, cy_t, w_t, h_t = target_boxes.unbind(-1)

    pred_left = cx_p - w_p / 2
    pred_top = cy_p - h_p / 2
    pred_right = cx_p + w_p / 2
    pred_bottom = cy_p + h_p / 2

    target_left = cx_t - w_t / 2
    target_top = cy_t - h_t / 2
    target_right = cx_t + w_t / 2
    target_bottom = cy_t + h_t / 2

    inter_width = (
        torch.minimum(pred_right, target_right)
        - torch.maximum(pred_left, target_left)
    ).clamp(min=0)
    inter_height = (
        torch.minimum(pred_bottom, target_bottom)
        - torch.maximum(pred_top, target_top)
    ).clamp(min=0)
    intersection = inter_width * inter_height

    pred_area = w_p.clamp(min=0) * h_p.clamp(min=0)
    target_area = w_t.clamp(min=0) * h_t.clamp(min=0)
    union = pred_area + target_area - intersection
    iou = intersection / (union + eps)

    enclosing_width = (
        torch.maximum(pred_right, target_right)
        - torch.minimum(pred_left, target_left)
    )
    enclosing_height = (
        torch.maximum(pred_bottom, target_bottom)
        - torch.minimum(pred_top, target_top)
    )

    return iou, enclosing_width, enclosing_height


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """Complete IoU loss for aligned normalized ``cxcywh`` boxes."""

    pred_boxes = _upcast_non_float32(pred_boxes)
    target_boxes = _upcast_non_float32(target_boxes)
    iou, enclosing_width, enclosing_height = _aligned_iou_geometry(
        pred_boxes, target_boxes, eps=eps
    )
    center_distance = (pred_boxes[..., :2] - target_boxes[..., :2]).pow(2).sum(-1)
    enclosing_diagonal = enclosing_width.pow(2) + enclosing_height.pow(2) + eps

    pred_width, pred_height = pred_boxes[..., 2:].unbind(-1)
    target_width, target_height = target_boxes[..., 2:].unbind(-1)
    aspect_ratio = (4.0 / math.pi**2) * (
        torch.atan(target_width / target_height.clamp(min=eps))
        - torch.atan(pred_width / pred_height.clamp(min=eps))
    ).pow(2)
    with torch.no_grad():
        aspect_weight = aspect_ratio / (1.0 - iou + aspect_ratio + eps)

    return (
        1.0
        - iou
        + center_distance / enclosing_diagonal
        + aspect_weight * aspect_ratio
    )


def eiou_loss(pred_boxes, target_boxes, eps=1e-7):
    """Efficient IoU loss for aligned normalized ``cxcywh`` boxes."""

    pred_boxes = _upcast_non_float32(pred_boxes)
    target_boxes = _upcast_non_float32(target_boxes)
    iou, enclosing_width, enclosing_height = _aligned_iou_geometry(
        pred_boxes, target_boxes, eps=eps
    )
    center_distance = (pred_boxes[..., :2] - target_boxes[..., :2]).pow(2).sum(-1)
    enclosing_diagonal = enclosing_width.pow(2) + enclosing_height.pow(2) + eps

    width_distance = (pred_boxes[..., 2] - target_boxes[..., 2]).pow(2)
    height_distance = (pred_boxes[..., 3] - target_boxes[..., 3]).pow(2)

    return (
        1.0
        - iou
        + center_distance / enclosing_diagonal
        + width_distance / (enclosing_width.pow(2) + eps)
        + height_distance / (enclosing_height.pow(2) + eps)
    )

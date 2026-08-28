import torch
from .box_ops import generalized_box_iou, box_cxcywh_to_xyxy


def nwd_adaptive(pred_boxes, target_boxes, tau=1.5, eps=1e-6):

    cx_p, cy_p, w_p, h_p = pred_boxes.unbind(-1)
    cx_t, cy_t, w_t, h_t = target_boxes.unbind(-1)

    w2_sq = (
        (cx_p - cx_t) ** 2
        + (cy_p - cy_t) ** 2
        + ((w_p - w_t) ** 2 + (h_p - h_t) ** 2) / 4.0
    )

    scale_gt = torch.sqrt((w_t * h_t).clamp(min=eps))
    c_i = tau * scale_gt

    nwd = torch.exp(-torch.sqrt(w2_sq + eps) / c_i)
    return nwd


def nwd_loss(pred_boxes, target_boxes, constant=12.8, image_size=640.0, eps=1e-7):
    """Standard NWD loss for normalized ``cxcywh`` boxes."""

    pred_boxes_px = pred_boxes * image_size
    target_boxes_px = target_boxes * image_size
    center_distance = (pred_boxes_px[..., :2] - target_boxes_px[..., :2]).pow(2).sum(-1)
    wh_distance = (
        (pred_boxes_px[..., 2:] - target_boxes_px[..., 2:]).pow(2).sum(-1) / 4.0
    )
    wasserstein_distance = torch.sqrt(center_distance + wh_distance + eps)
    return 1.0 - torch.exp(-wasserstein_distance / constant)


def sgw_loss(pred_boxes, target_boxes, tau=1.5, gamma=100.0):

    nwd = nwd_adaptive(pred_boxes, target_boxes, tau=tau)
    loss_nwd = 1.0 - nwd

    giou = torch.diag(
        generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes.clone()), box_cxcywh_to_xyxy(target_boxes)
        )
    )
    loss_giou = 1.0 - giou

    _, _, w_t, h_t = target_boxes.unbind(-1)
    area_gt = w_t * h_t
    lambda_weight = torch.exp(-gamma * area_gt)

    fused_loss = lambda_weight * loss_nwd + (1.0 - lambda_weight) * loss_giou

    return fused_loss

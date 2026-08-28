import random
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image

from .._misc import convert_to_tv_tensor, _boxes_keys
from ...core import register


@register()
class SmallObjectCopyPaste(T.Transform):

    def __init__(self, p=0.5, area_threshold=1024, max_copies=10, iou_threshold=0.3):
        super().__init__()
        self.p = p
        self.area_threshold = area_threshold
        self.max_copies = max_copies
        self.iou_threshold = iou_threshold

    def _iou(self, box, boxes):

        if len(boxes) == 0:
            return torch.tensor([0.0])

        inter_x1 = torch.maximum(box[0], boxes[:, 0])
        inter_y1 = torch.maximum(box[1], boxes[:, 1])
        inter_x2 = torch.minimum(box[2], boxes[:, 2])
        inter_y2 = torch.minimum(box[3], boxes[:, 3])

        inter_area = torch.maximum(
            torch.tensor(0.0), inter_x2 - inter_x1
        ) * torch.maximum(torch.tensor(0.0), inter_y2 - inter_y1)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box_area + boxes_area - inter_area
        return inter_area / torch.clamp(union_area, min=1e-6)

    def forward(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        inputs = inputs if len(inputs) > 1 else inputs[0]
        image, target = inputs[0], inputs[1]

        dataset = inputs[2] if len(inputs) > 2 else None

        if "boxes" not in target or len(target["boxes"]) == 0:
            return inputs if len(inputs) > 1 else inputs[0]

        boxes = target["boxes"]
        labels = target["labels"]

        w, h = image.size

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        small_indices = torch.where(areas < self.area_threshold)[0]

        if len(small_indices) == 0:
            return inputs if len(inputs) > 1 else inputs[0]

        small_indices = small_indices.tolist()
        random.shuffle(small_indices)
        small_indices = small_indices[: self.max_copies]

        new_boxes = []
        new_labels = []

        all_boxes = boxes.clone()

        new_image = image.copy()

        for idx in small_indices:
            box = boxes[idx]
            label = labels[idx]
            bw, bh = int(box[2] - box[0]), int(box[3] - box[1])

            if bw <= 1 or bh <= 1:
                continue

            crop_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            try:
                crop_patch = image.crop(crop_box)
            except Exception:
                continue

            for _ in range(10):
                new_x = random.randint(0, max(1, w - bw - 1))
                new_y = random.randint(0, max(1, h - bh - 1))

                new_box = torch.tensor(
                    [new_x, new_y, new_x + bw, new_y + bh], dtype=boxes.dtype
                )

                ious = self._iou(new_box, all_boxes)
                if ious.max() < self.iou_threshold:

                    new_image.paste(crop_patch, (new_x, new_y))
                    new_boxes.append(new_box)
                    new_labels.append(label)
                    all_boxes = torch.cat([all_boxes, new_box.unsqueeze(0)], dim=0)
                    break

        if len(new_boxes) > 0:

            new_boxes_tensor = torch.stack(new_boxes).to(boxes.device)
            new_labels_tensor = torch.stack(new_labels).to(labels.device)

            combined_boxes = torch.cat([boxes, new_boxes_tensor], dim=0)
            combined_labels = torch.cat([labels, new_labels_tensor], dim=0)

            target["boxes"] = convert_to_tv_tensor(
                combined_boxes, "boxes", box_format="xyxy", spatial_size=[h, w]
            )
            target["labels"] = combined_labels

            if "iscrowd" in target:
                target["iscrowd"] = torch.cat(
                    [
                        target["iscrowd"],
                        torch.zeros(
                            len(new_boxes),
                            dtype=target["iscrowd"].dtype,
                            device=target["iscrowd"].device,
                        ),
                    ]
                )
            if "area" in target:
                new_areas = (new_boxes_tensor[:, 2] - new_boxes_tensor[:, 0]) * (
                    new_boxes_tensor[:, 3] - new_boxes_tensor[:, 1]
                )
                target["area"] = torch.cat([target["area"], new_areas])

        if dataset is not None:
            return new_image, target, dataset
        return new_image, target

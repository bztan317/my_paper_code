import torch
import os
from PIL import Image
from typing import Optional, Callable

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class YOLOOBBDetection(DetDataset):
    __inject__ = [
        "transforms",
    ]

    def __init__(
        self,
        img_folder: str,
        ann_folder: str,
        class_file: str = None,
        transforms: Optional[Callable] = None,
    ):
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.transforms = transforms

        if class_file and os.path.exists(class_file):
            with open(class_file, "r") as f:
                self.class_names = [
                    line.strip() for line in f.readlines() if line.strip()
                ]
        else:

            self.class_names = ["truck", "bus", "van", "lorry", "car"]

        self.num_classes = len(self.class_names)

        self.image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            self.image_files.extend(
                [f for f in os.listdir(img_folder) if f.lower().endswith(ext)]
            )
        self.image_files.sort()

        print(
            f"[YOLOOBBDetection] Loaded {len (self .image_files )} images from {img_folder }"
        )
        print(
            f"[YOLOOBBDetection] Classes: {self .num_classes } - {self .class_names }"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)
        return image, target

    def load_item(self, index: int):

        img_name = self.image_files[index]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.ann_folder, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])

                    if len(parts) >= 9:

                        coords = [float(x) for x in parts[1:9]]
                        xs = coords[0::2]
                        ys = coords[1::2]
                        xmin = min(xs) * w
                        ymin = min(ys) * h
                        xmax = max(xs) * w
                        ymax = max(ys) * h
                    else:

                        cx, cy, bw, bh = map(float, parts[1:5])

                        cx_abs = cx * w
                        cy_abs = cy * h
                        bw_abs = bw * w
                        bh_abs = bh * h

                        xmin = cx_abs - bw_abs / 2
                        ymin = cy_abs - bh_abs / 2
                        xmax = cx_abs + bw_abs / 2
                        ymax = cy_abs + bh_abs / 2

                    xmin = max(0, min(xmin, w))
                    ymin = max(0, min(ymin, h))
                    xmax = max(0, min(xmax, w))
                    ymax = max(0, min(ymax, h))

                    if xmax - xmin < 1 or ymax - ymin < 1:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = convert_to_tv_tensor(
            boxes_tensor, "boxes", box_format="xyxy", spatial_size=[h, w]
        )
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([index])
        target["area"] = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if len(boxes) > 0
            else torch.tensor([])
        )
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        target["orig_size"] = torch.tensor([h, w])

        return image, target

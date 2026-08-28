import torch
import os
from PIL import Image
from pathlib import Path
from typing import Optional, Callable

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class VisDroneDetection(DetDataset):
    __inject__ = [
        "transforms",
    ]

    def __init__(
        self, img_folder: str, ann_folder: str, transforms: Optional[Callable] = None
    ):
        super().__init__()
        self.img_folder = Path(img_folder)
        self.ann_folder = Path(ann_folder)
        self.transforms = transforms

        self.images = sorted(
            [
                p
                for p in self.img_folder.glob("*")
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        self.targets = []

        for img_path in self.images:
            ann_path = self.ann_folder / f"{img_path .stem }.txt"
            self.targets.append(ann_path)

        self.category2name = {
            1: "pedestrian",
            2: "people",
            3: "bicycle",
            4: "car",
            5: "van",
            6: "truck",
            7: "tricycle",
            8: "awning-tricycle",
            9: "bus",
            10: "motor",
        }
        self.category2label = {k: k - 1 for k in self.category2name.keys()}
        self.label2category = {v: k for k, v in self.category2label.items()}

    def __len__(self):
        return len(self.images)

    def load_item(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        w, h = image.size

        ann_path = self.targets[index]

        output = {}
        output["image_id"] = torch.tensor([index])
        for k in ["area", "boxes", "labels", "iscrowd"]:
            output[k] = []

        if ann_path.exists():
            for line in ann_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",") if "," in line else line.split()
                if len(parts) < 8:
                    continue

                try:
                    x, y, bw, bh = map(float, parts[0:4])
                    cat = int(float(parts[5]))
                except ValueError:
                    continue

                if cat <= 0 or cat > 10:
                    continue
                if bw <= 1 or bh <= 1:
                    continue

                box = [x, y, x + bw, y + bh]

                output["boxes"].append(box)
                output["labels"].append(self.category2label[cat])
                output["area"].append(bw * bh)
                output["iscrowd"].append(0)

        boxes = (
            torch.tensor(output["boxes"], dtype=torch.float32)
            if len(output["boxes"]) > 0
            else torch.zeros(0, 4)
        )
        output["boxes"] = convert_to_tv_tensor(
            boxes, "boxes", box_format="xyxy", spatial_size=[h, w]
        )
        output["labels"] = torch.tensor(output["labels"], dtype=torch.int64)
        output["area"] = torch.tensor(output["area"], dtype=torch.float32)
        output["iscrowd"] = torch.tensor(output["iscrowd"], dtype=torch.int64)
        output["orig_size"] = torch.tensor([w, h])

        return image, output

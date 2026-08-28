import torch
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from typing import Optional, Callable

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class CODroneDetection(DetDataset):
    __inject__ = [
        "transforms",
    ]

    def __init__(
        self,
        root: str,
        ann_file: str = "train.txt",
        label_file: str = "label_list.txt",
        transforms: Optional[Callable] = None,
    ):

        import re

        with open(os.path.join(root, ann_file), "r") as f:
            lines = []
            for x in f.readlines():
                x = x.strip()
                if not x:
                    continue

                if "," in x:
                    parts = x.split(",", 1)
                else:

                    match = re.search(r"\.(jpg|jpeg|png|bmp)\s+", x, re.IGNORECASE)
                    if match:
                        split_idx = match.end() - 1
                        parts = [x[:split_idx].strip(), x[split_idx:].strip()]
                    else:
                        parts = x.split(None, 1)

                if len(parts) == 2:
                    lines.append(parts)
                else:
                    print(f"[Warning] Ignore invalid line in {ann_file }: '{x }'")

        def _join(root_dir: str, p: str) -> str:
            return p if os.path.isabs(p) else os.path.join(root_dir, p)

        self.images = [_join(root, lin[0]) for lin in lines]
        self.targets = [_join(root, lin[1]) for lin in lines]
        assert len(self.images) == len(self.targets)

        with open(os.path.join(root, label_file), "r") as f:
            labels = f.readlines()
            labels = [lab.strip() for lab in labels]

        self.transforms = transforms
        self.labels_map = {lab: i for i, lab in enumerate(labels)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)

        return image, target

    def load_item(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        target_tree = ET_parse(self.targets[index]).getroot()

        output = {}
        output["image_id"] = torch.tensor([index])
        for k in ["area", "boxes", "labels", "iscrowd"]:
            output[k] = []

        for obj in target_tree.findall("object"):
            name = obj.findtext("name")
            if name not in self.labels_map:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            if bndbox.find("x0") is not None:
                xs = [float(bndbox.findtext(f"x{i }")) for i in range(4)]
                ys = [float(bndbox.findtext(f"y{i }")) for i in range(4)]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)

            elif bndbox.find("xmin") is not None:
                xmin = float(bndbox.findtext("xmin"))
                ymin = float(bndbox.findtext("ymin"))
                xmax = float(bndbox.findtext("xmax"))
                ymax = float(bndbox.findtext("ymax"))
            else:
                continue

            if xmax - xmin <= 1 or ymax - ymin <= 1:
                continue

            box = [xmin, ymin, xmax, ymax]
            output["boxes"].append(box)
            output["labels"].append(name)
            output["area"].append((xmax - xmin) * (ymax - ymin))
            output["iscrowd"].append(0)

        w, h = image.size
        boxes = (
            torch.tensor(output["boxes"])
            if len(output["boxes"]) > 0
            else torch.zeros(0, 4)
        )
        output["boxes"] = convert_to_tv_tensor(
            boxes, "boxes", box_format="xyxy", spatial_size=[h, w]
        )

        output["labels"] = torch.tensor(
            [self.labels_map[lab] for lab in output["labels"]], dtype=torch.int64
        )
        output["area"] = torch.tensor(output["area"], dtype=torch.float32)
        output["iscrowd"] = torch.tensor(output["iscrowd"], dtype=torch.int64)
        output["orig_size"] = torch.tensor([w, h])

        return image, output

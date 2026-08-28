import random
import torch
import os
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Optional, Callable

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class UAVDTDetection(DetDataset):
    __inject__ = [
        "transforms",
    ]

    def __init__(
        self,
        img_folder: str,
        ann_folder: str,
        split_file: str,
        transforms: Optional[Callable] = None,
        mask_ignore: bool = True,
        split_ratio: Optional[float] = None,
        split_part: str = "all",
        split_seed: int = 42,
        split_level: str = "sequence",
        **kwargs,
    ):
        super().__init__()
        self.img_folder = Path(img_folder)
        self.ann_folder = Path(ann_folder)
        self.split_folder = Path(split_file)
        self.transforms = transforms
        self.mask_ignore = mask_ignore
        self.split_ratio = split_ratio
        self.split_part = split_part
        self.split_seed = split_seed
        self.split_level = split_level

        all_sequences = []
        for attr_file in sorted(self.split_folder.glob("*_attr.txt")):
            seq_name = attr_file.stem.replace("_attr", "").strip()
            seq_path = self.img_folder / seq_name
            if seq_path.exists():
                all_sequences.append(seq_name)

        self.sequences = self._select_sequences(all_sequences)

        self.category2name = {
            1: "car",
            2: "truck",
            3: "bus",
        }
        self.category2label = {k: k - 1 for k in self.category2name.keys()}
        self.label2category = {v: k for k, v in self.category2label.items()}

        self.images = []
        self.targets = []

        for seq_name in self.sequences:
            seq_img_folder = self.img_folder / seq_name
            ann_file = self.ann_folder / f"{seq_name }_gt_whole.txt"
            ignore_file = self.ann_folder / f"{seq_name }_gt_ignore.txt"

            if not ann_file.exists():
                continue

            gt_dict = {}
            for line in ann_file.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 9:
                    continue
                try:
                    frame_idx = int(parts[0])
                    x = float(parts[2])
                    y = float(parts[3])
                    bw = float(parts[4])
                    bh = float(parts[5])
                    cat = int(parts[8])
                except (ValueError, IndexError):
                    continue

                if cat <= 0 or cat > 3:
                    continue
                if bw <= 1 or bh <= 1:
                    continue

                if frame_idx not in gt_dict:
                    gt_dict[frame_idx] = []
                gt_dict[frame_idx].append(
                    {
                        "box": [x, y, x + bw, y + bh],
                        "label": self.category2label[cat],
                        "area": bw * bh,
                    }
                )

            ignore_dict = {}
            if self.mask_ignore and ignore_file.exists():
                for line in ignore_file.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        frame_idx = int(parts[0])
                        ix = float(parts[2])
                        iy = float(parts[3])
                        iw = float(parts[4])
                        ih = float(parts[5])

                        if frame_idx not in ignore_dict:
                            ignore_dict[frame_idx] = []
                        ignore_dict[frame_idx].append([ix, iy, iw, ih])
                    except (ValueError, IndexError):
                        continue

            img_files = sorted([p for p in seq_img_folder.glob("*.jpg")])
            for img_path in img_files:
                frame_num = int(img_path.stem.replace("img", ""))
                self.images.append(img_path)
                self.targets.append(
                    {
                        "gt": gt_dict.get(frame_num, []),
                        "ignore": ignore_dict.get(frame_num, []),
                    }
                )

        if self.split_level == "image":
            self._apply_image_holdout_split()

    def _select_sequences(self, all_sequences):
        if self.split_ratio is None or self.split_part == "all":
            return all_sequences
        if self.split_part not in ("train", "val"):
            raise ValueError(
                f"split_part must be 'all', 'train', or 'val', got {self .split_part !r }"
            )
        if not (0.0 < self.split_ratio < 1.0):
            raise ValueError(f"split_ratio must be in (0, 1), got {self .split_ratio }")
        if self.split_level not in ("sequence", "image"):
            raise ValueError(
                f"split_level must be 'sequence' or 'image', got {self .split_level !r }"
            )
        if self.split_level == "image":
            return all_sequences

        n = len(all_sequences)
        if n == 0:
            return all_sequences

        rng = random.Random(self.split_seed)
        indices = list(range(n))
        rng.shuffle(indices)
        split_idx = max(1, int(n * self.split_ratio))
        split_idx = min(split_idx, n - 1)
        if self.split_part == "train":
            keep = sorted(indices[:split_idx])
        else:
            keep = sorted(indices[split_idx:])
        return [all_sequences[i] for i in keep]

    def _apply_image_holdout_split(self):
        if self.split_ratio is None or self.split_part == "all":
            return
        if self.split_part not in ("train", "val"):
            raise ValueError(
                f"split_part must be 'all', 'train', or 'val', got {self .split_part !r }"
            )
        if not (0.0 < self.split_ratio < 1.0):
            raise ValueError(f"split_ratio must be in (0, 1), got {self .split_ratio }")

        n = len(self.images)
        if n == 0:
            return

        rng = random.Random(self.split_seed)
        indices = list(range(n))
        rng.shuffle(indices)
        split_idx = int(n * self.split_ratio)
        if self.split_part == "train":
            keep = sorted(indices[:split_idx])
        else:
            keep = sorted(indices[split_idx:])

        self.images = [self.images[i] for i in keep]
        self.targets = [self.targets[i] for i in keep]

    def __len__(self):
        return len(self.images)

    def load_item(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        w, h = image.size

        target_info = self.targets[index]

        if self.mask_ignore and len(target_info["ignore"]) > 0:
            draw = ImageDraw.Draw(image)
            for ix, iy, iw, ih in target_info["ignore"]:
                draw.rectangle([ix, iy, ix + iw, iy + ih], fill=(120, 120, 120))

        output = {}
        output["image_id"] = torch.tensor([index])
        for k in ["area", "boxes", "labels", "iscrowd"]:
            output[k] = []

        for obj in target_info["gt"]:
            output["boxes"].append(obj["box"])
            output["labels"].append(obj["label"])
            output["area"].append(obj["area"])
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

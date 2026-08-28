import torch
import torch.nn as nn

from datetime import datetime
import sys
from pathlib import Path
from typing import Dict
import atexit

from ..misc import dist_utils
from ..core import BaseConfig


class _FileLogger:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def to(m: nn.Module, device: str):
    if m is None:
        return None
    return m.to(device)


def remap_key(key: str) -> str:
    k = key
    if "encoder.edge_fusion_neck" in k:
        k = k.replace("encoder.edge_fusion_neck", "encoder.ega_neck")
    if "encoder.ega_neck.fusion_p3" in k:
        k = k.replace("encoder.ega_neck.fusion_p3", "encoder.ega_neck.fusions.0")
    elif "encoder.ega_neck.fusion_p4" in k:
        k = k.replace("encoder.ega_neck.fusion_p4", "encoder.ega_neck.fusions.1")
    elif "encoder.ega_neck.fusion_p5" in k:
        k = k.replace("encoder.ega_neck.fusion_p5", "encoder.ega_neck.fusions.2")
    return k


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg
        try:
            from ..data.dataset.aitod_dataset import AITODDetection
            from ..data.dataset.ds_dataset import YOLOOBBDetection
        except ImportError:
            pass
        if "val_dataloader" in self.cfg.yaml_cfg:
            dataset_cfg = self.cfg.yaml_cfg["val_dataloader"].get("dataset", {})
            if dataset_cfg and dataset_cfg.get("type") in (
                "AITODDetection",
                "YOLOOBBDetection",
            ):
                dataset_cfg.pop("root", None)
                dataset_cfg.pop("image_set", None)

    def _setup(
        self,
    ):

        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = cfg.model

        if self.cfg.tuning:
            print(f"tuning checkpoint from {self .cfg .tuning }")
            self.load_tuning_state(self.cfg.tuning)

        self.model = dist_utils.warp_model(
            self.model.to(device),
            sync_bn=cfg.sync_bn,
            find_unused_parameters=cfg.find_unused_parameters,
        )

        self.criterion = to(cfg.criterion, device)
        self.postprocessor = to(cfg.postprocessor, device)

        self.ema = to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = self.cfg.last_epoch

        base_output_dir = Path(cfg.output_dir)
        output_dir = base_output_dir

        is_resume_or_eval = getattr(self.cfg, "resume", None) is not None or getattr(
            self.cfg, "test_only", False
        )

        if output_dir.exists() and not is_resume_or_eval:

            idx = 1
            while True:
                cand = base_output_dir.parent / f"{base_output_dir .name }{idx }"
                if not cand.exists():
                    output_dir = cand
                    break
                idx += 1

        self.output_dir = output_dir

        self.cfg.output_dir = str(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if dist_utils.is_main_process():
            log_file_path = self.output_dir / "run.log"
            self.run_log_file = open(log_file_path, "a", encoding="utf-8")

            self.run_log_file.write(
                f"\n{'='*50 }\n--- New Run Started: {datetime .now ().isoformat ()} ---\n{'='*50 }\n"
            )

            if not isinstance(sys.stdout, _FileLogger):
                sys.stdout = _FileLogger(sys.stdout, self.run_log_file)
            if not isinstance(sys.stderr, _FileLogger):
                sys.stderr = _FileLogger(sys.stderr, self.run_log_file)

            conf_dict = {
                k: v for k, v in self.cfg.__dict__.items() if not str(k).startswith("_")
            }
            priority = [
                "yaml_cfg",
                "output_dir",
                "epoches",
                "device",
                "resume",
                "tuning",
            ]

            core_items = []
            for k in priority:
                if k in conf_dict:
                    val = conf_dict.pop(k)
                    if k == "yaml_cfg" and isinstance(val, dict):
                        core_items.append(f"task: {val .get ('task','N/A')}")
                    else:
                        core_items.append(f"{k }: {val }")

            mod_items = []
            for k, v in conf_dict.items():
                val_str = str(v)
                if "\n" in val_str or len(val_str) > 100:
                    mod_items.append(f"{k }: <{type (v ).__name__ }>")
                else:
                    mod_items.append(f"{k }: {val_str }")

            print(f"[Config] {' | '.join (core_items )}")
            print(f"[Modules] {' | '.join (mod_items )}\n")

        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text(f"config", "{:s}".format(cfg.__repr__()), 0)

    def cleanup(
        self,
    ):
        if self.writer:
            atexit.register(self.writer.close)
        if hasattr(self, "run_log_file"):
            if isinstance(sys.stdout, _FileLogger):
                sys.stdout = sys.stdout.stream
            if isinstance(sys.stderr, _FileLogger):
                sys.stderr = sys.stderr.stream
            self.run_log_file.close()

    def _patch_aitod_dataset(self):
        if not hasattr(self, "val_dataloader") or self.val_dataloader is None:
            return None
        dataset = self.val_dataloader.dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        is_aitod = dataset.__class__.__name__ == "AITODDetection"
        is_yoloobb = dataset.__class__.__name__ == "YOLOOBBDetection"
        if is_aitod or is_yoloobb:
            from PIL import Image
            import os
            import types

            def fast_load_item(self_ds, index: int):
                img_name = self_ds.image_files[index]
                img_path = os.path.join(self_ds.img_folder, img_name)
                image = Image.open(img_path)
                w, h = image.size

                label_name = os.path.splitext(img_name)[0] + ".txt"
                label_path = os.path.join(self_ds.ann_folder, label_name)

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

                            if is_yoloobb and len(parts) >= 9:
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
                if is_aitod:
                    from ..data.dataset.aitod_dataset import convert_to_tv_tensor
                else:
                    from ..data.dataset.ds_dataset import convert_to_tv_tensor
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

            orig_load_item = dataset.load_item
            dataset.load_item = types.MethodType(fast_load_item, dataset)
            print("[Info] Temporarily patched load_item for fast COCO API build...")
            return (dataset, orig_load_item)
        return None

    def _unpatch_aitod_dataset(self, patch_info):
        if patch_info:
            dataset, orig_load_item = patch_info
            dataset.load_item = orig_load_item
            print("[Info] Restored original load_item for evaluation")

    def train(
        self,
    ):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle
        )
        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        patch_info = self._patch_aitod_dataset()
        self.evaluator = self.cfg.evaluator
        self._unpatch_aitod_dataset(patch_info)

        if self.cfg.resume:
            print(f"Resume checkpoint from {self .cfg .resume }")
            self.load_resume_state(self.cfg.resume)

    def eval(
        self,
    ):
        self._setup()

        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        patch_info = self._patch_aitod_dataset()
        self.evaluator = self.cfg.evaluator
        self._unpatch_aitod_dataset(patch_info)

        if self.cfg.resume:
            print(f"Resume checkpoint from {self .cfg .resume }")
            self.load_resume_state(self.cfg.resume)

    def to(self, device):
        for k, v in self.__dict__.items():
            if hasattr(v, "to"):
                v.to(device)

    def state_dict(self):

        state = {}
        state["date"] = datetime.now().isoformat()

        state["last_epoch"] = self.last_epoch

        for k, v in self.__dict__.items():
            if hasattr(v, "state_dict"):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state

    def _remap_state_dict(self, state_dict):
        new_state_dict = {}
        for mk, mv in state_dict.items():
            new_state_dict[remap_key(mk)] = mv
        return new_state_dict

    def load_state_dict(self, state):

        if "last_epoch" in state:
            self.last_epoch = state["last_epoch"]
            print("Load last_epoch")

        for k, v in self.__dict__.items():
            if hasattr(v, "load_state_dict") and k in state:
                v = dist_utils.de_parallel(v)
                val_state = state[k]
                if k == "model":
                    val_state = self._remap_state_dict(val_state)
                elif k == "ema" and "module" in val_state:
                    val_state["module"] = self._remap_state_dict(val_state["module"])
                v.load_state_dict(val_state)
                print(f"Load {k }.state_dict")

            if hasattr(v, "load_state_dict") and k not in state:
                print(f"Not load {k }.state_dict")

    def load_resume_state(self, path: str):

        if path.startswith("http"):
            state = torch.hub.load_state_dict_from_url(path, map_location="cpu")
        else:
            state = torch.load(path, map_location="cpu")

        self.load_state_dict(state)

    def load_tuning_state(
        self,
        path: str,
    ):

        if path.startswith("http"):
            state = torch.hub.load_state_dict_from_url(path, map_location="cpu")
        else:
            state = torch.load(path, map_location="cpu")

        module = dist_utils.de_parallel(self.model)

        if "ema" in state:
            loaded_module = self._remap_state_dict(state["ema"]["module"])
            stat, infos = self._matched_state(module.state_dict(), loaded_module)
        else:
            loaded_module = self._remap_state_dict(state["model"])
            stat, infos = self._matched_state(module.state_dict(), loaded_module)

        module.load_state_dict(stat, strict=False)
        print(f"Load model.state_dict, {infos }")

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {"missed": missed_list, "unmatched": unmatched_list}

    def fit(
        self,
    ):
        raise NotImplementedError("")

    def val(
        self,
    ):
        raise NotImplementedError("")

import time
import json
import datetime
import math

import numpy as np
import torch

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


def _format_hms(seconds: float) -> str:

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    if ms == 1000:
        ms = 999
    return f"{h }:{m :02d}:{s :02d}.{ms :03d}"


def _get_ap50_by_area(coco_evaluator) -> dict:

    empty = {
        "ap50_all": -1.0,
        "ap50_small": -1.0,
        "ap50_medium": -1.0,
        "ap50_large": -1.0,
    }

    if coco_evaluator is None:
        return empty
    if "bbox" not in coco_evaluator.coco_eval:
        return empty

    coco_eval = coco_evaluator.coco_eval["bbox"]
    if not hasattr(coco_eval, "eval") or coco_eval.eval is None:
        return empty

    precision = coco_eval.eval["precision"]

    results = {}
    for area_idx, key in enumerate(
        ["ap50_all", "ap50_small", "ap50_medium", "ap50_large"]
    ):

        p = precision[0, :, :, area_idx, 2]
        valid = p[p > -1]
        results[key] = float(valid.mean()) if valid.size > 0 else 0.0

    return results


def _generate_plots(output_dir):

    import importlib.util, sys
    from pathlib import Path

    log_path = output_dir / "log.txt"
    if not log_path.exists():
        print("[plot] log.txt not found, skipping chart generation.")
        return

    solver_dir = Path(__file__).resolve().parent
    project_root = solver_dir.parent.parent
    plot_script = project_root / "tools" / "plot_training_log.py"

    if not plot_script.exists():
        print(f"[plot] Could not find {plot_script }, skipping.")
        return

    spec = importlib.util.spec_from_file_location("plot_training_log", plot_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        raw_rows = mod.load_rows(log_path)
        rows, duplicate_lines = mod.dedupe_by_epoch(raw_rows)
        mod.plot_train_curves(
            rows, output_dir / "train_curves.png", len(duplicate_lines)
        )
        mod.plot_eval_curves(rows, output_dir / "eval_curves.png")
        mod.write_summary(
            rows, raw_rows, duplicate_lines, output_dir / "log_summary.txt"
        )
        print(f"[plot] Charts saved to {output_dir }")
    except Exception as exc:
        print(f"[plot] Chart generation failed: {exc }")


class DetSolver(BaseSolver):

    def fit(
        self,
    ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        flops_shape = (1, 3, 640, 640)
        n_flops = None
        if dist_utils.is_main_process():
            n_flops = profiler_utils.estimate_forward_flops(
                dist_utils.de_parallel(self.model), input_shape=flops_shape
            )
        if n_flops is not None:
            print(
                f"number of trainable parameters: {n_parameters }  |  "
                f"estimated forward: {n_flops /1e9 :.2f} GFLOPs @ input{flops_shape }"
            )
        else:
            print(
                f"number of trainable parameters: {n_parameters }  |  "
                f"estimated forward GFLOPs: N/A (profiler reported no flops)"
            )

        best_stat = {"epoch": -1}
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        for epoch in range(start_epoch, args.epoches):
            epoch_start = time.time()

            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch :04}.pth"
                    )
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model

            eval_print_freq = getattr(args, "eval_print_freq", 10)
            if eval_print_freq is None:
                eval_print_freq = 10

            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                print_freq=eval_print_freq,
            )

            ap50_area = _get_ap50_by_area(coco_evaluator)
            if dist_utils.is_main_process():
                print(
                    f"  AP@0.50 | all={ap50_area ['ap50_all']:.4f}"
                    f"  small={ap50_area ['ap50_small']:.4f}"
                    f"  medium={ap50_area ['ap50_medium']:.4f}"
                    f"  large={ap50_area ['ap50_large']:.4f}"
                )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k }_{i }", v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat["epoch"] == epoch and self.output_dir:
                    dist_utils.save_on_master(
                        self.state_dict(), self.output_dir / "best.pth"
                    )

            epoch_time = time.time() - epoch_start
            epoch_time_hms = _format_hms(epoch_time)

            cumulative_seconds = time.time() - start_time
            cumulative_hms = _format_hms(cumulative_seconds)

            print(
                f"Epoch [{epoch }] time: {epoch_time_hms } | Cumulative: {cumulative_hms }"
            )
            print(f"best_stat: {best_stat }")

            log_stats = {
                **{f"train_{k }": v for k, v in train_stats.items()},
                **{f"test_{k }": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
                **(
                    {"n_gflops": round(n_flops / 1e9, 6)} if n_flops is not None else {}
                ),
                "epoch_time_sec": round(epoch_time, 6),
                "epoch_time_hms": epoch_time_hms,
                "cumulative_sec": round(cumulative_seconds, 6),
                "cumulative_hms": cumulative_hms,
                **ap50_area,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                total_loss = train_stats.get("loss", 0.0)
                lr = train_stats.get("lr", 0.0)
                test_bbox = test_stats.get("coco_eval_bbox", [0.0] * 12)

                coco_str = ",".join([f"{v :.4f}" for v in test_bbox])

                epoch_time_file = self.output_dir / "epoch_metrics.csv"
                if not epoch_time_file.exists():
                    with epoch_time_file.open("w") as f_et:

                        f_et.write(
                            "epoch,epoch_seconds,epoch_hms,cumulative_seconds,cumulative_hms,"
                            "train_loss,lr,"
                            "AP_0.50_0.95_all,AP_0.50_all,AP_0.75_all,"
                            "AP_0.50_0.95_small,AP_0.50_0.95_medium,AP_0.50_0.95_large,"
                            "AR_maxDet1_all,AR_maxDet10_all,AR_maxDet100_all,"
                            "AR_maxDet100_small,AR_maxDet100_medium,AR_maxDet100_large,"
                            "AP50_area_all,AP50_area_small,AP50_area_medium,AP50_area_large\n"
                        )
                with epoch_time_file.open("a") as f_et:
                    ap50_str = (
                        f"{ap50_area ['ap50_all']:.4f},"
                        f"{ap50_area ['ap50_small']:.4f},"
                        f"{ap50_area ['ap50_medium']:.4f},"
                        f"{ap50_area ['ap50_large']:.4f}"
                    )
                    f_et.write(
                        f"{epoch },{epoch_time :.6f},{epoch_time_hms },"
                        f"{cumulative_seconds :.6f},{cumulative_hms },"
                        f"{total_loss :.4f},{lr :.6e},{coco_str },{ap50_str }\n"
                    )

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch :03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = _format_hms(total_time)
        print(f"Training time: {total_time_str }")

        if self.output_dir and dist_utils.is_main_process():
            _generate_plots(self.output_dir)

    def val(
        self,
    ):
        self.eval()

        module = self.ema.module if self.ema else self.model

        eval_print_freq = getattr(self.cfg, "eval_print_freq", 10)
        if eval_print_freq is None:
            eval_print_freq = 10

        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            print_freq=eval_print_freq,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )
        return

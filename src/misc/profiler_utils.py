import re
import torch
import torch.nn as nn
from torch import Tensor

from typing import List, Optional, Tuple


def estimate_forward_flops(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
) -> Optional[float]:

    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x = torch.randn(*input_shape, device=device, dtype=dtype)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    wait, warmup, active, repeat = 0, 1, 1, 1
    skip_first = 0
    n_step = skip_first + (wait + warmup + active) * repeat

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            skip_first=skip_first,
        ),
        with_flops=True,
    ) as prof:
        for _ in range(n_step):
            with torch.no_grad():
                _ = model(x)
            prof.step()

    statistics = prof.key_averages()
    num_flops = sum(e.flops for e in statistics if e.flops is not None and e.flops > 0)
    if active:
        num_flops = num_flops / active

    if was_training:
        model.train()

    if num_flops <= 0:
        return None
    return float(num_flops)


def stats(
    model: nn.Module,
    data: Tensor = None,
    input_shape: List = [1, 3, 640, 640],
    device: str = "cpu",
    verbose=False,
) -> str:

    is_training = model.training

    model.train()
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    model.eval()
    model = model.to(device)

    if data is None:
        data = torch.rand(*input_shape, device=device)

    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    num_active = 2
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=num_active, repeat=1),
        with_flops=True,
    ) as p:
        for _ in range(5):
            _ = model(data)
            p.step()

    if is_training:
        model.train()

    info = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    num_flops = (
            sum(
                [float(v.strip()) for v in re.findall(r"(\d+\.?\d* *\n)", info)]
            )
            / num_active
    )

    if verbose:

        print(f"Total number of trainable parameters: {num_params }")
        print(f"Total number of flops: {int (num_flops )}M with {input_shape }")

    return {"n_parameters": num_params, "n_flops": num_flops, "info": info}

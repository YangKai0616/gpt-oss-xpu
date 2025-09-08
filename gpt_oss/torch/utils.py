import os
import torch
import torch.distributed as dist


def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed() -> torch.device:
    """Initialize the model for distributed inference."""
    # Initialize distributed inference
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()

    if xpu_available:
        backend = "xccl"
        device_type = "xpu"
    else:
        backend = "nccl"
        device_type = "cuda"

    if world_size > 1:
        dist.init_process_group(
            backend=backend, init_method="env://", world_size=world_size, rank=rank
        )

    if xpu_available:
        torch.xpu.set_device(rank)
    else:
        torch.cuda.set_device(rank)
    device = torch.device(f"{device_type}:{rank}")

    # Warm up backend to avoid first-time latency
    if world_size > 1:
        x = torch.ones(1, device=device)
        dist.all_reduce(x)
        if xpu_available:
            torch.xpu.synchronize(device)
        else:
            torch.cuda.synchronize(device)

    suppress_output(rank)
    return device

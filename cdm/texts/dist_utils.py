"""
Distributed Data Parallel utilities for multi-GPU training and sampling.

Usage:
    # Launch with torchrun:
    torchrun --nproc_per_node=4 -m cdm.texts.main --config-name cdm distributed.enabled=true

    # Or with a specific number of GPUs:
    torchrun --nproc_per_node=2 -m cdm.texts.main --config-name smc distributed.enabled=true
"""

import builtins
import datetime
import os

import torch
import torch.distributed as dist


def init_distributed_mode(cfg):
    """Initialize distributed training from torchrun environment variables.

    Sets cfg.distributed, cfg.rank, cfg.world_size, cfg.gpu, cfg.device.
    If distributed is not enabled in config or env vars are missing, falls back
    to single-GPU mode transparently.
    """
    enabled = getattr(cfg, "distributed", None)
    if enabled is None or not enabled:
        # Check for nested OmegaConf structure
        try:
            enabled = cfg.distributed.enabled
        except Exception:
            enabled = False
    else:
        enabled = cfg.distributed.enabled

    if not enabled:
        cfg.distributed_active = False
        cfg.rank = 0
        cfg.world_size = 1
        cfg.gpu = 0
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
        cfg.world_size = int(os.environ.get("SLURM_NTASKS", torch.cuda.device_count()))
    else:
        print("[!] distributed.enabled=true but no RANK/WORLD_SIZE env vars found. "
              "Launch with: torchrun --nproc_per_node=N -m cdm.texts.main ...")
        cfg.distributed_active = False
        cfg.rank = 0
        cfg.world_size = 1
        cfg.gpu = 0
        return

    cfg.distributed_active = True
    torch.cuda.set_device(cfg.gpu)
    cfg.device = f"cuda:{cfg.gpu}"

    dist.init_process_group(
        backend="nccl",
        world_size=cfg.world_size,
        rank=cfg.rank,
        timeout=datetime.timedelta(minutes=30)
    )
    dist.barrier()
    if str(getattr(cfg, "pos_resample_method", "")) == "ssp":
        import cdm.texts.resampling  # noqa: F401
    _setup_print_for_distributed(cfg.rank == 0)


def _setup_print_for_distributed(is_master):
    """Override builtins.print so only rank 0 prints (with timestamp)."""
    builtin_print = builtins.print

    def print_fn(*args, **kwargs):
        force = kwargs.pop("force", False)
        # Writes aimed at a file (the per-rank run logs) always go through, otherwise a
        # non-zero rank's traceback would be lost entirely.
        if is_master or force or kwargs.get("file") is not None:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            builtin_print(f"[{now}]", *args, **kwargs)

    builtins.print = print_fn


def is_dist_active():
    """Check if distributed training is currently active."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_active():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_active():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if is_dist_active():
        dist.barrier()


def cleanup():
    """Clean up distributed process group."""
    if is_dist_active():
        dist.destroy_process_group()


def all_reduce_mean(x):
    """Average a scalar value across all processes."""
    world_size = get_world_size()
    if world_size <= 1:
        return x
    x_tensor = torch.tensor(x, dtype=torch.float64).cuda()
    dist.all_reduce(x_tensor)
    x_tensor /= world_size
    return x_tensor.item()


def all_gather_list(data_list, world_size=None):
    """Gather a list from all ranks into a single combined list on all ranks.

    Each rank contributes ``data_list`` (a Python list). Returns the
    concatenation of all ranks' lists (ordered by rank). Delegates
    serialization + padded all_gather to
    ``torch.distributed.all_gather_object`` (PyTorch ≥ 1.8).
    """
    if world_size is None:
        world_size = get_world_size()
    if world_size <= 1:
        return list(data_list)

    gathered = [None] * world_size
    dist.all_gather_object(gathered, data_list)

    result = []
    for sublist in gathered:
        result.extend(sublist)
    return result


def all_gather_tensor(tensor):
    """Gather a tensor from all ranks, concatenated along dim 0."""
    world_size = get_world_size()
    if world_size <= 1:
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def global_softmax(local_log_w, dim=0, return_ess=False):
    """Softmax over the concatenation of `local_log_w` across all DDP ranks.

    Returns the per-rank slice of the global softmax. If `return_ess=True`,
    also returns the GLOBAL normalized ESS (= 1 / (N_global * sum(w^2))).
    Computing ESS reuses the already-gathered tensor — no extra
    communication beyond what global_softmax already does.

    Falls back to plain local softmax (and local ESS) when DDP is inactive.
    Currently only `dim=0` supported (because all_gather_tensor concats on dim 0).
    """
    if not is_dist_active() or get_world_size() == 1:
        w = torch.softmax(local_log_w, dim=dim)
        if return_ess:
            n = local_log_w.shape[dim]
            ess = (1.0 / (n * w.float().pow(2).sum().clamp(min=1e-12))).item()
            return w, ess
        return w

    if dim != 0:
        raise NotImplementedError("global_softmax currently only supports dim=0; extend if multi-dim support is needed")

    rank = get_rank()
    local_size = local_log_w.shape[0]

    global_log_w = all_gather_tensor(local_log_w.detach().contiguous())
    global_w = torch.softmax(global_log_w, dim=0)

    start = rank * local_size
    local_slice = global_w[start : start + local_size]

    if return_ess:
        n_global = global_w.shape[0]
        ess = (1.0 / (n_global * global_w.float().pow(2).sum().clamp(min=1e-12))).item()
        return local_slice, ess
    return local_slice


def all_reduce_gradients(parameters):
    """Average gradients across all processes (manual DDP alternative).

    Use this when the module is called multiple times per forward pass
    (e.g., chunked processing), which is incompatible with nn.parallel.DDP.
    Call after loss.backward() and before optimizer.step().
    """
    if not is_dist_active():
        return
    world_size = get_world_size()
    for param in parameters:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

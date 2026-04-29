# Copyright 2026 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for PyTorch DistributedDataParallel (DDP) training.

When launched with ``torchrun`` (or any process manager that exports the
standard ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` environment variables),
:func:`init_distributed` initializes a process group so that the rest of the
framework can transparently switch from single-process / ``DataParallel``
behaviour to true DDP.

If those env vars are not set, :func:`init_distributed` is a no-op and every
helper here returns the single-process defaults (rank=0, world_size=1).
"""
import os
from typing import Optional, Union

import torch
import torch.distributed as dist

__all__ = [
    "init_distributed",
    "cleanup_distributed",
    "is_distributed",
    "get_rank",
    "get_local_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "all_reduce_mean",
    "broadcast_object",
]


def _env_set() -> bool:
    """Return True iff torchrun-style env vars are present."""
    return all(v in os.environ for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def init_distributed(backend: Optional[str] = None, timeout_seconds: int = 1800) -> bool:
    """Initialize the default ``torch.distributed`` process group from env vars.

    Reads ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` (and ``MASTER_ADDR`` /
    ``MASTER_PORT``) which ``torchrun`` populates automatically. If these are
    not present this function does nothing and returns False, so it is safe to
    call unconditionally at the top of a training script.

    Args:
        backend: Distributed backend. Defaults to ``nccl`` when CUDA is
            available, otherwise ``gloo``.
        timeout_seconds: Process-group timeout.

    Returns:
        True iff a distributed process group was initialized.
    """
    if dist.is_available() and dist.is_initialized():
        return True
    if not _env_set() or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    import datetime
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=timeout_seconds))
    return True


def cleanup_distributed() -> None:
    """Destroy the default process group if one was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """Return True iff a torch.distributed process group is active."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank of the current process, or 0 if not distributed."""
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0")) if _env_set() else 0


def get_local_rank() -> int:
    """Local (per-node) rank of the current process, or 0 if not distributed."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def get_world_size() -> int:
    """World size, or 1 if not distributed."""
    if is_distributed():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1")) if _env_set() else 1


def is_main_process() -> bool:
    """True on rank 0 (or when running non-distributed)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes. No-op if not distributed."""
    if is_distributed():
        dist.barrier()


def all_reduce_mean(value: Union[float, int, torch.Tensor],
                    device: Optional[torch.device] = None) -> Union[float, torch.Tensor]:
    """Average a scalar or tensor across all ranks.

    Returns the input unchanged when not distributed. For Python scalars the
    result is returned as a Python float; for tensors a tensor is returned on
    the same device as the input.
    """
    if not is_distributed():
        return value
    backend = dist.get_backend()
    is_scalar = not isinstance(value, torch.Tensor)
    if is_scalar:
        if device is None:
            # NCCL requires CUDA tensors; gloo can use CPU regardless of CUDA availability.
            if backend == "nccl" and torch.cuda.is_available():
                device = torch.device(f"cuda:{get_local_rank()}")
            else:
                device = torch.device("cpu")
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    else:
        tensor = value.detach().clone()
        if backend == "nccl" and tensor.device.type != "cuda" and torch.cuda.is_available():
            tensor = tensor.to(f"cuda:{get_local_rank()}")
        elif backend == "gloo" and tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    if is_scalar:
        return tensor.item()
    return tensor


def broadcast_object(obj, src: int = 0):
    """Broadcast a Python object from rank ``src`` to all ranks."""
    if not is_distributed():
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]

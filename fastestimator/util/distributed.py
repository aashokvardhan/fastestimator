# Copyright 2024 The FastEstimator Authors. All Rights Reserved.
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
"""Utilities for distributed training with multi-GPU support."""
import os
from typing import Optional

import torch
import torch.distributed as dist


def init_distributed_mode(backend: str = 'nccl', init_method: str = 'env://') -> bool:
    """Initialize distributed training mode.

    This function sets up distributed training using PyTorch's distributed package.
    It automatically detects the distributed environment variables and initializes
    the process group. Also enables cudnn.benchmark for improved GPU performance
    with fixed-size inputs.

    Args:
        backend: Backend to use for distributed training. Options are:
            - 'nccl': NVIDIA NCCL (recommended for GPU training)
            - 'gloo': Gloo backend (works on CPU and GPU)
            - 'mpi': MPI backend
        init_method: URL specifying how to initialize the process group.
            Default 'env://' uses environment variables.

    Returns:
        True if distributed mode was successfully initialized, False otherwise.

    Example:
        >>> # Set environment variables before running (typically done by launcher):
        >>> # export MASTER_ADDR=localhost
        >>> # export MASTER_PORT=12355
        >>> # export WORLD_SIZE=4
        >>> # export RANK=0
        >>> if init_distributed_mode():
        >>>     print("Distributed training initialized")
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', rank % torch.cuda.device_count()))
        # Export environment variables so that other FE utilities (e.g. get_device) can find them
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
    else:
        print('Distributed training environment variables not set. Running in single-GPU/CPU mode.')
        return False

    # Set device
    if backend == 'nccl' and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Enable cudnn benchmark for better performance with fixed-size inputs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Initialize process group
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)

    # Synchronize all processes
    dist.barrier()

    # Suppress printing on non-main processes to reduce log clutter
    setup_for_distributed(rank == 0)

    return True


def cleanup_distributed():
    """Clean up distributed training resources.

    This should be called at the end of training to properly clean up
    distributed resources.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, world_size: int, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduce tensor across all processes in distributed training.

    Args:
        tensor: Input tensor to reduce.
        world_size: Total number of processes.
        op: Reduction operation (SUM, AVG, MIN, MAX, etc.).

    Returns:
        Reduced tensor.
    """
    if not dist.is_initialized() or world_size == 1:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=op)

    if op == dist.ReduceOp.SUM:
        rt = rt / world_size

    return rt


def gather_tensor(tensor: torch.Tensor, dst: int = 0) -> Optional[list]:
    """Gather tensors from all processes to a destination process.

    Args:
        tensor: Input tensor to gather.
        dst: Destination rank where tensors will be gathered.

    Returns:
        List of tensors if current process is dst, None otherwise.
    """
    if not dist.is_initialized():
        return [tensor]

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == dst:
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gather_list=tensor_list, dst=dst)
        return tensor_list
    else:
        dist.gather(tensor, dst=dst)
        return None


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source process to all other processes.

    Args:
        tensor: Tensor to broadcast (only matters on src process).
        src: Source rank to broadcast from.

    Returns:
        The broadcasted tensor.
    """
    if not dist.is_initialized():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def synchronize():
    """Synchronize all processes in distributed training.

    This creates a barrier that all processes must reach before continuing.
    Useful for ensuring all processes are at the same point in execution.
    """
    if dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0).

    Returns:
        True if main process or not in distributed mode, False otherwise.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def setup_for_distributed(is_master: bool):
    """Disable printing for non-main processes.

    This helps reduce clutter in logs when running distributed training.

    Args:
        is_master: Whether current process is the main process.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

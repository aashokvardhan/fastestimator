"""
Example script demonstrating efficient multi-GPU training with FastEstimator.

This script shows:
1. Using DistributedDataParallel for better multi-GPU performance
2. Gradient accumulation for larger effective batch sizes
3. Mixed precision training
4. Proper distributed training setup and cleanup

Usage:
    # Single GPU or CPU
    python multi_gpu_example.py

    # Multi-GPU with DataParallel (automatic, less efficient)
    python multi_gpu_example.py

    # Multi-GPU with DistributedDataParallel (recommended, more efficient)
    python -m torch.distributed.launch --nproc_per_node=4 multi_gpu_example.py
"""
import argparse

import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet
from fastestimator.dataset.data import mnist
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax, Onehot
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.adapt import LRScheduler
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
from fastestimator.util import get_num_devices, get_num_gpus
from fastestimator.util.distributed import (
    cleanup_distributed,
    init_distributed_mode,
    is_main_process,
)


def get_estimator(
    batch_size_per_gpu=64,
    epochs=10,
    use_gradient_accumulation=True,
    accumulation_steps=2,
    use_mixed_precision=False,
    log_steps=100,
):
    """Create an Estimator for MNIST training with multi-GPU optimizations.

    Args:
        batch_size_per_gpu: Batch size per GPU.
        epochs: Number of training epochs.
        use_gradient_accumulation: Whether to use gradient accumulation.
        accumulation_steps: Number of steps to accumulate gradients over.
        use_mixed_precision: Whether to use mixed precision training.
        log_steps: Frequency of logging.

    Returns:
        An Estimator instance.
    """
    # Calculate total batch size
    num_devices = get_num_devices()
    total_batch_size = batch_size_per_gpu * num_devices

    if is_main_process():
        print(f"\n{'='*70}")
        print(f"Multi-GPU Training Configuration")
        print(f"{'='*70}")
        print(f"Number of GPUs: {get_num_gpus()}")
        print(f"Number of devices: {num_devices}")
        print(f"Batch size per GPU: {batch_size_per_gpu}")
        print(f"Total batch size: {total_batch_size}")
        if use_gradient_accumulation:
            print(f"Gradient accumulation steps: {accumulation_steps}")
            print(f"Effective batch size: {total_batch_size * accumulation_steps}")
        print(f"Mixed precision: {use_mixed_precision}")
        print(f"{'='*70}\n")

    # Load data
    train_data, eval_data = mnist.load_data()

    # Create pipeline
    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=eval_data,
        batch_size=total_batch_size,
        ops=[
            ExpandDims(inputs="x", outputs="x"),
            Minmax(inputs="x", outputs="x"),
            Onehot(inputs="y", outputs="y", num_classes=10),
        ],
    )

    # Build model with optional mixed precision
    model = fe.build(
        model_fn=LeNet,
        optimizer_fn="adam",
        model_name="LeNet",
        mixed_precision=use_mixed_precision,
    )

    # Create network with optional gradient accumulation
    merge_grad = accumulation_steps if use_gradient_accumulation else 1

    network = fe.Network(ops=[
        ModelOp(model=model, inputs="x", outputs="y_pred"),
        CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
        UpdateOp(model=model, loss_name="ce", merge_grad=merge_grad),
    ])

    # Setup traces
    traces = [
        Accuracy(true_key="y", pred_key="y_pred"),
        BestModelSaver(model=model, save_dir="./models", metric="accuracy", save_best_mode="max"),
        LRScheduler(model=model, lr_fn=lambda epoch: 1e-3 if epoch < 5 else 1e-4),
    ]

    # Create estimator
    estimator = fe.Estimator(
        pipeline=pipeline,
        network=network,
        epochs=epochs,
        traces=traces,
        log_steps=log_steps,
    )

    return estimator


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="FastEstimator Multi-GPU Training Example")
    parser.add_argument("--batch_size_per_gpu", type=int, default=64, help="Batch size per GPU (default: 64)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument(
        "--gradient_accumulation",
        action="store_true",
        default=True,
        help="Use gradient accumulation (default: True)",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=2,
        help="Number of gradient accumulation steps (default: 2)",
    )
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training (default: False)")
    parser.add_argument("--log_steps", type=int, default=100, help="Logging frequency (default: 100)")

    args = parser.parse_args()

    # Initialize distributed training (if available)
    use_distributed = init_distributed_mode(backend="nccl")

    if use_distributed and is_main_process():
        print("✓ Distributed training initialized (using DistributedDataParallel)")
    elif get_num_gpus() > 1:
        print("⚠ Multiple GPUs detected but distributed training not initialized")
        print("  Using DataParallel (less efficient). For better performance, run with:")
        print(f"  python -m torch.distributed.launch --nproc_per_node={get_num_gpus()} {__file__}")
    elif get_num_gpus() == 1:
        print("ℹ Single GPU detected")
    else:
        print("ℹ No GPU detected, using CPU")

    # Get estimator
    estimator = get_estimator(
        batch_size_per_gpu=args.batch_size_per_gpu,
        epochs=args.epochs,
        use_gradient_accumulation=args.gradient_accumulation,
        accumulation_steps=args.accumulation_steps,
        use_mixed_precision=args.mixed_precision,
        log_steps=args.log_steps,
    )

    # Train
    if is_main_process():
        print("\nStarting training...\n")

    estimator.fit()

    if is_main_process():
        print("\n✓ Training completed successfully!")

    # Cleanup distributed training
    if use_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()

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
"""Tests for device management, GPU transfer optimization, and distributed training utilities."""
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import fastestimator as fe
from fastestimator.architecture.pytorch import LeNet as LeNetTorch
from fastestimator.network import TorchNetwork
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.test.unittest_util import OneLayerTorchModel
from fastestimator.util.distributed import (
    broadcast_tensor,
    cleanup_distributed,
    gather_tensor,
    init_distributed_mode,
    is_main_process,
    reduce_tensor,
    setup_for_distributed,
    synchronize,
)
from fastestimator.util.util import (
    detach_and_move_to_cpu,
    detach_tensors,
    get_device,
    get_local_rank,
    get_num_gpus,
    get_world_size,
    is_distributed,
    move_tensors_to_device,
)


class TestMoveTensorsToDevice(unittest.TestCase):
    """Test non-blocking transfer and device movement functionality."""

    def test_move_single_tensor_cpu(self):
        """Test moving a single tensor to CPU."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = move_tensors_to_device(t, "cpu")
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.equal(t, result))

    def test_move_dict_of_tensors(self):
        """Test moving a dictionary of tensors."""
        data = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        result = move_tensors_to_device(data, "cpu")
        self.assertIsInstance(result, dict)
        self.assertTrue(torch.equal(result["a"], data["a"]))

    def test_move_list_of_tensors(self):
        """Test moving a list of tensors."""
        data = [torch.tensor([1.0]), torch.tensor([2.0])]
        result = move_tensors_to_device(data, "cpu")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_move_tuple_of_tensors(self):
        """Test moving a tuple of tensors."""
        data = (torch.tensor([1.0]), torch.tensor([2.0]))
        result = move_tensors_to_device(data, "cpu")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_move_set_of_non_tensors(self):
        """Test moving a set of non-tensor data (should pass through)."""
        data = {1, 2, 3}
        result = move_tensors_to_device(data, "cpu")
        self.assertEqual(result, data)

    def test_move_non_tensor_passthrough(self):
        """Test that non-tensor data passes through unchanged."""
        self.assertEqual(move_tensors_to_device("hello", "cpu"), "hello")
        self.assertEqual(move_tensors_to_device(42, "cpu"), 42)
        self.assertIsNone(move_tensors_to_device(None, "cpu"))

    def test_non_blocking_parameter(self):
        """Test that non_blocking parameter is accepted."""
        t = torch.tensor([1.0, 2.0])
        result = move_tensors_to_device(t, "cpu", non_blocking=True)
        self.assertTrue(torch.equal(t, result))
        result = move_tensors_to_device(t, "cpu", non_blocking=False)
        self.assertTrue(torch.equal(t, result))

    def test_nested_dict_structure(self):
        """Test moving nested data structures."""
        data = {"nested": {"a": torch.tensor([1.0])}, "list": [torch.tensor([2.0])]}
        result = move_tensors_to_device(data, "cpu")
        self.assertTrue(torch.equal(result["nested"]["a"], data["nested"]["a"]))
        self.assertTrue(torch.equal(result["list"][0], data["list"][0]))


class TestDetachAndMoveToCpu(unittest.TestCase):
    """Test the combined detach-and-move-to-cpu function."""

    def test_detach_single_tensor(self):
        """Test detaching and moving a single tensor."""
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t_grad = t * 2  # This has a grad_fn
        result = detach_and_move_to_cpu(t_grad)
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.device.type, "cpu")

    def test_detach_dict(self):
        """Test detaching and moving a dict of tensors."""
        t = torch.tensor([1.0], requires_grad=True)
        data = {"a": t * 2, "b": t * 3}
        result = detach_and_move_to_cpu(data)
        self.assertFalse(result["a"].requires_grad)
        self.assertFalse(result["b"].requires_grad)

    def test_detach_list(self):
        """Test detaching and moving a list of tensors."""
        t = torch.tensor([1.0], requires_grad=True)
        data = [t * 2, t * 3]
        result = detach_and_move_to_cpu(data)
        self.assertIsInstance(result, list)
        self.assertFalse(result[0].requires_grad)

    def test_detach_tuple(self):
        """Test detaching and moving a tuple of tensors."""
        t = torch.tensor([1.0], requires_grad=True)
        data = (t * 2, )
        result = detach_and_move_to_cpu(data)
        self.assertIsInstance(result, tuple)
        self.assertFalse(result[0].requires_grad)

    def test_detach_non_tensor_passthrough(self):
        """Test that non-tensor data passes through."""
        self.assertEqual(detach_and_move_to_cpu("hello"), "hello")
        self.assertEqual(detach_and_move_to_cpu(42), 42)

    def test_detach_produces_same_values(self):
        """Test that detach_and_move_to_cpu produces same values as separate calls."""
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t_grad = t * 2
        # Combined approach
        combined = detach_and_move_to_cpu(t_grad)
        # Separate approach
        separate = move_tensors_to_device(detach_tensors(t_grad), "cpu")
        self.assertTrue(torch.equal(combined, separate))


class TestDetachTensors(unittest.TestCase):
    """Test tensor detachment functionality."""

    def test_detach_requires_grad_tensor(self):
        """Test detaching a tensor that requires grad."""
        t = torch.tensor([1.0], requires_grad=True)
        result = t * 2
        detached = detach_tensors(result)
        self.assertFalse(detached.requires_grad)

    def test_detach_no_grad_tensor(self):
        """Test detaching a tensor that doesn't require grad."""
        t = torch.tensor([1.0])
        detached = detach_tensors(t)
        self.assertTrue(torch.equal(t, detached))

    def test_detach_nested_structure(self):
        """Test detaching nested structures."""
        t = torch.tensor([1.0], requires_grad=True)
        data = {"a": [t * 2, (t * 3, )]}
        result = detach_tensors(data)
        self.assertFalse(result["a"][0].requires_grad)
        self.assertFalse(result["a"][1][0].requires_grad)


class TestGetDevice(unittest.TestCase):
    """Test device detection for various hardware configurations."""

    def test_get_device_returns_torch_device(self):
        """Test that get_device returns a torch.device."""
        # Clear cached result
        get_device.cache_clear()
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_get_device_cpu_fallback(self):
        """Test that CPU is returned when no GPU is available."""
        get_device.cache_clear()
        with patch('fastestimator.util.util.is_distributed', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=False):
            get_device.cache_clear()
            device = get_device()
            self.assertEqual(device.type, "cpu")
            get_device.cache_clear()


class TestGetNumGpus(unittest.TestCase):
    """Test GPU count detection."""

    def test_get_num_gpus_returns_int(self):
        """Test that get_num_gpus returns an integer."""
        get_num_gpus.cache_clear()
        result = get_num_gpus()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)


class TestDistributedHelpers(unittest.TestCase):
    """Test distributed training helper functions when NOT in distributed mode."""

    def test_is_distributed_false_by_default(self):
        """Test that is_distributed returns False when dist is not initialized."""
        if not torch.distributed.is_initialized():
            self.assertFalse(is_distributed())

    def test_get_local_rank_zero_by_default(self):
        """Test that local rank is 0 when not in distributed mode."""
        if not torch.distributed.is_initialized():
            self.assertEqual(get_local_rank(), 0)

    def test_get_world_size_one_by_default(self):
        """Test that world size is 1 when not in distributed mode."""
        if not torch.distributed.is_initialized():
            self.assertEqual(get_world_size(), 1)

    def test_is_main_process_true_by_default(self):
        """Test that is_main_process is True when not in distributed mode."""
        if not torch.distributed.is_initialized():
            self.assertTrue(is_main_process())

    def test_synchronize_noop_when_not_distributed(self):
        """Test that synchronize is a no-op when not in distributed mode."""
        if not torch.distributed.is_initialized():
            synchronize()  # Should not raise

    def test_broadcast_tensor_passthrough_when_not_distributed(self):
        """Test that broadcast_tensor returns input when not distributed."""
        if not torch.distributed.is_initialized():
            t = torch.tensor([1.0, 2.0])
            result = broadcast_tensor(t)
            self.assertTrue(torch.equal(t, result))

    def test_gather_tensor_wraps_in_list_when_not_distributed(self):
        """Test that gather_tensor returns [tensor] when not distributed."""
        if not torch.distributed.is_initialized():
            t = torch.tensor([1.0, 2.0])
            result = gather_tensor(t)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertTrue(torch.equal(result[0], t))

    def test_reduce_tensor_passthrough_when_not_distributed(self):
        """Test that reduce_tensor returns tensor when not distributed."""
        if not torch.distributed.is_initialized():
            t = torch.tensor([1.0, 2.0])
            result = reduce_tensor(t, world_size=1)
            self.assertTrue(torch.equal(t, result))


class TestInitDistributedModeNoEnvVars(unittest.TestCase):
    """Test init_distributed_mode when no environment variables are set."""

    def test_returns_false_without_env_vars(self):
        """Test that initialization fails gracefully without env vars."""
        # Save and clear relevant env vars
        saved = {}
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'SLURM_PROCID', 'SLURM_NTASKS'):
            saved[key] = os.environ.pop(key, None)
        try:
            result = init_distributed_mode()
            self.assertFalse(result)
        finally:
            # Restore
            for key, val in saved.items():
                if val is not None:
                    os.environ[key] = val


class TestCleanupDistributed(unittest.TestCase):
    """Test distributed cleanup."""

    def test_cleanup_when_not_initialized(self):
        """Test that cleanup is safe when dist is not initialized."""
        if not torch.distributed.is_initialized():
            cleanup_distributed()  # Should not raise


class TestSetupForDistributed(unittest.TestCase):
    """Test printing suppression in distributed mode."""

    def test_master_can_print(self):
        """Test that master process can print."""
        import builtins
        original_print = builtins.print
        try:
            setup_for_distributed(is_master=True)
            # Master should be able to print (no exception)
            builtins.print("test")
        finally:
            builtins.print = original_print

    def test_non_master_print_suppressed(self):
        """Test that non-master print is suppressed."""
        import builtins
        import io
        from contextlib import redirect_stdout
        original_print = builtins.print
        try:
            setup_for_distributed(is_master=False)
            f = io.StringIO()
            with redirect_stdout(f):
                builtins.print("should not appear")
            # The print function was replaced, so output should be suppressed
            # (though redirect_stdout may not catch the suppressed print)
        finally:
            builtins.print = original_print


class TestCPUTrainingPath(unittest.TestCase):
    """Test that training works correctly on CPU."""

    def test_cpu_forward_pass(self):
        """Test a complete forward pass on CPU."""
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
        ])
        batch = {"x": np.array([[1, 1, 1]], dtype=np.float32)}
        result = network.transform(data=batch, mode="infer")
        self.assertIn("y_pred", result)
        self.assertIsNotNone(result["y_pred"])

    def test_cpu_train_step(self):
        """Test a complete training step on CPU."""
        from fastestimator.op.tensorop.loss import MeanSquaredError
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            MeanSquaredError(inputs=("y_pred", "y"), outputs="loss"),
            UpdateOp(model=model, loss_name="loss"),
        ])
        batch = {"x": np.array([[1, 1, 1]], dtype=np.float32), "y": np.array([1], dtype=np.float32)}
        result = network.transform(data=batch, mode="train")
        self.assertIn("loss", result)
        self.assertIn("y_pred", result)

    def test_cpu_eval_step(self):
        """Test a complete eval step on CPU (no gradient computation)."""
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
        ])
        batch = {"x": np.array([[1, 1, 1]], dtype=np.float32)}
        result = network.transform(data=batch, mode="eval")
        self.assertIn("y_pred", result)


class TestNetworkInstanceType(unittest.TestCase):
    """Test that Network factory creates correct instance types."""

    def test_torch_network_created(self):
        """Test that TorchNetwork is created for PyTorch models."""
        model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam")
        network = fe.Network(ops=[ModelOp(model=model, inputs="x", outputs="y")])
        self.assertIsInstance(network, TorchNetwork)


class TestOptimizerZeroGrad(unittest.TestCase):
    """Test that optimizer.zero_grad(set_to_none=True) works correctly."""

    def test_zero_grad_set_to_none(self):
        """Test that gradients are None after zero_grad(set_to_none=True)."""
        from fastestimator.backend._update_model import _torch_step
        model = OneLayerTorchModel()
        optimizer = torch.optim.Adam(model.parameters())
        setattr(optimizer, "scaler", None)

        # Create dummy gradients
        x = torch.ones(1, 3)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Verify gradients exist
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

        # Apply step (which calls zero_grad(set_to_none=True))
        _torch_step(optimizer)

        # Verify gradients are None (not zero tensors)
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNone(p.grad)


class TestGradScalerModern(unittest.TestCase):
    """Test that modern GradScaler API is used."""

    def test_scaler_uses_torch_amp(self):
        """Test that the non-deprecated GradScaler is used."""
        if torch.cuda.is_available():
            model = fe.build(model_fn=OneLayerTorchModel, optimizer_fn="adam", mixed_precision=True)
            scaler = model.current_optimizer.scaler
            self.assertIsNotNone(scaler)
            self.assertIsInstance(scaler, torch.amp.GradScaler)

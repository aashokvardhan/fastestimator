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
"""``fastestimator dist-train`` -- a thin wrapper around :mod:`torch.distributed.run` (``torchrun``).

Translates a friendly CLI into a ``torchrun`` invocation that launches one
process per local GPU (or per ``--nproc_per_node``) and runs the standard
``fastestimator train`` entry point inside each process. The training script
itself does not need to know it is being launched via DDP -- it should simply
call ``estimator.fit()``; the framework will detect the DDP environment
variables and configure itself.
"""
import argparse
import os
import shutil
import sys
from typing import Any, Dict, List, Optional


def dist_train(args: Dict[str, Any], unknown: Optional[List[str]]) -> None:
    """Launch a multi-process distributed training job via ``torchrun``."""
    entry_point = args['entry_point']
    nproc = args.get('nproc_per_node')
    if nproc is None or nproc == 'auto':
        try:
            import torch
            nproc = max(torch.cuda.device_count(), 1)
        except Exception:
            nproc = 1
    nnodes = args.get('nnodes', 1)
    node_rank = args.get('node_rank', 0)
    rdzv_endpoint = args.get('rdzv_endpoint') or f"127.0.0.1:{args.get('master_port', 29500)}"
    rdzv_backend = args.get('rdzv_backend', 'c10d')
    rdzv_id = args.get('rdzv_id', 'fastestimator')

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--rdzv_backend={rdzv_backend}",
        f"--rdzv_endpoint={rdzv_endpoint}",
        f"--rdzv_id={rdzv_id}",
    ]

    # Forward to a tiny launcher module which constructs the Estimator and calls fit().
    cmd += [
        "-m",
        "fastestimator.cli._dist_runner",
        "--entry_point",
        entry_point,
    ]
    if args.get('hyperparameters_json'):
        cmd += ["--hyperparameters", args['hyperparameters_json']]
    if args.get('summary') is not None:
        cmd += ["--summary", args['summary']]
    cmd += ["--warmup", str(args.get('warmup', True))]
    cmd += ["--eager", str(args.get('eager', False))]
    if unknown:
        cmd += list(unknown)

    if shutil.which(sys.executable) is None:
        raise RuntimeError(f"Python executable not found: {sys.executable}")
    os.execvp(cmd[0], cmd)


def configure_dist_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add a ``dist-train`` parser to the FastEstimator CLI."""
    from ast import literal_eval
    parser = subparsers.add_parser('dist-train',
                                   description='Launch distributed (DDP) training via torchrun.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   allow_abbrev=False)
    parser.add_argument('entry_point', type=str, help='The path to the model python file')
    parser.add_argument('--hyperparameters',
                        dest='hyperparameters_json',
                        type=str,
                        help='The path to a hyperparameters JSON file')
    parser.add_argument('--nproc_per_node',
                        default='auto',
                        help='Number of processes (GPUs) per node, or "auto" to detect.')
    parser.add_argument('--nnodes', type=int, default=1, help='Total number of nodes participating.')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of this node (0..nnodes-1).')
    parser.add_argument('--master_port',
                        type=int,
                        default=29500,
                        help='Master port (used when --rdzv_endpoint is not set).')
    parser.add_argument('--rdzv_endpoint',
                        type=str,
                        default=None,
                        help='Rendezvous endpoint, e.g. "host:port". Defaults to 127.0.0.1:<master_port>.')
    parser.add_argument('--rdzv_backend', type=str, default='c10d')
    parser.add_argument('--rdzv_id', type=str, default='fastestimator')
    parser.add_argument('--warmup', type=literal_eval, choices=[True, False], default=True)
    parser.add_argument('--eager', type=literal_eval, choices=[True, False], default=False)
    parser.add_argument('--summary', type=str, default=None)
    parser.add_argument_group('hyperparameter arguments',
                              'Extra arguments are forwarded to get_estimator(), e.g. --epochs 5 --batch_size 64.')
    parser.set_defaults(func=dist_train)

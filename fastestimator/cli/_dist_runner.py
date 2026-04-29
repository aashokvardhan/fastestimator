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
"""Per-rank entry point invoked by ``fastestimator dist-train`` via torchrun.

This module is launched once per process by :mod:`torch.distributed.run`. It
parses the same arguments that ``fastestimator train`` accepts, forwards
unknown arguments to ``get_estimator``, and calls ``estimator.fit``.

Distributed initialization itself happens inside ``Estimator.fit`` -- we just
need to run that on each rank.
"""
import argparse
import sys
from ast import literal_eval

from fastestimator.cli.train import _get_estimator


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument('--entry_point', type=str, required=True)
    parser.add_argument('--hyperparameters', dest='hyperparameters_json', type=str, default=None)
    parser.add_argument('--summary', type=str, default=None)
    parser.add_argument('--warmup', type=literal_eval, choices=[True, False], default=True)
    parser.add_argument('--eager', type=literal_eval, choices=[True, False], default=False)
    args, unknown = parser.parse_known_args(sys.argv[1:])
    estimator = _get_estimator(vars(args), unknown)
    estimator.fit(warmup=args.warmup, eager=args.eager, summary=args.summary)


if __name__ == "__main__":
    main()

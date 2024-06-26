# Copyright 2023 The FastEstimator Authors. All Rights Reserved.
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
from typing import Iterable, Tuple, Union

from albumentations.augmentations.blur.transforms import ZoomBlur as ZoomBlurAlb

from fastestimator.op.numpyop.univariate.univariate import ImageOnlyAlbumentation
from fastestimator.util.traceability_util import traceable


@traceable()
class ZoomBlur(ImageOnlyAlbumentation):
    """Apply zoom blur transform.

    Args:
        max_factor: range for max factor for blurring. If max_factor is a single float, the range will be (1, limit).
            All max_factor values should be larger than 1. If max_factor is a tuple, it represents the
            (min, max) range for the factor.
        step_factor: Step size for zoom. All step_factor values should be positive. If single float will be used as
            step parameter for np.arange. If tuple of float step_factor will be in range
            [step_factor[0], step_factor[1])
        inputs: Key(s) of images to be modified.
        outputs: Key(s) into which to write the modified images.
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
    """
    def __init__(self,
                 inputs: Union[str, Iterable[str]],
                 outputs: Union[str, Iterable[str]],
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 max_factor: Union[float, Tuple[float, float]] = (1,1.31),
                 step_factor: Union[float, Tuple[float, float]] = (0.01,0.03)
                 ):
        super().__init__(ZoomBlurAlb(max_factor=max_factor,
                                     step_factor=step_factor,
                                     always_apply=True),
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         ds_id=ds_id)

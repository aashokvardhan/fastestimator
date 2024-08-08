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
import random
from typing import Any, Iterable, List, Optional, Union

import numpy as np
from albumentations import BboxParams, KeypointParams
from albumentations.augmentations import CropNonEmptyMaskIfExists as CropNonEmptyMaskIfExistsAlb
from albumentations.core.types import NUM_MULTI_CHANNEL_DIMENSIONS

from fastestimator.op.numpyop.multivariate.multivariate import MultiVariateAlbumentation
from fastestimator.util.traceability_util import traceable


class CustomCropNonEmptyMaskIfExistsAlb(CropNonEmptyMaskIfExistsAlb):

    def update_params(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(np.copy(masks[0]))  # need copy as we perform in-place mod afterwards
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            msg = "Can not find mask for CropNonEmptyMaskIfExists"
            raise RuntimeError(msg)

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            mask = mask.sum(axis=-1) if mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(list(non_zero_yx))
            x_min = x - random.randint(0, self.width - 1)
            y_min = y - random.randint(0, self.height - 1)
            x_min = np.clip(x_min, 0, mask_width - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
        else:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height

        crop_coords = x_min, y_min, x_max, y_max

        params["crop_coords"] = crop_coords
        return params


@traceable()
class CropNonEmptyMaskIfExists(MultiVariateAlbumentation):
    """Crop an area with mask if mask is non-empty, otherwise crop randomly.

    Args:
        mode: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Op in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        image_in: The key of an image to be modified.
        mask_in: The key of a mask to be modified (with the same random factors as the image).
        masks_in: The list of mask keys to be modified (with the same random factors as the image).
        bbox_in: The key of a bounding box(es) to be modified (with the same random factors as the image).
        keypoints_in: The key of keypoints to be modified (with the same random factors as the image).
        image_out: The key to write the modified image (defaults to `image_in` if None).
        mask_out: The key to write the modified mask (defaults to `mask_in` if None).
        masks_out: The list of keys to write the modified masks (defaults to `masks_in` if None).
        bbox_out: The key to write the modified bounding box(es) (defaults to `bbox_in` if None).
        keypoints_out: The key to write the modified keypoints (defaults to `keypoints_in` if None).
        bbox_params: Parameters defining the type of bounding box ('coco', 'pascal_voc', 'albumentations' or 'yolo').
        keypoint_params: Parameters defining the type of keypoints ('xy', 'yx', 'xya', 'xys', 'xyas', 'xysa').
        height: Vertical size of crop in pixels.
        width: Horizontal size of crop in pixels.
        ignore_values: Values to ignore in mask, `0` values are always ignored (e.g. if background value is 5 set
            `ignore_values=[5]` to ignore).
        ignore_channels: Channels to ignore in mask (e.g. if background is a first channel set `ignore_channels=[0]` to
            ignore).

    Image types:
        uint8, float32
    """

    def __init__(self,
                 height: int,
                 width: int,
                 ignore_values: Optional[List[int]] = None,
                 ignore_channels: Optional[List[int]] = None,
                 mode: Union[None, str, Iterable[str]] = None,
                 ds_id: Union[None, str, Iterable[str]] = None,
                 image_in: Optional[str] = None,
                 mask_in: Optional[str] = None,
                 masks_in: Optional[Iterable[str]] = None,
                 bbox_in: Optional[str] = None,
                 keypoints_in: Optional[str] = None,
                 image_out: Optional[str] = None,
                 mask_out: Optional[str] = None,
                 masks_out: Optional[Iterable[str]] = None,
                 bbox_out: Optional[str] = None,
                 keypoints_out: Optional[str] = None,
                 bbox_params: Union[BboxParams, str, None] = None,
                 keypoint_params: Union[KeypointParams, str, None] = None):
        super().__init__(
            CustomCropNonEmptyMaskIfExistsAlb(height=height,
                                              width=width,
                                              ignore_values=ignore_values,
                                              ignore_channels=ignore_channels,
                                              always_apply=True),
            image_in=image_in,
            mask_in=mask_in,
            masks_in=masks_in,
            bbox_in=bbox_in,
            keypoints_in=keypoints_in,
            image_out=image_out,
            mask_out=mask_out,
            masks_out=masks_out,
            bbox_out=bbox_out,
            keypoints_out=keypoints_out,
            bbox_params=bbox_params,
            keypoint_params=keypoint_params,
            mode=mode,
            ds_id=ds_id)

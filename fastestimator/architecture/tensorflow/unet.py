# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
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
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model


class UNetEncoderBlock(Model):
    """
        A UNet encoder block.

        This class is intentionally not @traceable (models and layers are handled by a different process).

        Args:
            in_channels: How many channels enter the encoder.
            out_channels: How many channels leave the encoder.
    """
    def __init__(self, out_channels: int) -> None:
        super(UNetEncoderBlock, self).__init__(name='')
        self.config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}

        self.encoder_seq = tf.keras.Sequential(
            [Conv2D(out_channels, 3, **self.config), Conv2D(out_channels, 3, **self.config)])

        self.max_pooling = MaxPooling2D(pool_size=(2, 2))

    def call(self, input_tensor, training=False):
        conv_output = self.encoder_seq(input_tensor)
        max_pooling_output = self.max_pooling(conv_output)
        return conv_output, max_pooling_output


class UNetDecoderBlock(Model):
    """
        A UNet encoder block.

        This class is intentionally not @traceable (models and layers are handled by a different process).

        Args:
            mid_channels: How many channels are used for the decoder's intermediate layer.
            out_channels: How many channels leave the decoder.
    """
    def __init__(self, mid_channels: int, out_channels: int) -> None:
        super(UNetDecoderBlock, self).__init__(name='')
        config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
        up_config = {'size': (2, 2), 'interpolation': 'bilinear'}
        self.decoder_seq = tf.keras.Sequential([
            Conv2D(mid_channels, 3, **config),
            Conv2D(mid_channels, 3, **config),
            UpSampling2D(**up_config),
            Conv2D(out_channels, 3, **config)
        ])

    def call(self, input_tensor, training=False):
        return self.decoder_seq(input_tensor)


class UNet(Model):
    """
        A UNet implementation in Tensorflow.

        This class is intentionally not @traceable (models and layers are handled by a different process).

        Args:
            input_size: The size of the input tensor (height, width, channels).

        Raises:
            ValueError: Length of `input_size` is not 3.
            ValueError: `input_size`[0] or `input_size`[1] is not a multiple of 16.
    """
    def __init__(self, input_size: Tuple(int, int, int) = (128, 128, 1)) -> None:
        UNet._check_input_size(input_size)
        super(UNet, self).__init__(name='')
        config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}
        self.input_size = input_size
        inputs = Input(input_size)
        self.enc1 = UNetEncoderBlock(out_channels=64)
        self.enc2 = UNetEncoderBlock(out_channels=128)
        self.enc3 = UNetEncoderBlock(out_channels=256)
        self.enc4 = UNetEncoderBlock(out_channels=512)

        self.bottle_neck = UNetDecoderBlock(mid_channels=1024, out_channels=512)
        self.dec4 = UNetDecoderBlock(mid_channels=512, out_channels=256)
        self.dec3 = UNetDecoderBlock(mid_channels=256, out_channels=128)
        self.dec2 = UNetDecoderBlock(mid_channels=128, out_channels=64)

        self.dec1 = tf.keras.Sequential(
            [Conv2D(64, 3, **config), Conv2D(64, 3, **config), Conv2D(1, 1, activation='sigmoid')])
        self.call(inputs)

    def call(self, input_tensor, training=False):
        x1, x_e1 = self.enc1(input_tensor)
        x2, x_e2 = self.enc2(x_e1)
        x3, x_e3 = self.enc3(x_e2)
        x4, x_e4 = self.enc4(x_e3)

        x_bottle_neck = self.bottle_neck(x_e4)

        merge6 = concatenate([x_bottle_neck, x4], axis=-1)
        x_d4 = self.dec4(merge6)

        merge7 = concatenate([x_d4, x3], axis=-1)
        x_d3 = self.dec3(merge7)

        merge8 = concatenate([x_d3, x2], axis=-1)
        x_d2 = self.dec2(merge8)

        merge9 = concatenate([x_d2, x1], axis=-1)
        x_out = self.dec1(merge9)
        return x_out

    def _check_input_size(input_size):
        if len(input_size) != 3:
            raise ValueError("Length of `input_size` is not 3 (channel, height, width)")

        height, width, _ = input_size

        if height < 16 or not (height / 16.0).is_integer() or width < 16 or not (width / 16.0).is_integer():
            raise ValueError("Both height and width of input_size need to be multiples of 16 (16, 32, 48...)")

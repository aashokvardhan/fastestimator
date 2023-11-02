import math
from typing import Any, Dict

import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from medical_diffusion.loss.perceivers import LPIPS
from pytorch_msssim import ssim

import fastestimator as fe
from fastestimator.backend._reduce_mean import reduce_mean
from fastestimator.backend._reduce_sum import reduce_sum
from fastestimator.dataset.data.montgomery import load_data
from fastestimator.op.numpyop.multivariate import Resize
from fastestimator.op.numpyop.univariate import ChannelTranspose, Minmax, ReadImage
from fastestimator.op.tensorop import TensorOp
from fastestimator.op.tensorop.loss import L1_Loss
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver

torch.multiprocessing.set_sharing_strategy('file_system')


class TimeEmbbeding(nn.Module):
    def __init__(self, emb_dim=64, downscale_freq_shift=1, max_period=10000, flip_sin_to_cos=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.downscale_freq_shift = downscale_freq_shift
        self.max_period = max_period
        self.flip_sin_to_cos = flip_sin_to_cos
        self.pos_emb_dim = emb_dim // 4
        self.time_emb = nn.Sequential(nn.Linear(self.pos_emb_dim, self.emb_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.emb_dim, self.emb_dim))

    def get_sinusoidal_pos_emb(self, x):
        device = x.device
        half_dim = self.pos_emb_dim // 2
        emb = math.log(self.max_period) / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(-emb * torch.arange(half_dim, device=device))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.pos_emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, time):
        sinusoidal_emd = self.get_sinusoidal_pos_emb(time)
        return self.time_emb(sinusoidal_emd)


class GaussianNoiseScheduler(TensorOp):
    def __init__(
        self,
        inputs,
        outputs,
        mode,
        timesteps=1000,
        T=None,
        beta_start=0.0001,  # default 1e-4, stable-diffusion ~ 1e-3
        beta_end=0.02,
        betas=None,
    ):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.timesteps = timesteps
        self.T = timesteps if T is None else T

        self.timesteps_array = torch.linspace(0, self.T - 1, self.timesteps, dtype=torch.long)
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float64)**2
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.betas = betas.to(torch.float32)  # (0 , 1)
        self.alphas = alphas.to(torch.float32)
        self.alphas_cumprod = alphas_cumprod.to(torch.float32)

        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def sample(self, x_0):
        """Randomly sample t from [0,T] and return x_t and x_T based on x_0"""
        t = torch.randint(0, self.T, (x_0.shape[0], ), dtype=torch.long,
                          device=x_0.device)  # NOTE: High is exclusive, therefore [0, T-1]
        x_T = self.x_final(x_0)
        return self.estimate_x_t(x_0, t, x_T), x_T, t

    @staticmethod
    def extract(x, t, ndim):
        """Extract values from x at t and reshape them to n-dim tensor"""
        return x.gather(0, t).reshape(-1, *((1, ) * (ndim - 1)))

    def estimate_x_t(self, x_0, t, x_T=None):
        # NOTE: t == 0 means diffused for 1 step (https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils.py#L108)
        # NOTE: t == 0 means not diffused for cold-diffusion (in contradiction to the above comment) https://github.com/arpitbansal297/Cold-Diffusion-Models/blob/c828140b7047ca22f995b99fbcda360bc30fc25d/denoising-diffusion-pytorch/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L361
        x_T = self.x_final(x_0) if x_T is None else x_T

        # ndim = x_0.ndim
        # x_t = (self.extract(self.sqrt_alphas_cumprod, t, ndim)*x_0 +
        #         self.extract(self.sqrt_one_minus_alphas_cumprod, t, ndim)*x_T)
        def clipper(b):
            tb = t[b]
            if tb < 0:
                return x_0[b]
            elif tb >= self.T:
                return x_T[b]
            else:
                return self.sqrt_alphas_cumprod[tb] * x_0[b] + self.sqrt_one_minus_alphas_cumprod[tb] * x_T[b]

        x_t = torch.stack([clipper(b) for b in range(t.shape[0])])
        return x_t

    def estimate_x_0(self, x_t, x_T, t):
        ndim = x_t.ndim
        x_0 = (self.extract(self.sqrt_recip_alphas_cumprod, t, ndim) * x_t -
               self.extract(self.sqrt_recipm1_alphas_cumprod, t, ndim) * x_T)
        return x_0

    def estimate_x_T(self, x_t, x_0, t):
        ndim = x_t.ndim
        return ((self.extract(self.sqrt_recip_alphas_cumprod, t, ndim) * x_t - x_0) /
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, ndim))

    @classmethod
    def x_final(cls, x):
        return torch.randn_like(x)

    @classmethod
    def _clip_x_0(cls, x_0):
        # See "static/dynamic thresholding" in Imagen https://arxiv.org/abs/2205.11487

        # "static thresholding"
        m = 1  # Set this to about 4*sigma = 4 if latent diffusion is used
        x_0 = x_0.clamp(-m, m)

        return x_0

    def forward(self, x, state: Dict[str, Any]):
        data = x
        return self.sample(data)


class SequentialEmb(nn.Sequential):
    def forward(self, input_data, emb):
        for layer in self:
            if isinstance(layer, ResnetBlock):
                input_data = layer(input_data, emb)
            else:
                input_data = layer(input_data)
        return input_data


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, hid_chs=[256, 256, 512, 1024], emb_dim=4):
        super().__init__()
        self.depth = len(hid_chs)

        self.in_blocks = nn.ModuleList()

        # ----------- In-Convolution ------------
        self.in_blocks.append(SequentialEmb(nn.Conv2d(in_ch, hid_chs[0], kernel_size=3, stride=1, padding=1)))
        input_block_channels = [hid_chs[0]]

        # -------------- Encoder ----------------
        block_in = hid_chs[0]
        for i in range(len(hid_chs)):
            block_out = hid_chs[i]
            self.in_blocks.append(SequentialEmb(ResnetBlock(block_in, block_out, emb_dim)))
            self.in_blocks.append(SequentialEmb(ResnetBlock(block_out, block_out, emb_dim)))
            block_in = block_out
            input_block_channels += [block_out, block_out]
            if i != len(hid_chs) - 1:
                self.in_blocks.append(SequentialEmb(Downsample(block_in)))
                input_block_channels += [block_in]

        # ----------- Middle ------------
        self.middle_block = SequentialEmb(
            ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=emb_dim),
            ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=emb_dim))

        # ------------ Decoder ----------
        self.out_blocks = nn.ModuleList([])
        for i in reversed(range(len(hid_chs))):
            for j in range(3):
                self.out_blocks.append(
                    SequentialEmb(
                        ResnetBlock(block_in + input_block_channels.pop(),
                                    out_channels=hid_chs[i],
                                    temb_channels=emb_dim)))
                block_in = hid_chs[i]
                if i != 0 and j == 2:
                    self.out_blocks.append(Upsample(block_in))

        # --------------- Out-Convolution ----------------
        self.outc = nn.Sequential(
            Normalize(block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, 3, padding=1),
        )

    def forward(self, x, emb=None):
        x_input_block = []
        # --------- Encoder --------------
        for i, module in enumerate(self.in_blocks):
            x = module(x, emb)
            x_input_block.append(x)

        # ---------- Middle --------------
        x = self.middle_block(x, emb)

        # -------- Decoder -----------
        for module in self.out_blocks:
            if isinstance(module, SequentialEmb):
                x = torch.cat([x, x_input_block.pop()], dim=1)
                x = module(x, emb)
            else:
                x = module(x)

        return self.outc(x)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self):
        return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])

    def sample_loss(self):
        return self.sample(), torch.mean(self.kl())


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x, emb=None):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x, emb=None):
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        x = self.conv(x)
        return x


def nonlinearity(x):
    return x * torch.sigmoid(x)  # swish


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, self.out_channels)
        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = torch.nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)

    def forward(self, x, temb=None):
        h = x
        h = self.conv1(nonlinearity(self.norm1(h)))
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(nonlinearity(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        return x + h


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels=4, hid_chs=[64, 128, 256, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_ch = 0
        self.num_resolution = len(hid_chs)
        self.inc = nn.Conv2d(self.in_channels, hid_chs[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = hid_chs[0]
        for i in range(len(hid_chs)):
            down = nn.Module()
            block_out = hid_chs[i]
            down.block = nn.ModuleList(
                [ResnetBlock(block_in, block_out, self.temb_ch), ResnetBlock(block_out, block_out, self.temb_ch)])
            block_in = block_out
            if i != len(hid_chs):
                down.downsample = Downsample(block_out)
            self.down.append(down)

        self.mid = nn.ModuleList([
            ResnetBlock(hid_chs[-1], hid_chs[-1], self.temb_ch),
            ResnetBlock(hid_chs[-1], hid_chs[-1], self.temb_ch),
        ])

        # end
        self.norm_out = Normalize(hid_chs[-1])
        self.conv_out = torch.nn.Conv2d(hid_chs[-1], self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None

        # downsampling
        hs = [self.inc(x)]
        for i_level in range(self.num_resolution):
            for i in range(2):
                h = self.down[i_level].block[i](hs[-1], temb)
                hs.append(h)

            if i_level != self.num_resolution - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = self.mid[0](hs[-1], temb)
        h = self.mid[1](hs[-1], temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, z_channels, out_channels=3, hid_chs=[64, 128, 256, 512]):
        super().__init__()
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.temb_ch = 0
        self.num_resolutions = len(hid_chs)
        self.inc = nn.Conv2d(self.z_channels, hid_chs[-1], kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.ModuleList(
            [ResnetBlock(hid_chs[-1], hid_chs[-1], self.temb_ch), ResnetBlock(hid_chs[-1], hid_chs[-1], self.temb_ch)])

        self.up = nn.ModuleList()
        block_in = hid_chs[-1]
        for i_level in reversed(range(self.num_resolutions)):
            up = nn.Module()
            block_out = hid_chs[i_level]
            up.block = nn.ModuleList([
                ResnetBlock(block_in, block_out, self.temb_ch),
                ResnetBlock(block_out, block_out, self.temb_ch),
                ResnetBlock(block_out, block_out, self.temb_ch)
            ])
            block_in = block_out
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.append(up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        # downsampling
        h = self.inc(x)
        # middle
        h = self.mid[0](h, temb)
        h = self.mid[1](h, temb)
        for i_level in range(self.num_resolutions):
            for i_block in range(3):
                h = self.up[i_level].block[i_block](h, temb)
            if i_level != self.num_resolutions - 1:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class AutoEncoderKl(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, z_channels=16, emb_channels=16, hid_chs=[64, 128, 256, 512]):
        super().__init__()
        # ----------- In-Convolution ------------
        self.encoder = Encoder(in_channels, 2 * z_channels)
        self.decoder = Decoder(z_channels, out_channels)
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        h = self.post_quant_conv(z)
        x = self.decoder(h)
        return x

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.encode(x_in)
        # --------- Quantizer --------------
        z_q, emb_loss = h.sample_loss()
        # -------- Decoder -----------
        dec = self.decode(z_q)
        return dec, emb_loss


class PerceptionLoss(TensorOp):
    def __init__(self, inputs, outputs, mode, perceptual_loss_weight=1.0):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.perceiver = LPIPS()
        self.perceiver.to('cuda')
        self.perceiver.eval()
        self.perceptual_loss_weight = perceptual_loss_weight

    def forward(self, data, state: Dict[str, Any]):
        pred, target = data
        self.perceiver.eval()
        return reduce_sum(self.perceiver(pred, target) * self.perceptual_loss_weight)


class SSIMLoss(TensorOp):
    def __init__(self, inputs, outputs, mode):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)

    def forward(self, data, state: Dict[str, Any]):
        pred, target = data
        return reduce_sum(1 - ssim(((pred + 1) / 2).clamp(0, 1), (target.type(pred.dtype) + 1) / 2,
                                   data_range=1,
                                   size_average=False,
                                   nonnegative_ssim=True).reshape(-1, *[1] * (pred.ndim - 1)))


class CombinedLoss(TensorOp):
    def __init__(self,
                 inputs,
                 outputs,
                 mode,
                 batch_size=8,
                 embedding_loss_weight=1e-5,
                 perceptual_loss_weight=0.5,
                 ssim_loss_weight=4):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.ssim_loss_weight = ssim_loss_weight

    def forward(self, data, state: Dict[str, Any]):
        l1_loss, perception_loss, ssim_loss, emb_loss = data
        loss = l1_loss + reduce_mean(perception_loss * self.perceptual_loss_weight +
                                     ssim_loss * self.ssim_loss_weight) + emb_loss * self.embedding_loss_weight
        return loss


class LatentEmbedder(nn.Module):
    def __init__(self, model, weights_path=None):
        super().__init__()
        self.model = model
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path), strict=False)

    def forward(self, x):
        return self.model.encode(x).sample()


if __name__ == "__main__":
    #training parameters
    epochs = 40
    batch_size = 8
    image_size = 256
    train_steps_per_epoch = None
    save_dir = 'output/apphub/latent_embedder'

    train_data = load_data('/raid/shared_data')

    pipeline = fe.Pipeline(
        train_data=train_data,
        eval_data=train_data.split(0.1),
        batch_size=batch_size,
        ops=[
            ReadImage(inputs="image", parent_path=train_data.parent_path, outputs="image", color_flag='color'),
            Delete(['mask_left', 'mask_right']),
            Resize(image_in="image", width=image_size, height=image_size),
            Minmax(inputs="image", outputs="image"),
            ChannelTranspose(inputs="image", outputs="image")
        ])

    auto_encode_model = fe.build(model_fn=lambda: AutoEncoderKl(),
                                 optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-4),
                                 model_name="encoder")

    network = fe.Network(ops=[
        ModelOp(model=auto_encode_model, inputs="image", outputs=['pred', 'emb_loss']),
        L1_Loss(inputs=['pred', 'image'], outputs='l1_loss', mode="train", average_loss=False),
        PerceptionLoss(inputs=['pred', 'image'], outputs='perception_loss', mode="!test"),
        SSIMLoss(inputs=['pred', 'image'], outputs='ssim_loss', mode="!test"),
        CombinedLoss(inputs=['l1_loss', 'perception_loss', 'ssim_loss', 'emb_loss'], outputs='loss', mode="train"),
        UpdateOp(model=auto_encode_model, loss_name="loss")
    ])

    traces = [BestModelSaver(model=auto_encode_model, save_dir=save_dir, metric='ssim_loss')]

    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             monitor_names=["emb_loss", "l1_loss", "perception_loss", "ssim_loss", 'loss'],
                             epochs=epochs,
                             traces=traces,
                             train_steps_per_epoch=400,
                             eval_steps_per_epoch=40,
                             log_steps=40)

    estimator.fit()

    noise_estimator = fe.build(model_fn=lambda: UNet(in_ch=16, out_ch=16, hid_chs=[256, 256, 512, 1024], emb_dim=1024),
                               optimizer_fn=lambda x: torch.optim.AdamW(x, lr=1e-4),
                               model_name="noise_estimator")

    time_embedder = fe.build(model_fn=lambda: TimeEmbbeding(emb_dim=1024),
                             optimizer_fn=lambda x: torch.optim.AdamW(x, lr=1e-4),
                             model_name="time_embedder")

    auto_encode_model = fe.build(
        model_fn=lambda: LatentEmbedder(AutoEncoderKl(), 'output/apphub/latent_embedder/encoder_best_ssim_loss.pt'),
        optimizer_fn=lambda x: torch.optim.AdamW(x, lr=1e-4),
        model_name="encoder")

    network = fe.Network(ops=[
        ModelOp(model=auto_encode_model, inputs="image", outputs='encoded_image', trainable=False, gradients=False),
        GaussianNoiseScheduler(
            inputs='encoded_image', outputs=['encoded_image_t', 'encoded_image_T', 't'], mode='train'),
        ModelOp(model=time_embedder, inputs='t', outputs='time_embedding', mode='train'),
        ModelOp(model=noise_estimator, inputs=['encoded_image_t', 'time_embedding'], outputs='pred', mode='train'),
        L1_Loss(inputs=['pred', 'encoded_image_T'], outputs='l1_loss', mode="train"),
        UpdateOp(model=time_embedder, loss_name="l1_loss"),
        UpdateOp(model=noise_estimator, loss_name="l1_loss")
    ])

    traces = [
        BestModelSaver(model=time_embedder, save_dir=save_dir, metric='l1_loss'),
        BestModelSaver(model=noise_estimator, save_dir=save_dir, metric='l1_loss')
    ]

    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             monitor_names=['l1_loss'],
                             epochs=20,
                             traces=traces,
                             train_steps_per_epoch=1000,
                             eval_steps_per_epoch=50,
                             log_steps=20)

    estimator.fit()

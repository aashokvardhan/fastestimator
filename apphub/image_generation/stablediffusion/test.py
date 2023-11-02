from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from pytorch_msssim import SSIM, ssim
from medical_diffusion.models.utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from medical_diffusion.loss.gan_losses import hinge_d_loss
from medical_diffusion.loss.perceivers import LPIPS
from medical_diffusion.models.model_base import BasicModel, VeryBasicModel


class DiagonalGaussianDistribution(nn.Module):

    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        batch_size = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar)/batch_size

        return z, kl


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, emb_channels, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb_channels = emb_channels
        self.beta = beta

        self.embedder = nn.Embedding(num_embeddings, emb_channels)
        self.embedder.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        assert z.shape[1] == self.emb_channels, "Channels of z and codebook don't match"
        z_ch = torch.moveaxis(z, 1, -1) # [B, C, *] -> [B, *, C]
        z_flattened = z_ch.reshape(-1, self.emb_channels) # [B, *, C] -> [Bx*, C], Note: or use contiguous() and view()

        # distances from z to embeddings e: (z - e)^2 = z^2 + e^2 - 2 e * z
        dist = (    torch.sum(z_flattened**2, dim=1, keepdim=True)
                 +  torch.sum(self.embedder.weight**2, dim=1)
                -2* torch.einsum("bd,dn->bn", z_flattened, self.embedder.weight.t())
        ) # [Bx*, num_embeddings]

        min_encoding_indices = torch.argmin(dist, dim=1) # [Bx*]
        z_q = self.embedder(min_encoding_indices) # [Bx*, C]
        z_q = z_q.view(z_ch.shape) # [Bx*, C] -> [B, *, C]
        z_q = torch.moveaxis(z_q, -1, 1) # [B, *, C] -> [B, C, *]

        # Compute Embedding Loss
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss



class Discriminator(nn.Module):
    def __init__(self,
        in_channels=1,
        spatial_dims = 3,
        hid_chs =    [32,       64,      128,      256,  512],
        kernel_sizes=[(1,3,3), (1,3,3), (1,3,3),    3,   3],
        strides =    [  1,     (1,2,2), (1,2,2),    2,   2],
        act_name=("Swish", {}),
        norm_name = ("GROUP", {'num_groups':32, "affine": True}),
        dropout=None
        ):
        super().__init__()

        self.inc =  BasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=hid_chs[0],
            kernel_size=kernel_sizes[0], # 2*pad = kernel-stride -> kernel = 2*pad + stride => 1 = 2*0+1, 3, =2*1+1, 2 = 2*0+2, 4 = 2*1+2
            stride=strides[0],
            norm_name=norm_name,
            act_name=act_name,
            dropout=dropout,
        )

        self.encoder = nn.Sequential(*[
            BasicBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[i-1],
                out_channels=hid_chs[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                act_name=act_name,
                norm_name=norm_name,
                dropout=dropout)
            for i in range(1, len(hid_chs))
        ])


        self.outc =  BasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hid_chs[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            act_name=None,
            norm_name=None,
            dropout=None,
            zero_conv=True
        )



    def forward(self, x):
        x = self.inc(x)
        x = self.encoder(x)
        return self.outc(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self,
        in_channels=1,
        spatial_dims = 3,
        hid_chs =    [64, 128, 256, 512, 512],
        kernel_sizes=[4,   4,   4,  4,   4],
        strides =    [2,   2,   2,  1,   1],
        act_name=("LeakyReLU", {'negative_slope': 0.2}),
        norm_name = ("BATCH", {}),
        dropout=None
        ):
        super().__init__()

        self.inc =  BasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=hid_chs[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            norm_name=None,
            act_name=act_name,
            dropout=dropout,
        )

        self.encoder = nn.Sequential(*[
            BasicBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[i-1],
                out_channels=hid_chs[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                act_name=act_name,
                norm_name=norm_name,
                dropout=dropout)
            for i in range(1, len(strides))
        ])


        self.outc =  BasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hid_chs[-1],
            out_channels=1,
            kernel_size=4,
            stride=1,
            norm_name=None,
            act_name=None,
            dropout=False,
        )

    def forward(self, x):
        x = self.inc(x)
        x = self.encoder(x)
        return self.outc(x)




class VQVAE(BasicModel):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 spatial_dims=2,
                 emb_channels=4,
                 num_embeddings=8192,
                 hid_chs=[64, 128, 256, 512],
                 kernel_sizes=[3, 3, 3, 3],
                 strides=[1, 2, 2, 2],
                 norm_name=("GROUP", {
                     'num_groups': 32,
                     "affine": True
                 }),
                 act_name=("Swish", {}),
                 dropout=0.0,
                 use_res_block=True,
                 deep_supervision=False,
                 learnable_interpolation=True,
                 use_attention='none',
                 beta=0.25,
                 embedding_loss_weight=1.0,
                 perceiver=LPIPS,
                 perceiver_kwargs={},
                 perceptual_loss_weight=1.0,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={'lr': 1e-4},
                 lr_scheduler=None,
                 lr_scheduler_kwargs={},
                 loss=torch.nn.L1Loss,
                 loss_kwargs={'reduction': 'none'},
                 sample_every_n_steps=1000):
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )
        self.sample_every_n_steps=sample_every_n_steps
        self.loss_fct = loss(**loss_kwargs)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None
        self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides)
        self.depth = len(strides)
        self.deep_supervision = deep_supervision

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(spatial_dims, in_channels, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0],
                                  act_name=act_name, norm_name=norm_name)

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims,
                hid_chs[i-1],
                hid_chs[i],
                kernel_sizes[i],
                strides[i],
                kernel_sizes[i],
                norm_name,
                act_name,
                dropout,
                use_res_block,
                learnable_interpolation,
                use_attention[i])
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = BasicBlock(spatial_dims, hid_chs[-1], emb_channels, 1)


        # ----------- Quantizer --------------
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            emb_channels=emb_channels,
            beta=beta
        )

        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name)

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims,
                hid_chs[i+1],
                hid_chs[i],
                kernel_size=kernel_sizes[i+1],
                stride=strides[i+1],
                upsample_kernel_size=strides[i+1],
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                skip_channels=0)
            for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True)
            for i in range(1, deep_supervision+1)
        ])


    def encode(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        return z

    def decode(self, z):
        z, _ = self.quantizer(z)
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h)
        x = self.outc(h)
        return x

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.inc(x_in)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        out_hor = []
        h = self.inc_dec(z_q)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(self.outc_ver) else None
            h = self.decoders[i](h)
        out = self.outc(h)

        return out, out_hor[::-1], emb_loss

    
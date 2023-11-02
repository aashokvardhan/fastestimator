import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
        z_ch = torch.moveaxis(z, 1, -1)  # [B, C, *] -> [B, *, C]
        z_flattened = z_ch.reshape(-1, self.emb_channels)  # [B, *, C] -> [Bx*, C], Note: or use contiguous() and view()

        # distances from z to embeddings e: (z - e)^2 = z^2 + e^2 - 2 e * z
        dist = (torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(self.embedder.weight**2, dim=1) -
                2 * torch.einsum("bd,dn->bn", z_flattened, self.embedder.weight.t()))  # [Bx*, num_embeddings]

        min_encoding_indices = torch.argmin(dist, dim=1)  # [Bx*]
        z_q = self.embedder(min_encoding_indices)  # [Bx*, C]
        z_q = z_q.view(z_ch.shape)  # [Bx*, C] -> [B, *, C]
        z_q = torch.moveaxis(z_q, -1, 1)  # [B, *, C] -> [B, C, *]

        # Compute Embedding Loss
        loss = self.beta * torch.mean((z_q.detach() - z)**2) + torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout, temb_channels=0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.block1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True),
                                    nn.SiLU())

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True),
            nn.SiLU(),
        )

        self.short_cut = None
        if in_channels != out_channels:
            self.short_cut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.short_cut = nn.Identity()

    def forward(self, x, temb=None):
        h = self.block1(x)

        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]

        h = self.block2(h)

        if self.short_cut is not None:
            x = self.short_cut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(self, *, in_channels, z_channels, encoder_channels=(64, 128, 256, 512), dropout=0.0):
        super().__init__()
        self.in_channels = in_channels

        self.block1 = nn.Sequential(
            ResnetBlock(in_channels=in_channels, out_channels=encoder_channels[0], dropout=dropout),
            nn.Conv2d(encoder_channels[0], encoder_channels[1], kernel_size=3, stride=2, padding=1),
            ResnetBlock(in_channels=encoder_channels[1], out_channels=encoder_channels[1], dropout=dropout),
            nn.Conv2d(encoder_channels[1], encoder_channels[2], kernel_size=3, stride=2, padding=1),
            ResnetBlock(in_channels=encoder_channels[2], out_channels=encoder_channels[2], dropout=dropout),
            torch.nn.Conv2d(encoder_channels[2], encoder_channels[3], kernel_size=3, stride=2, padding=1),
            ResnetBlock(in_channels=encoder_channels[3], out_channels=encoder_channels[3], dropout=dropout),
            nn.Conv2d(encoder_channels[3], z_channels, kernel_size=1, stride=1))

    def forward(self, x):
        return self.block1(x)


class Decoder(nn.Module):
    def __init__(self, *, out_channels, z_channels, encoder_channels=(64, 128, 256, 512), dropout=0.0, **ignore_kwargs):
        super().__init__()
        self.block1 = ResnetBlock(in_channels=z_channels, out_channels=encoder_channels[-1], dropout=dropout)
        self.block2 = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], encoder_channels[-2], kernel_size=3, stride=1, padding=1),
            ResnetBlock(in_channels=encoder_channels[-2], out_channels=encoder_channels[-2], dropout=dropout))
        self.block3 = nn.Sequential(
            nn.Conv2d(encoder_channels[-2], encoder_channels[-3], kernel_size=3, stride=1, padding=1),
            ResnetBlock(in_channels=encoder_channels[-3], out_channels=encoder_channels[-3], dropout=dropout))
        self.block4 = nn.Sequential(
            torch.nn.Conv2d(encoder_channels[-3], encoder_channels[-4], kernel_size=3, stride=1, padding=1),
            ResnetBlock(in_channels=encoder_channels[-4], out_channels=encoder_channels[-4], dropout=dropout))
        self.block5 = nn.Conv2d(encoder_channels[-4], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.block5(x)


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
                     'num_groups': 32, "affine": True
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
        super().__init__(optimizer=optimizer,
                         optimizer_kwargs=optimizer_kwargs,
                         lr_scheduler=lr_scheduler,
                         lr_scheduler_kwargs=lr_scheduler_kwargs)
        self.sample_every_n_steps = sample_every_n_steps
        self.loss_fct = loss(**loss_kwargs)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None
        self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention] * len(strides)
        self.depth = len(strides)
        self.deep_supervision = deep_supervision

        # ----------- Encoder ----------------
        self.encoders = Encoder(in_channels=in_channels, z_channels=emb_channels, encoder_channels=hid_chs)

        # ----------- Quantizer --------------
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, emb_channels=emb_channels, beta=beta)

        # ------------ Decoder ----------
        self.decoders = Decoder(out_channels=out_channels, z_channels=emb_channels, encoder_channels=hid_chs)

    def encode(self, x):
        h = self.encoders(x)
        return h

    def decode(self, z):
        z, _ = self.quantizer(z)
        h = self.decoders(z)
        return h

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.encoders(x_in)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(h)

        # -------- Decoder -----------
        out_hor = []
        h = self.decoders(z_q)

        return h, out_hor[::-1], emb_loss * self.embedding_loss_weight

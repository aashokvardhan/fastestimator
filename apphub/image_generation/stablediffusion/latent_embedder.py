import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

group_normal_config = {'activation': 'relu', 'padding': 'same', 'kernel_initializer': 'he_normal'}


def normalization(channels: int):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups and `eps`.
    """
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]

    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2)**2) * torch.exp(-logvar2))


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.std).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar +
                    other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


def normalization(channels: int):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups and `eps`.
    """
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout, conv_shortcut=False, temb_channels=0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_norm = normalization(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.temb_proj = None
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.silu_normalization = torch.nn.SiLU()
        self.out_norm = normalization(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.short_cut = None
        if in_channels != out_channels:
            if conv_shortcut:
                self.short_cut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.short_cut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.in_norm(h)
        h = self.silu_normalization(h)
        h = self.conv1(h)

        if self.temb_proj:
            h = h + self.temb_proj(self.silu_normalization(temb))[:, :, None, None]

        h = self.out_norm(h)
        h = self.silu_normalization(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.short_cut is not None:
            x = self.short_cut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalization(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.scale = in_channels**-0.5

    def forward(self, x):
        # Normalize `x`
        x_norm = self.norm(x)
        # Get query, key and vector embeddings
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Reshape to query, key and vector embeddings from
        # `[batch_size, channels, height, width]` to
        # `[batch_size, channels, height * width]`
        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        attn = torch.nn.functional.softmax(attn, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        out = torch.einsum('bij,bcj->bci', attn, v)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        out = out.view(b, c, h, w)

        out = self.proj_out(out)

        # Add residual connection
        return x + out


class Encoder(nn.Module):
    def __init__(self,
                 *,
                 channels,
                 in_channels,
                 resolution,
                 z_channels,
                 num_res_blocks,
                 attn_resolutions,
                 channel_multipliers=(1, 2, 4, 8),
                 dropout=0.0,
                 resamp_with_conv=True,
                 double_z=True,
                 **ignore_kwargs):
        super().__init__()
        self.channels = channels
        self.num_resolutions = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.temb_channels = 0

        self.in_ch_mult = (1, ) + tuple(channel_multipliers)
        # Initial $3 \times 3$ convolution layer that maps the image to `channels`
        self.conv_in = torch.nn.Conv2d(in_channels, self.channels, kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):

            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * self.in_ch_mult[i_level]
            block_out = channels * channel_multipliers[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out

                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.out_group_normalization = normalization(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        self.silu_norm = torch.nn.SiLU()

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.out_group_normalization(h)
        h = self.silu_norm(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self,
                 *,
                 channels,
                 in_channels,
                 resolution,
                 z_channels,
                 out_channels,
                 num_res_blocks,
                 attn_resolutions,
                 channel_multipliers=(1, 2, 4, 8),
                 dropout=0.0,
                 resamp_with_conv=True,
                 give_pre_end=False,
                 tanh_out=False,
                 attn_type="vanilla",
                 **ignorekwargs):
        super().__init__()
        self.channels = channels
        self.num_resolutions = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.temb_ch = 0

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1, ) + tuple(channel_multipliers)
        block_in = channels * channel_multipliers[self.num_resolutions - 1]
        curr_res = resolution // 2**(self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channel_multipliers[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.silu_norm = nn.SiLU()
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.silu_norm(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class Autoencoder(Module):
    def __init__(self, ddconfig, emb_channels, z_channels, learn_logvar=False):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = torch.nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, img):
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(img)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        # Decode the image of shape `[batch_size, channels, height, width]`
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs,
                                            reconstructions,
                                            posterior,
                                            optimizer_idx,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs,
                                        reconstructions,
                                        posterior,
                                        0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val" + postfix)

        discloss, log_dict_disc = self.loss(inputs,
                                            reconstructions,
                                            posterior,
                                            1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val" + postfix)

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
            self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

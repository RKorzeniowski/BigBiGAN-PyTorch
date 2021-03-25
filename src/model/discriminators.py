import torch
import torch.nn as nn

from src.model import layers
from src.training_utils import training_utils

class WDisc(nn.Module):
    def clip_weights(self, weight_cutoff):
        for param in self.parameters():
            param.data.clamp_(-weight_cutoff, weight_cutoff)


class DiscBigGAN(nn.Module):
    def __init__(self, mult_chs, ks, num_cls, sn, w_init):
        super().__init__()

        m_pre_chs, m_post_chs, colors = mult_chs["pre"], mult_chs["post"], mult_chs["colors"]
        resblocks_output = m_post_chs[-1]
        m_post_chs = training_utils.get_channel_inputs(m_post_chs, input_dim=m_pre_chs[-1])
        m_pre_chs = training_utils.get_channel_inputs(m_pre_chs, input_dim=colors)

        self.pre_down_blocks = nn.Sequential(*[
            layers.DownResnetBlock(in_ch=in_m, out_ch=out_m, ks=ks, sn=sn, bias=False, w_init=w_init)
            for in_m, out_m in m_pre_chs
        ]) # tf 64 -> # here 64

        self.non_loc = layers.SelfAttn(mult_chs["pre"][-1], sn=sn) # tf 64 # here 64
        self.post_down_blocks = nn.Sequential(*[
            layers.DownResnetBlock(in_ch=in_m, out_ch=out_m, ks=ks, sn=sn, bias=False, w_init=w_init)
            for in_m, out_m in m_post_chs
        ]) # tf -> 128 -> 256 # here 128 -> 256

        self.res_block = layers.ConstResnetBlock(resblocks_output, resblocks_output, ks, sn=sn, bias=False, w_init=w_init)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=resblocks_output, out_features=1)
        self.cls_embedding = nn.Embedding(num_embeddings=num_cls, embedding_dim=resblocks_output) # does this embedding dim should be different than the generetor ones?

        if w_init is not None: w_init(self.linear.weight)

    def forward(self, x, cls):
        y = self.pre_down_blocks(x)
        y = self.non_loc(y)
        y = self.post_down_blocks(y)
        y = self.res_block(y)
        y = self.relu(y)
        y1 = torch.sum(y, dim=[-2, -1])
        y2 = self.linear(y1)

        if cls is not None:
            cls_embed = self.cls_embedding(cls)
            y2 = y2 + torch.sum(cls_embed * y1, dim=-1, keepdim=True)

        return y1, y2

    @classmethod
    def from_config(cls, config):
        return cls(
            mult_chs=config.disc_mult_chs,
            ks=config.ks,
            num_cls=config.num_cls,
            w_init=config.w_init,
            sn=config.spectral_norm,
        )


class DiscBigWGAN(DiscBigGAN, WDisc):
    pass


class DiscMLP(nn.Module):
    def __init__(self, n_blocks, mlp_dim, in_dim, dropout, sn, w_init):
        super().__init__()
        mlp = [layers.LinearResnetBlock(in_dim=mlp_dim, out_dim=mlp_dim, dropout=dropout, sn=sn, w_init=w_init)
               for _ in range(n_blocks - 1)]
        mlp = [layers.LinearResnetBlock(in_dim=in_dim, out_dim=mlp_dim, dropout=dropout, sn=sn, w_init=w_init)] + mlp
        self.mpls = nn.Sequential(*mlp)
        self.linear = nn.Linear(in_features=mlp_dim, out_features=1)
        if w_init is not None: w_init(self.linear.weight)


class LatentDisc(DiscMLP):
    def forward(self, z):
        y1 = self.mpls(z.float())
        y2 = self.linear(y1)
        return y1, y2

    @classmethod
    def from_config(cls, config):
        return cls(
            n_blocks=config.latent_disc_blocks,
            mlp_dim=config.latent_disc_mlp_dim,
            in_dim=config.latent_dim,
            dropout=config.dropout,
            w_init=config.w_init,
            sn=config.spectral_norm,
        )


class LatentWDisc(LatentDisc, WDisc):
    pass


class CombDisc(DiscMLP):
    def forward(self, img, latent):
        x = torch.cat([img, latent], dim=-1)
        y = self.mpls(x)
        y = self.linear(y)
        return y

    @classmethod
    def from_config(cls, config):
        return cls(
            n_blocks=config.comb_disc_blocks,
            mlp_dim=config.comb_disc_mlp_dim,
            in_dim=config.latent_disc_mlp_dim + config.disc_mult_chs["post"][-1],
            dropout=config.dropout,
            w_init=config.w_init,
            sn=config.spectral_norm,
        )


class CombWDisc(CombDisc, WDisc):
    pass

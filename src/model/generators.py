import torch
import torch.nn as nn

from src.model import layers
from src.training_utils import training_utils


class GenBigGAN(nn.Module):
    def __init__(self, mult_chs, ks, num_cls, latent_dim, embedding_dim, sn, w_init):
        super().__init__()
        self.ch = mult_chs["pre"][0]
        self.conditional = num_cls > 0

        m_pre_chs, m_post_chs, out_ch = mult_chs["pre"], mult_chs["post"], mult_chs["colors"]
        self.splits = (len(m_pre_chs) + len(m_post_chs) + 1)
        split_latent_dim = latent_dim // self.splits
        assert latent_dim % self.splits == 0, "latent has to be divisible by number of CondResnetBlocks layers"

        m_post_chs = training_utils.get_channel_inputs(m_post_chs, input_dim=m_pre_chs[-1])
        m_pre_chs = training_utils.get_channel_inputs(m_pre_chs, input_dim=m_pre_chs[0])
        top_block = [True] + [False] * (len(m_pre_chs) - 1)

        cond_dim = split_latent_dim + embedding_dim if self.conditional else split_latent_dim

        if self.conditional:
            self.class_embedding = nn.Embedding(num_embeddings=num_cls, embedding_dim=embedding_dim)
        self.linear = layers.LinearSN(in_features=split_latent_dim, out_features=4 * 4 * self.ch, sn=sn, w_init=w_init)
        # tf 4 * 4 * 256 # here 4 * 4 * 256
        self.pre_up_blocks = nn.Sequential(*[
            layers.UpResnetBlock(in_m, out_m, ks, cond_dim, sn, bias=False, w_init=w_init, first=f)
            for (in_m, out_m), f in zip(m_pre_chs, top_block)
        ]) # tf 256 -> 128 # here 256, 128
        self.non_loc = layers.SelfAttn(mult_chs["pre"][-1], sn=sn) # tf 128 -> # here 128
        # should be 2 times bigger same as output of prev block i.e. 256 // 2
        # but this implementation keeps the same dim so  ch // 2 -> attn -> ch // 4
        self.post_up_blocks = nn.Sequential(*[
            layers.UpResnetBlock(in_m, out_m, ks, cond_dim, sn, bias=False, w_init=w_init)
            for in_m, out_m in m_post_chs
        ]) # tf -> 64 # 64

        self.bn = nn.BatchNorm2d(mult_chs["post"][-1])
        self.relu = nn.ReLU()
        self.conv = layers.ConvTranspose2dSN(
            in_channels=mult_chs["post"][-1], out_channels=out_ch,
            kernel_size=ks, padding=1, sn=sn, bias=False, w_init=w_init)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, cls):
        z = z.float()
        all_z = z.chunk(self.splits, dim=-1)
        z, conds = all_z[0], all_z[1:]

        if self.conditional:
            cls_embed = self.class_embedding(cls)
            conds = [torch.cat([conds[d], cls_embed], dim=-1) for d in range(len(conds))]

        z = self.linear(z)
        z = z.reshape(-1, self.ch, 4, 4)

        for i, layer in enumerate(self.pre_up_blocks):
            z = layer(z, cond=conds[i])

        z = self.non_loc(z)

        for i, layer in enumerate(self.post_up_blocks, start=len(self.pre_up_blocks)):
            z = layer(z, cond=conds[i])

        z = self.bn(z)
        z = self.relu(z)
        z = self.conv(z)
        x = self.sigmoid(z)
        return x

    @classmethod
    def from_config(cls, config):
        return cls(
            mult_chs=config.gen_mult_chs,
            ks=config.ks,
            num_cls=config.num_cls,
            latent_dim=config.latent_dim,
            embedding_dim=config.embedding_dim,
            w_init=config.w_init,
            sn=config.spectral_norm,
        )

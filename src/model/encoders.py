import torch
import torch.nn as nn
from torchvision import models

from src.model import layers


class ResNetEnc(nn.Module):
    # add stochastic
    def __init__(self, resnet, ch, mlp_dim, dropout, w_init):
        super().__init__()
        in_dim = resnet.fc.in_features
        res_block1 = layers.LinearResnetBlock(in_dim, mlp_dim, dropout, sn=False, w_init=w_init)
        res_block2 = layers.LinearResnetBlock(mlp_dim, mlp_dim, dropout, sn=False, w_init=w_init)
        fc = nn.Sequential(res_block1, res_block2)
        resnet.fc = fc
        resnet.conv1 = nn.Conv2d(ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model = resnet

    def forward(self, x):
        return self.model(x)

    @classmethod
    def from_config(cls, config):
        model_version = config.enc_model
        if model_version == "resnet_18":
            resnet = models.resnet18(pretrained=False, progress=True)
        elif model_version == "resnet_50":
            resnet = models.resnet50(pretrained=False, progress=True)
        else:
            raise ValueError("This type of encoder is not supported")

        return cls(
            resnet=resnet,
            mlp_dim=config.enc_out_dim,
            dropout=config.dropout,
            ch=config.enc_mult_chs,
            w_init=config.w_init
        )


class RevNetEnc(nn.Module):
    def __init__(self, mult_chs, ks, in_mlp_dim, mlp_dim, latent_dim, w_init):
        super().__init__()
        m_chs, in_ch = mult_chs["blocks"], mult_chs["colors"]
        ch = m_chs[0]

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=ch, kernel_size=7, stride=2, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        rev_blocks = []
        for i, (in_ch, out_ch) in enumerate(zip(m_chs, m_chs)):
            if i and i % 2 == 0:
                rev_blocks.append(layers.RevPaddingLayer(stride=2))
            rev_blocks.append(layers.RevNetBlock(in_ch=in_ch, out_ch=out_ch, ks=ks, stride=1, w_init=w_init))

        self.rev_blocks = nn.Sequential(*rev_blocks)
        self.res_lin1 = layers.LinearResnetBlock(
            in_dim=in_mlp_dim, out_dim=mlp_dim, dropout=0, sn=False, w_init=w_init)
        self.res_lin2 = layers.LinearResnetBlock(
            in_dim=mlp_dim, out_dim=mlp_dim, dropout=0, sn=False, w_init=w_init)
        self.linear_dist_params = nn.Linear(in_features=mlp_dim, out_features=2*latent_dim)
        self.softplus = nn.Softplus(beta=1)

        self.init_weights(w_init)

    def init_weights(self, w_init):
        if w_init is not None:
            w_init(self.conv.weight)
            w_init(self.linear_dist_params.weight)

    def forward(self, x):
        x = self.upsample(x)
        y = self.conv(x)
        y = self.max_pool(y)
        for layer in self.rev_blocks:
            y = layer(y)
        y = torch.mean(y, dim=[-2, -1])
        y = self.res_lin1(y)
        y = self.res_lin2(y)
        y = self.linear_dist_params(y)
        mu, unnormed_std = torch.split(y, y.shape[1]//2, dim=1)
        std = self.softplus(unnormed_std)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    @classmethod
    def from_config(cls, config):
        return cls(
            mult_chs=config.enc_mult_chs,
            ks=config.ks,
            in_mlp_dim=config.enc_in_mlp_dim,
            mlp_dim=config.enc_hidden,
            latent_dim=config.latent_dim,
            w_init=config.w_init,
        )

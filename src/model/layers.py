import torch
import torch.nn as nn


class BNLinearResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, sn):
        super().__init__()
        self.linear1 = LinearSN(in_features=in_dim, out_features=out_dim, sn=sn)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = LinearSN(in_features=out_dim, out_features=out_dim, sn=sn)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu2 = nn.ReLU()

        self.skip_linear = LinearSN(in_features=in_dim, out_features=out_dim, sn=sn)

    def forward(self, x):
        x_skip = self.skip_linear(x)

        y = self.linear1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.linear2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = y + x_skip
        return y


class LinearResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, sn, w_init):
        super().__init__()
        self.linear1 = LinearSN(in_features=in_dim, out_features=out_dim, sn=sn, w_init=w_init)
        self.dropout1 = nn.Dropout(dropout)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.3)
        self.linear2 = LinearSN(in_features=out_dim, out_features=out_dim, sn=sn, w_init=w_init)
        self.dropout2 = nn.Dropout(dropout)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.3)

        self.skip_linear = LinearSN(in_features=in_dim, out_features=out_dim, sn=sn, w_init=w_init)
        self.skip_leaky_relu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        x_skip = self.skip_linear(x)
        x_skip = self.skip_leaky_relu(x_skip)

        y = self.linear1(x)
        y = self.dropout1(y)
        y = self.leaky_relu1(y)

        y = self.linear2(y)
        y = self.dropout2(y)
        y = self.leaky_relu2(y)

        y = y + x_skip
        return y


class UpResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, cond_dim, sn, bias, w_init, first=False):
        super().__init__()
        self.skip_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.skip_conv = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1,
            stride=1, sn=sn, bias=bias, w_init=w_init)

        # BN channels=channels * 2 in fist gen layer rest normal
        # why?
        self.bn1 = CondBatchNorm2d(cond_dim=cond_dim, num_features=in_ch)
        self.relu1 = nn.ReLU()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)
        self.bn2 = CondBatchNorm2d(cond_dim=cond_dim, num_features=out_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = Conv2dSN(
            in_channels=out_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)

    def forward(self, x, cond):
        x_skip = self.skip_upsample(x)
        x_skip = self.skip_conv(x_skip)

        y = self.bn1(x, cond)
        y = self.relu1(y)
        y = self.upsample(y)
        y = self.conv1(y)

        y = self.bn2(y, cond)
        y = self.relu2(y)
        y = self.conv2(y)

        y = y + x_skip
        return y


class DownResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sn, bias, w_init):
        super().__init__()
        self.skip_conv = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1,
            stride=1, sn=sn, bias=bias, w_init=w_init)
        self.skip_downsample = nn.AvgPool2d(kernel_size=ks, stride=2, padding=1)

        self.relu1 = nn.ReLU()
        self.conv1 = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)
        self.relu2 = nn.ReLU()
        self.conv2 = Conv2dSN(
            in_channels=out_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)
        self.downsample = nn.AvgPool2d(kernel_size=ks, stride=2, padding=1)

    def forward(self, x):
        x_skip = self.skip_conv(x)
        x_skip = self.skip_downsample(x_skip)

        # y = self.bn(x) in biggan
        y = self.relu1(x)
        y = self.conv1(y)
        # y = self.bn(y) in biggan
        y = self.relu2(y)
        y = self.conv2(y)
        y = self.downsample(y)
        y = y + x_skip
        return y


class ConstResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sn, bias, w_init):
        super().__init__()
        self.conv1 = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)
        self.relu = nn.ReLU()
        self.conv2 = Conv2dSN(
            in_channels=out_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        return y + x


class SelfAttn(nn.Module):
    # https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    def __init__(self, in_dim, sn):
        super(SelfAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2dSN(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, sn=sn)
        self.key_conv = Conv2dSN(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, sn=sn)
        self.value_conv = Conv2dSN(in_channels=in_dim, out_channels=in_dim, kernel_size=1, sn=sn)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class CondBatchNorm2d(nn.Module):
    # https://discuss.pytorch.org/t/conditional-batch-normalization/14412
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.linear = nn.Linear(in_features=cond_dim, out_features=num_features * 2)
        self.linear.weight.data[:, :cond_dim].normal_(1, 0.02)
        self.linear.weight.data[:, cond_dim:].zero_()

    def forward(self, x, cond):
        out = self.bn(x)
        gamma, beta = self.linear(cond).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


# class CondBatchNorm2d(torch.nn.Module):
#     def __init__(self, num_features, cond_dim, eps=2e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):
#         super().__init__()
#         self.num_features = num_features
#         # self.num_cats = num_cats
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.affine:
#             self.bn_weight = nn.Linear(in_features=cond_dim, out_features=num_features)
#             self.bn_bias = nn.Linear(in_features=cond_dim, out_features=num_features)
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(num_features))
#             self.register_buffer('running_var', torch.ones(num_features))
#             self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_var', None)
#             self.register_parameter('num_batches_tracked', None)
#         self.reset_parameters()
#
#     def reset_running_stats(self):
#         if self.track_running_stats:
#             self.running_mean.zero_()
#             self.running_var.fill_(1)
#             self.num_batches_tracked.zero_()
#
#     def reset_parameters(self):
#         self.reset_running_stats()
#         if self.affine:
#             self.bn_weight.weight.data.normal_(1, 0.02)
#             self.bn_bias.weight.data.zero_()
#
#     def forward(self, input, cats):
#         exponential_average_factor = 0.0
#
#         if self.training and self.track_running_stats:
#             self.num_batches_tracked += 1
#             if self.momentum is None:  # use cumulative moving average
#                 exponential_average_factor = 1.0 / self.num_batches_tracked.item()
#             else:  # use exponential moving average
#                 exponential_average_factor = self.momentum
#
#         out = torch.nn.functional.batch_norm(
#             input,
#             self.running_mean,
#             self.running_var,
#             None,
#             None,
#             self.training or not self.track_running_stats,
#             exponential_average_factor,
#             self.eps,
#         )
#
#         if self.affine:
#             shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
#             weight = self.bn_weight(cats).view(shape)
#             bias = self.bn_bias(cats).view(shape)
#             out = out * weight + bias
#
#         return out

# tf:    x,     mean,         variance,    offset,      scale,                              variance_epsilon
# torch: input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05

class BottleneckResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride, w_init):
        super().__init__()
        self.stride = stride
        self.skip_conv = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1,
            stride=stride, sn=False, w_init=w_init)

        self.conv1 = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, bias=False, sn=False, w_init=w_init)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = Conv2dSN(
            in_channels=out_ch, out_channels=out_ch, kernel_size=ks,
            stride=stride, padding=1, bias=False, sn=False, w_init=w_init)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)

        if (x.shape[1] != y.shape[1]) or self.stride > 1:
            x = self.skip_conv(x)

        return y + x


class RevNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride, w_init):
        super().__init__()
        self.res_block = BottleneckResnetBlock(
            in_ch=in_ch // 2, out_ch=out_ch // 2, ks=ks, stride=stride, w_init=w_init)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        y1 = x1 + self.res_block(x2)
        y2 = x2
        return torch.cat([y1, y2], dim=1)


class RevPaddingLayer(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.pool(x)
        zeros = torch.zeros_like(x)
        zeros_left, zeros_right = zeros.chunk(2, dim=1)
        y = torch.cat([zeros_left, x, zeros_right], dim=1)
        return y


class SpectralLayer(nn.Module):
    def __init__(self, layer, sn, w_init, *args, **kwargs):
        super().__init__()
        self.layer = layer
        if w_init is not None: w_init(self.layer.weight)
        self.layer = nn.utils.spectral_norm(self.layer) if sn else self.layer

    def forward(self, x):
        return self.layer(x)


class LinearSN(SpectralLayer):
    def __init__(self, sn, w_init=None, *args, **kwargs):
        layer = nn.Linear(*args, **kwargs)
        super().__init__(layer, sn, w_init, *args, **kwargs)


class Conv2dSN(SpectralLayer):
    def __init__(self, sn, w_init=None, *args, **kwargs):
        layer = nn.Conv2d(*args, **kwargs)
        super().__init__(layer, sn, w_init, *args, **kwargs)


class ConvTranspose2dSN(SpectralLayer):
    def __init__(self, *args, sn, w_init=None, **kwargs):
        layer = nn.ConvTranspose2d(*args, **kwargs)
        super().__init__(layer, sn, w_init, *args, **kwargs)

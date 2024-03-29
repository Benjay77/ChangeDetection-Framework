'''
 * @Author       : Benjay·Shaw
 * @Date         : 2021-08-16 16:27:01
 * @LastEditors  : Benjay·Shaw
 * @LastEditTime : 2021-08-20 08:43:42
 * @Description  : DynamicReLU
'''
import torch
import torch.nn as nn


class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)
        self.sigmoid = nn.Sigmoid()
        # λ控制 残差范围
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        # a\b系数初始值  a1=1,a2=b1=b2=0,即ReLU
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.avg_pool(theta)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2 * k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)
        # ab值= λ*残差值 + 初始值
        relu_coefs = theta.view(-1, 2 * self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        # 激活函数 y=ax+b
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2 * k * channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2 * self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result


class DyReLUC(nn.Module):
    def __init__(self, channels, reduction=4, k=2, tau=10, gamma=1 / 3):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.k = k
        self.tau = tau
        self.gamma = gamma

        self.coef = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 2 * k * channels, 1),
            nn.Sigmoid()
        )
        self.sptial = nn.Conv2d(channels, 1, 1)

        # default parameter setting
        # lambdaA = 1.0, lambdaB = 0.5;
        # alphaA1 = 1, alphaA2=alphaB1=alphaB2=0
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('bias', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def forward(self, x):
        N, C, H, W = x.size()
        coef = self.coef(x)
        coef = 2 * coef - 1

        # coefficient update
        coef = coef.view(-1, self.channels, 2 * self.k) * self.lambdas + self.bias

        # spatial
        gamma = self.gamma * H * W
        spatial = self.sptial(x)
        spatial = spatial.view(N, self.channels, -1) / self.tau
        spatial = torch.softmax(spatial, dim=-1) * gamma
        spatial = torch.clamp(spatial, 0, 1).view(N, 1, H, W)

        # activations
        # NCHW --> HWNC1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # HWNC1 * NCK --> HWNCK
        output = x_perm * coef[:, :, :self.k] + coef[:, :, self.k:]

        # permute spatial from NCHW to HWNC1
        spatial = spatial.permute(2, 3, 0, 1).unsqueeze(-1)
        output = spatial * output

        # maxout and HWNC --> NCHW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result

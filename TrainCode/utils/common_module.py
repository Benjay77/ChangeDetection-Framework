"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 公用模块
"""
import math

from torch.nn import functional, init
from torch.utils import model_zoo

from utils.attention import *
from utils.common_function import weights_init_kaiming


class Decoder(nn.Module):
    def __init__(self, in_ch, classes, attention_name='SELayer'):
        super(Decoder, self).__init__()
        num_classes = classes
        if attention_name == 'SELayer':
            self.attention_name = SELayer(in_ch)
        if attention_name == 'SKConv':
            self.attention_name = SKConv(in_ch, in_ch)
        # self.selayer = SELayer(in_ch)

        self.conv3_1 = nn.Conv2d(
            in_ch // 4, num_classes * 8, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(
            num_classes * 8, num_classes * 4, kernel_size=3, padding=1)

        self.ps2 = nn.PixelShuffle(2)
        self.ps3 = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.attention_name(x)  # [320, 64, 64]
        # print('x8: ', x.shape)
        x = self.ps2(x)  # [80, 128, 128]
        # print('x9: ', x.shape)

        x = self.conv3_1(x)  # [16, 128, 128]
        # print('x10: ', x.shape)
        x = self.conv3_2(x)  # [8, 128, 128]
        # print('x11: ', x.shape)

        x = self.ps3(x)  # [2, 256, 256]
        # print('x12: ', x.shape)
        return x


class Upsample(nn.Module):
    def __init__(self, in_ch, net_name='se_resnet50'):
        super(Upsample, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(
                in_ch * 2, in_ch * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        if net_name == 'se_resnet50':
            self.ps = nn.PixelShuffle(2)
        if net_name == 'sk_resnet50':
            self.ps = nn.PixelShuffle(4)

    def forward(self, x):
        x = self.conv1_1(x)  # [512, 32, 32]
        # print('x13: ', x.shape)
        x = self.conv1_2(x)  # [1024, 32, 32]
        # print('x14: ', x.shape)
        x = self.ps(x)  # [256, 64, 64]
        # print('x15: ', x.shape)
        return x


class CosineS(nn.Module):
    def __init__(self):
        super(CosineS, self).__init__()
        self.cosines = nn.CosineSimilarity(dim=1)

    def forward(self, x1, x2):
        similarity = self.cosines(x1, x2)  # [4, 64, 64]
        # print('similarity: ', similarity.shape)
        similarity = torch.unsqueeze(similarity, dim=1)  # [1, 64, 64]
        # print('similarity: ', similarity.shape)
        result1 = x2 * similarity.expand_as(x2)  # [128, 64, 64]
        # print('result1: ', result1.shape)
        result2 = x1 * similarity.expand_as(x1)  # [128, 64, 64]
        # print('result2: ', result2.shape)
        result = torch.cat([result1, result2], dim=1)  # [256, 64, 64]
        # print('result: ', result.shape)
        return result


class LocallyAdap(nn.Module):
    ''' locally adaptive  '''

    def __init__(self, in_ch, in_ch_ratio, attention_name='SELayer'):
        super(LocallyAdap, self).__init__()

        self.similarity = CosineS()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch // in_ch_ratio, 3, padding=1),
            nn.BatchNorm2d(in_ch // in_ch_ratio),
            nn.ReLU(inplace=True)
        )
        self.conv.apply(weights_init_kaiming)
        if attention_name == 'SELayer':
            self.attention_name = SELayer(in_ch // in_ch_ratio)
        if attention_name == 'SKConv':
            self.attention_name = SKConv(in_ch // in_ch_ratio, in_ch // in_ch_ratio)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        # x1_theta = self.conv_theta(x1)
        # x2_phi = self.conv_phi(x2)
        f = self.similarity(x1, x2)  # [1536, 32, 32]
        # print('f1: ', f.shape)
        f = self.conv(f)  # [256, 32, 32]
        # print('f2: ', f.shape)

        y = self.attention_name(f)  # [256, 32, 32]
        # print('y1: ', y.shape)
        # W_y = self.W(y)
        return y


class LocallyAdap2(nn.Module):
    ''' locally adaptive  '''

    def __init__(self, in_ch, ):
        super(LocallyAdap2, self).__init__()

        self.similarity = CosineS()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True)
        )
        self.conv.apply(weights_init_kaiming)
        self.selayer = SELayer(in_ch // 2)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        # x1_theta = self.conv_theta(x1)
        # x2_phi = self.conv_phi(x2)
        f = self.similarity(x1, x2)  # []256, 64, 64
        # print('f3: ', f.shape)
        f = self.conv(f)  # [64, 64, 64]
        # print('f4: ', f.shape)
        y = self.selayer(f)  # 64, 64, 64[]
        # print('y2: ', y.shape)
        # W_y = self.W(y)
        return y


class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, input):
        return functional.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                                     training=False, eps=self.eps)
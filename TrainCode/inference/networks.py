"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 推理模型
"""
import ssl

from torch import nn
from torchvision import models

from common_module import *


# ssl._create_default_https_context = ssl._create_unverified_context
class Bottleneck(nn.Module):  # （残差块的生成）block类，控制生成conv block or identity block
    expansion = 4  # 每个stage中维度拓展的倍数,通过网络图发现，输出通道都是mid层的4倍

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = FixedBatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = FixedBatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = FixedBatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)  # DyReLUB(planes, conv_type='2d')
        # Squeeze-and-Excitation
        self.se = SELayer(planes * 4, reduction)
        # Downsample
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)  # se加在左分支，downsample不加se
        # 是否直连（如果是Identity block就是直连；如果是Conv Block就需要对参差边进行卷积，改变通道数和size）
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNet(nn.Module):

    def __init__(self, block, layers, strides=(2, 2, 2, 2),
                 dilations=(1, 1, 2, 4)):  # layers的数据描述的是一层中conv层+identity层的数量
        super(SEResNet, self).__init__()
        self.inplanes = 64  # 经过前面的stem预处理后，生成64通道
        # stem的网络层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = FixedBatchNorm(64)
        self.relu = nn.ReLU(inplace=True)  # DyReLUB(64, conv_type='2d')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        self.inplanes = 1024

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, 1000)

        # planes=64，128，256，512指mid的维度
        # blocksn_num=layers[0]=3表明layer1中有3个残差块（包括conv+Identity）
        # stride=1指的是每个layer的起始conv的strip,既是残差块的左分支的步长，又是右分支步长

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        # 残差块右面小分支downsample模块
        if stride != 1 or self.inplanes != planes * block.expansion:  # 如果stride!=1则要加conv，输出不等于输入也要加conv
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                FixedBatchNorm(planes * block.expansion),
            )
        # 存放每个layers的残差Block
        # Conv Block（对于每层都是要先经过conv Block，然后在Identity block，所以conv先加入list中）
        layers = [block(self.inplanes, planes, stride, downsample,
                        dilation=1)]  # 生成conv Block,#resnet初始化时就有了inplane，调用make—layer层时赋值了(plane,stride)
        # planes=64是中间层，下面这行操作是为下一个残差块生成输入通道数
        self.inplanes = planes * block.expansion
        # identity block，通过block来查看某层layers需要多少个identity block
        for i in range(1, blocks):  # blocks是个数字
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


class SE_Resnet50(nn.Module):
    def __init__(self, nums_class, pretrained=True, **kwargs):
        super(SE_Resnet50, self).__init__()

        se_resnet50 = SEResNet(Bottleneck, layers=[3, 4, 6, 3], **kwargs)
        if pretrained:
            state_dict = model_zoo.load_url(model_urls['se_resnet50'])
            # state_dict.pop('fc.weights')
            # state_dict.pop('fc.bias')
            model_state_dict = se_resnet50.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            model_state_dict.update(pretrained_dict)
            se_resnet50.load_state_dict(model_state_dict)

        self.stage1 = nn.Sequential(se_resnet50.conv1, se_resnet50.bn1, se_resnet50.relu,
                                    se_resnet50.maxpool)
        self.stage2 = nn.Sequential(se_resnet50.layer1)
        self.stage3 = nn.Sequential(se_resnet50.layer2)
        self.stage4 = nn.Sequential(se_resnet50.layer3)
        self.stage5 = nn.Sequential(se_resnet50.layer4)

        self.dec = Decoder(320, nums_class)

        self.locally1 = LocallyAdap(768, 3)
        self.locally2 = LocallyAdap(128, 2)

        self.up1 = Upsample(256)

        self.fc_dp1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc_dp2 = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc_dp3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc_dp4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc_dp5 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )

    def encoder(self, x):
        x1 = self.stage1(x)  # [64, 64, 64]
        # print('x1: ', x1.shape)
        x2 = self.stage2(x1)  # [256, 64, 64]
        # print('x2: ', x2.shape)
        x3 = self.stage3(x2)  # [512, 32, 32]
        # print('x3: ', x3.shape)
        x4 = self.stage4(x3)  # [1024, 16, 16]
        # print('x4: ', x4.shape)
        x5 = self.stage5(x4)  # [2048, 8, 8]
        # print('x5: ', x5.shape)

        return x5, x4, x3, x2, x1

    def forward(self, x_prev, x_now):
        x5_prev, x4_prev, x3_prev, x2_prev, x1_prev = self.encoder(x_prev)
        x5_now, x4_now, x3_now, x2_now, x1_now = self.encoder(
            x_now)  # [2048, 16, 16], [1024, 32, 32], [512, 32, 32], [256, 64, 64], [64, 64, 64]
        # print('x5_now, x4_now, x3_now, x2_now, x1_now : ', x5_now.shape, x4_now.shape, x3_now.shape, x2_now.shape, x1_now.shape)

        x5_prev_s = self.fc_dp5(x5_prev)  # [256, 16, 16]
        # print('x5_prev_s: ', x5_prev_s.shape)
        x5_now_s = self.fc_dp5(x5_now)
        # print('x5_now_s: ', x5_now_s.shape)

        x4_prev_s = self.fc_dp4(x4_prev)  # [256, 16, 16]
        # print('x4_prev_s: ', x4_prev_s.shape)
        x4_now_s = self.fc_dp4(x4_now)  # [256, 16, 16]
        # print('x4_now_s: ', x4_now_s.shape)

        x3_prev_s = self.fc_dp3(x3_prev)  # [256, 128, 128]
        # print('x3_prev_s: ', x3_prev_s.shape)
        x3_now_s = self.fc_dp3(x3_now)  # [256, 128, 128]
        # print('x3_now_s: ', x3_now_s.shape)

        high_prev = torch.cat([x5_prev_s, x4_prev_s, x3_prev_s], dim=1)  # [768, 128, 128]
        # print('high_prev: ', high_prev.shape)
        high_now = torch.cat([x5_now_s, x4_now_s, x3_now_s], dim=1)  # [768, 128, 128]
        # print('high_now: ', high_now.shape)

        x_high = self.locally1(high_prev, high_now)  # [256, 128, 128]
        # print('x_high: ', x_high.shape)

        x_high_up = self.up1(x_high)  # [256, 256, 256]
        # print('x_high_up: ', x_high_up.shape)

        x2_prev_s = self.fc_dp2(x2_prev)  # [64, 256, 256]
        # print('x2_prev_s: ', x2_prev_s.shape)
        x2_now_s = self.fc_dp2(x2_now)  # [256, 128, 128]
        # print('x2_now_s: ', x5_now_s.shape)

        x1_prev_s = self.fc_dp1(x1_prev)  # [64, 256, 256]
        # print('x1_prev_s: ', x1_prev_s.shape)
        x1_now_s = self.fc_dp1(x1_now)  # [64, 256, 256]
        # print('x1_now_s: ', x1_now_s.shape)

        low_prev = torch.cat([x2_prev_s, x1_prev_s], dim=1)  # [128, 256, 256]
        # print('low_prev: ', low_prev.shape)
        low_now = torch.cat([x2_now_s, x1_now_s], dim=1)  # [128, 256, 256]
        # print('low_now: ', low_now.shape)

        x_low = self.locally2(low_prev, low_now)  # [64, 256, 256]
        # print('x_low: ', x_low.shape)

        x = torch.cat([x_low, x_high_up], dim=1)  # [320, 256, 256]
        # print('x6: ', x.shape)
        result = self.dec(x)  # [2, 1024, 1024]
        # print('x7: ', x.shape)

        # result = torch.argmax(x, dim=1, keepdim=True)
        return result
"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: SK_Resnet50
"""
from utils.common_module import *


class SKBlock(nn.Module):
    '''
    基于Res Block构造的SK Block
    ResNeXt由 1x1Conv（通道数：x） +  SKConv（通道数：x）  + 1x1Conv（通道数：2x） 构成
    '''
    expansion = 2  # 指 每个block中 通道数增大指定倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SKBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True))
        self.conv2 = SKConv(planes, planes, stride)
        self.conv3 = nn.Sequential(nn.Conv2d(planes, planes * self.expansion, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(planes * self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        shortcut = input
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        if self.downsample is not None:
            shortcut = self.downsample(input)
        output += shortcut
        return self.relu(output)


class SKNet(nn.Module):

    def __init__(self, nums_class=1000, block=SKBlock, nums_block_list=[3, 4, 6, 3]):
        super(SKNet, self).__init__()
        self.inplanes = 64
        # in_channel=3  out_channel=64  kernel=7x7 stride=2 padding=3
        self.conv = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # kernel=3x3 stride=2 padding=1
        self.layer1 = self._make_layer(block, 128, nums_block_list[0], stride=1)  # 构建表中 每个[] 的部分
        self.layer2 = self._make_layer(block, 256, nums_block_list[1], stride=2)
        self.layer3 = self._make_layer(block, 512, nums_block_list[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, nums_block_list[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)  # GAP全局平均池化
        # self.fc = nn.Linear(1024 * block.expansion, nums_class)  # 通道 2048 -> 1000
        # self.softmax = nn.Softmax(-1)  # 对最后一维进行softmax

    def forward(self, input):
        output = self.conv(input)
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        # output = self.avgpool(output)
        # output = output.squeeze(-1).squeeze(-1)
        # output = self.fc(output)
        # output = self.softmax(output)
        return output

    def _make_layer(self, block, planes, nums_block, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, nums_block):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class SK_Resnet50(nn.Module):
    def __init__(self, nums_class):
        super(SK_Resnet50, self).__init__()

        sk_resnet50 = SKNet(nums_class, SKBlock, [3, 4, 6, 3])
        self.stage1 = nn.Sequential(sk_resnet50.conv, sk_resnet50.maxpool)
        self.stage2 = nn.Sequential(sk_resnet50.layer1)
        self.stage3 = nn.Sequential(sk_resnet50.layer2)
        self.stage4 = nn.Sequential(sk_resnet50.layer3)
        self.stage5 = nn.Sequential(sk_resnet50.layer4)

        self.dec = Decoder(128, nums_class, 'SKConv')

        self.locally1 = LocallyAdap(768, 3, 'SKConv')
        self.locally2 = LocallyAdap(128, 2, 'SKConv')

        self.up1 = Upsample(256, 'sk_resnet50')

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
            nn.Conv2d(512, 256, 1, 2, bias=False),
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
            x_now)  # [2048, 8, 8], [1024, 16, 16], [512, 32, 32], [256, 64, 64], [64, 64, 64]
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

        # out = torch.argmax(result, dim=1, keepdim=True)
        return result


def sk_resnet101(nums_class=1000):
    return SKNet(nums_class, SKBlock, [3, 4, 23, 3])
"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 损失函数
"""
import torch
import torch.nn.functional as F
from torch import nn

from networks.vgg import Vgg16


class CACLoss:
    def __init__(self):
        super(CACLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_content = nn.MSELoss()
        self.criterion.cuda()
        self.criterion_content.cuda()
        self.vgg = Vgg16().cuda()

    def __call__(self, img_pre, img_now, result, label):
        # change attention map-based content loss (CAC Loss)
        ca_map = F.sigmoid(result[:, 0, :, :] - result[:, 1, :, :]).unsqueeze(1).expand_as(img_pre)
        img_prev_feature = self.vgg(img_pre * ca_map)
        img_now_feature = self.vgg(img_now * ca_map)
        loss_content = self.criterion_content(img_prev_feature[0], img_now_feature[0])
        # L (Θ; D) = α × LBCE (Θ; D) + LCAC (Θ; D)
        loss = loss_content + 50 * self.criterion(result, label)
        return loss

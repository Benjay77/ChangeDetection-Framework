"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 网络模型
"""
import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable as V

from networks.se_resnet50 import SE_Resnet50
from networks.sk_resnet50 import SK_Resnet50
from utils.loss import *


class MyNet:
    def __init__(self, args, eval_mode=False, pretrained=True):
        self.network = args.arch
        self.loss_function = args.loss_function
        if args.arch == 'SE_Resnet50':
            net = SE_Resnet50(nums_class=args.classes, pretrained=pretrained, strides=(1, 2, 1, 2))
        elif args.arch == 'SK_Resnet50':
            net = SK_Resnet50(nums_class=args.classes)

        self.net = net
        if args.use_multiple_GPU:
            self.net = nn.DataParallel(self.net.cuda(), device_ids=range(torch.cuda.device_count()))
            self.batch_size = torch.cuda.device_count() * args.batchsize_per_card
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device
            if torch.cuda.is_available():
                self.net = nn.DataParallel(self.net, device_ids=[0])
            self.batch_size = args.batchsize_per_card
        if args.loss_function == 'CACLoss':
            self.loss = CACLoss()
        if eval_mode:
            self.net.eval()
        else:
            self.net.train()
            self.old_lr = args.lr_init
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=args.lr_init)

    def set_input(self, img_pre, img_now, label):
        self.img_pre = img_pre
        self.img_now = img_now
        self.label = label

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        label = self.label.to(torch.float32)  
        result = self.net.forward(self.img_pre, self.img_now)
        label = torch.squeeze(label)
        label = label.long()
        loss = self.loss(self.img_pre, self.img_now, result, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def forward(self, volatile=False):
        self.img_pre = V(self.img_pre.cuda(), volatile=volatile)
        self.img_now = V(self.img_now.cuda(), volatile=volatile)
        self.label = V(self.label.cuda(), volatile=volatile)

    def save(self, path, save_model, is_best=False):
        if is_best:
            torch.save(self.net.state_dict(), path, _use_new_zipfile_serialization=False)
        else:
            torch.save(save_model, path, _use_new_zipfile_serialization=False)

    def load(self, path, is_best=False):
        if is_best:
            self.net.load_state_dict(torch.load(path))
        else:
            return torch.load(path)

    def update_lr(self, lrf, my_log, factor=False):
        if factor:
            new_lr = self.old_lr * lrf
            new_lr_rsi = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        update_lr_info = 'update learning rate: %f -> %f' % (self.old_lr, new_lr) + '\n'
        my_log.write(update_lr_info)
        print(update_lr_info)
        self.old_lr = new_lr
        return new_lr

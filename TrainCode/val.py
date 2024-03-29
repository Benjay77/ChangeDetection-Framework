"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 验证模块
"""
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional

from utils.my_net import MyNet
from torch.autograd import Variable as V


def val(opt, loader_val, model_path):
    model = MyNet(opt, eval_mode = True)
    model.net.load_state_dict(model.load(model_path)['net'])
    # model.load(model_path)
    with torch.no_grad():
        loop = tqdm(enumerate(loader_val), total = len(loader_val))
        val_epoch_loss = 0
        for i, (data_prev, data_now, label, _) in loop:
            model.set_input(data_prev, data_now, label)
            model.forward()
            label = model.label.to(torch.float32)
            result = model.net.forward(opt.img_pre, opt.img_now)
            label = torch.squeeze(model.label)
            label = label.long()
            val_loss = model.loss(model.img_pre, model.img_now, result, label)
            val_epoch_loss += val_loss.item()
            loop.set_postfix(loss = val_loss.item())
        val_epoch_loss /= len(loader_val)

        return val_epoch_loss
"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 公用函数
"""
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.exposure import match_histograms
from skimage.measure import compare_ssim
from skimage.measure.simple_metrics import compare_psnr
from torch.autograd import Variable as V


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weights.data, 1.0, 0.02)
        m.weight.data.normal_(mean = 0, std = math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Group') != -1:
        # nn.init.uniform(m.weights.data, 1.0, 0.02)
        m.weight.data.normal_(mean = 0, std = math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Img = np.uint8(np.clip(Img * 255, 0, 255))
    Img = Img.astype(np.float32) / 255.
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range = data_range)
    return PSNR / Img.shape[0]


def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Img = np.uint8(np.clip(Img * 255, 0, 255))
    Img = Img.astype(np.float32) / 255.
    Img = np.squeeze(Img)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    Iclean = np.squeeze(Iclean)

    SSIM = 0
    SSIM += compare_ssim(Iclean, Img, data_range = data_range)
    return SSIM


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k = 2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k = 2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k = 3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k = 3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def quantitative_cal(label, result):
    # result and label shape (B, 1, H, W) ?

    result = result.data.cpu().numpy().astype(np.bool)
    label = label.data.cpu().numpy().astype(np.bool)

    true_positive = result * label
    precision = true_positive.sum() / float(result.sum())
    recall = true_positive.sum() / float(label.sum())
    true_negative = (~result) * (~label)

    if float(label.sum()) == 0.0:
        F1_score = 1.0
    else:
        F1_score = 2 * (precision * recall) / (precision + recall)
    total_acc = (true_positive.sum() + true_negative.sum()) / float(label.size)
    return F1_score, total_acc


def cal_iou(result, label):
    # result and label shape (B, 1, H, W) ??
    result = result.data.cpu().numpy().astype(np.bool)
    label = label.data.cpu().numpy().astype(np.bool)

    intersection = result * label
    union = result + label

    return intersection.sum() / float(union.sum())


def subtract_abs(img_pre, img_now):
    pre = img_pre.cpu().data.numpy()
    now = img_now.cpu().data.numpy()
    result = cv2.absdiff(pre, now)
    if torch.cuda.is_available():
        result = V(torch.FloatTensor(result).cuda(), volatile = False)
    else:
        result = V(torch.FloatTensor(result), volatile = False)
    return result


# 随机直方图拉伸
def radiation_random(img_copy, img_target):
    # 先随机直方图
    img_target = match_histograms(img_target, img_copy, multichannel = True)

    max_r = np.max(img_target[:, :, 0])
    min_r = np.min(img_target[:, :, 0])
    max_g = np.max(img_target[:, :, 1])
    min_g = np.min(img_target[:, :, 1])
    max_b = np.max(img_target[:, :, 2])
    min_b = np.min(img_target[:, :, 2])
    #
    min_rr = np.random.randint(0, 70)
    max_rr = np.random.randint(min_rr * 2, 255)
    min_gg = np.random.randint(0, 70)
    max_gg = np.random.randint(min_gg * 2, 255)
    min_bb = np.random.randint(0, 70)
    max_bb = np.random.randint(min_bb * 2, 255)

    img_target[:, :, 0] = (img_target[:, :, 0] - min_r) / (max_r - min_r) * (max_rr - min_rr) + min_rr
    img_target[:, :, 1] = (img_target[:, :, 1] - min_g) / (max_g - min_g) * (max_gg - min_gg) + min_gg
    img_target[:, :, 2] = (img_target[:, :, 2] - min_b) / (max_b - min_b) * (max_bb - min_bb) + min_bb

    img_target = np.array(img_target).astype("uint8")

    return img_target


# 限制直方图均衡化
def limit_histogram_equalization(image):
    # if image.shape[2] == 3:
    #     b, g, r = cv2.split(image)
    # elif image.shape[2] == 4:
    #     b, g, r, _ = cv2.split(image)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    clahe_b = clahe.apply(b)
    clahe_g = clahe.apply(g)
    clahe_r = clahe.apply(r)
    clahe_merge = cv2.merge((clahe_r, clahe_g, clahe_b))
    return clahe_merge
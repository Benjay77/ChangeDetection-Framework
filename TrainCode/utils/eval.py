# -*- coding: utf-8 -*-
"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 评估模块
"""
import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def cal_kappa(self):
        if self.hist.sum() == 0:
            cal_kappa = 0
        else:
            po = np.diag(self.hist).sum() / self.hist.sum()
            pe = np.matmul(self.hist.sum(1), self.hist.sum(0).T) / self.hist.sum() ** 2
            if pe == 1:
                cal_kappa = 0
            else:
                cal_kappa = (po - pe) / (1 - pe)
        return cal_kappa

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength = self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis = 1) + self.hist.sum(axis = 0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # mean acc
        accuracy = np.diag(self.hist).sum() / self.hist.sum()
        precision = np.nanmean(np.diag(self.hist) / self.hist.sum(axis = 1))
        recall = np.nanmean(np.diag(self.hist) / self.hist.sum(axis = 0))
        f1_score = 2 * precision * recall / (precision + recall)
        # kappa
        kappa = self.cal_kappa()
        # freq = self.hist.sum(axis=1) / self.hist.sum()
        # fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return accuracy, precision, recall, f1_score, iou, miou, kappa  # , fwavacc


if __name__ == '__main__':
    img_type = '.png'
    mask_path = '../dataset/test/'
    predict_path = '../results/'
    pres = os.listdir(predict_path)
    masks = []
    predicts = []
    for im in pres:
        if im[-11:] == '_result' + img_type:
            mask_name = im.split('_result' + img_type)[0] + img_type
            ma_path = os.path.join(mask_path, mask_name)
            pre_path = os.path.join(predict_path, im)
            mask = cv2.imread(ma_path, 0)
            pre = cv2.imread(pre_path, 0)
            mask[mask > 0] = 1
            pre[pre > 0] = 1
            masks.append(mask)
            predicts.append(pre)

    el = IOUMetric(2)
    accuracy, precision, recall, f1_score, iou, miou, kappa = el.evaluate(predicts, masks)
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('miou: ', miou)
    print('f1_score: ', f1_score)
    print('kappa: ', kappa)
    # print('iou: ', iou)
    # print('fwavacc: ', fwavacc)
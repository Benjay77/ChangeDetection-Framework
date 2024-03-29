"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 数据处理
"""
import glob
import os
import os.path
import random

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as udata
from torchvision import transforms
from tqdm import tqdm

from utils.common_function import *


def s_normalize(img, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
    img[0, :, :] = (img[0, :, :] / 255. - mean[0]) / std[0]
    img[1, :, :] = (img[1, :, :] / 255. - mean[1]) / std[1]
    img[2, :, :] = (img[2, :, :] / 255. - mean[2]) / std[2]

    return img


def normalize(data):
    return data / 255


def data_process(args, aug_times, data_type, count):
    # 图片处理
    trans = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor(),
         transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    if data_type == 'test':
        files_i1 = glob.glob(os.path.join(args.dataset_dir, 'A', '*' + args.image_type))

        files_i2 = glob.glob(os.path.join(args.dataset_dir, 'B', '*' + args.image_type))

        files_l = glob.glob(os.path.join(args.dataset_dir, 'OUT', '*' + args.mask_type))
    else:
        files_i1 = glob.glob(os.path.join(args.dataset_dir, data_type, 'A', '*' + args.image_type))

        files_i2 = glob.glob(os.path.join(args.dataset_dir, data_type, 'B', '*' + args.image_type))

        files_l = glob.glob(os.path.join(args.dataset_dir, data_type, 'OUT', '*' + args.mask_type))
    files_i1.sort()
    files_i2.sort()
    files_l.sort()
    h5f = h5py.File(data_type + count + '.h5', 'w')
    h5f_image1 = h5f.create_group("image1")
    h5f_image2 = h5f.create_group("image2")
    h5f_label = h5f.create_group("label")
    if data_type == 'test':
        h5f_label_name = h5f.create_group("label_name")
    data_num = 0
    loop = tqdm(range(len(files_i1)))
    for i in loop:
        # img, label read
        img1 = cv2.imread(files_i1[i])
        img2 = cv2.imread(files_i2[i])
        label = cv2.imread(files_l[i])
        # radiation_random
        # copy_index = int(np.random.random() * len(files_i1))
        # img_pre_copy = cv2.imread(files_i1[copy_index])
        # img1 = radiation_random(img_pre_copy, img1)
        # copy_index = int(np.random.random() * len(files_i1))
        # img_pre_copy = cv2.imread(files_i1[copy_index])
        # img2 = radiation_random(img_pre_copy, img2)
        # img1 = limit_histogram_equalization(img1)
        # img2 = limit_histogram_equalization(img2)

        # h, w, c = img.shape
        # img1 = img[:, : h, :]
        # img2 = img[:, h :, :]

        # Image 1 part
        # Img1 = np.transpose(img1, [2, 0, 1])
        # Img1 = np.expand_dims(Img1, 0)
        # Img1 = s_normalize(np.float32(Img1))
        img1 = trans(img1)

        # Image 2 part
        # Img2 = np.transpose(img2, [2, 0, 1])
        # Img2 = np.expand_dims(Img2, 0)
        # Img2 = s_normalize(np.float32(Img2))
        img2 = trans(img2)

        # Label part
        # label = label[:, :, 0]
        # label = np.expand_dims(label, 2)
        # label = np.transpose(label, [2, 0, 1])
        # # Label = np.expand_dims(Label, 0)
        label = np.int_(normalize(label))
        # label = trans(label)
        label = np.expand_dims(label[:, :, 0], 0)

        data1 = img1
        data2 = img2
        data_label = label

        h5f_image1.create_dataset(str(data_num), data = data1)
        h5f_image2.create_dataset(str(data_num), data = data2)
        h5f_label.create_dataset(str(data_num), data = data_label)
        if data_type == 'test':
            label_name = files_l[i].split('/')[-1].split(args.mask_type)[0]
            label_name = np.array(label_name, dtype = object)
            h5f_label_name.create_dataset(str(data_num), dtype = h5py.special_dtype(vlen = str), data = label_name)
        data_num += 1
        if data_type != 'test':
            for m in range(aug_times):
                aug_int = np.random.randint(1, 8)

                data1_aug = data_augmentation(data1, aug_int)
                h5f_image1.create_dataset(str(data_num) + "_aug_%d" % (m + 1), data = data1_aug)

                data2_aug = data_augmentation(data2, aug_int)
                h5f_image2.create_dataset(str(data_num) + "_aug_%d" % (m + 1), data = data2_aug)

                datal_aug = data_augmentation(data_label, aug_int)
                h5f_label.create_dataset(str(data_num) + "_aug_%d" % (m + 1), data = datal_aug)

                data_num += 1
        loop.set_postfix(num = data_num)
    h5f.close()
    print(data_type + ' set, # samples %d\n' % data_num)


def prepare_data(args, aug_times = 0, is_train = True, count = '1'):
    if is_train:
        # train
        print('process training data')
        data_process(args, aug_times, 'train', count)
        # val
        print('\nprocess validation data')
        data_process(args, aug_times, 'val', count)
    else:
        # test
        print('\nprocess test data')
        data_process(args, aug_times, 'test', count)


class Dataset(udata.Dataset):
    def __init__(self, data_type = 'train', count = '1'):
        super(Dataset, self).__init__()
        self.data_type = data_type
        self.count = count
        if self.data_type == 'train':
            h5f = h5py.File('train' + self.count + '.h5', 'r')
        elif self.data_type == 'val':
            h5f = h5py.File('val' + self.count + '.h5', 'r')
        else:
            h5f = h5py.File('test' + self.count + '.h5', 'r')
        self.keys = list(h5f['image1'].keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.data_type == 'train':
            h5f = h5py.File('train' + self.count + '.h5', 'r')
        elif self.data_type == 'val':
            h5f = h5py.File('val' + self.count + '.h5', 'r')
        else:
            h5f = h5py.File('test' + self.count + '.h5', 'r')
        key = self.keys[index]
        data1 = np.array(h5f['image1'][key])
        data2 = np.array(h5f['image2'][key])
        label = np.array(h5f['label'][key])
        label_name = ''
        if self.data_type == 'test':
            label_name = str((h5f['label_name'][key])[()])
        h5f.close()
        return torch.as_tensor(data1), torch.as_tensor(data2), torch.as_tensor(label), label_name
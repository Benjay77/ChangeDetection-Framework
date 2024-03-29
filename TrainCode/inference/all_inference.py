# coding=utf-8
"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 推理文件
"""
import argparse
import glob
import os
import sys
import time
import warnings
import random

import cv2
import skimage.io
import torch
from tqdm import tqdm
from skimage.morphology import *

from networks import *
from common_function import *


def getargs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--image_pre_path', type = str,
                       default = '../dataset/test/A')  
    parse.add_argument('--image_now_path', type = str,
                       default = '../dataset/test/B')
    parse.add_argument('--save_path', type = str, default = '../results/')
    parse.add_argument('--arch', '-a', metavar = 'ARCH', default = 'SE_Resnet50',
                       help = 'SE_Resnet50/SK_Resnet50')
    parse.add_argument('--image_size', type = int, default = 512)
    parse.add_argument('--resolution', type = str, default = '')
    parse.add_argument("--interpret_type", type = str, default = '')
    parse.add_argument("--weight_name", type = str, default = '202108042112')
    parse.add_argument('--image_channel', type = int, default = 3)
    parse.add_argument('--overlay', type = int, default = 4096)
    parse.add_argument('--small_threshold', type = int, default = 200)
    parse.add_argument("--inference_num", type = int, default = 16)
    parse.add_argument("--threshold", type = float, default = 0.5)
    parse.add_argument('--image_type', type = str, default = '.tif')
    parse.add_argument('--label_type', type = str, default = '.tif')
    parse.add_argument('--label_flag', type = str, default = '_result')
    parse.add_argument("--weights", type = str, default = 'weights', help = 'path of weights files')
    parse.add_argument("--batchsize_per_card", type = int, default = 24)
    parse.add_argument("--kronecker_r1", nargs = '+', type = int,
                       default = [2, 4, 8])
    parse.add_argument("--kronecker_r2", nargs = '+', type = int,
                       default = [1, 3, 5])
    parse.add_argument('--use_multiple_GPU', type = bool, default = True)
    parse.add_argument('--device', default = '1', help = 'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parse.add_argument("--log_dir", default = '/logs', help = "log dir")
    return parse.parse_args()


# Data expansion
class InferenceModel:
    def __init__(self, opts):
        self.args = opts
        if self.args.arch == 'SE_Resnet50':
            net = SE_Resnet50(2, False, strides=(1, 2, 1, 2))
        self.net = net
        if self.args.use_multiple_GPU:
            self.net = torch.nn.DataParallel(self.net.cuda(), device_ids = range(torch.cuda.device_count()))
            if self.args.segmentation_detection:
                self.net_rsi = nn.DataParallel(self.net_rsi.cuda(), device_ids = range(torch.cuda.device_count()))
            self.batch_size = torch.cuda.device_count() * self.args.batchsize_per_card
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == 'cuda':
                self.net = nn.DataParallel(self.net, device_ids = [0])
                if self.args.segmentation_detection:
                    self.net_rsi = nn.DataParallel(self.net_rsi, device_ids = [0])
            else:
                device = 'cpu'
                self.net = nn.DataParallel(self.net, device_ids = [0]).to(device)
                if self.args.segmentation_detection:
                    self.net_rsi = nn.DataParallel(self.net_rsi, device_ids = [0]).to(device)
            self.batch_size = self.args.batchsize_per_card

    def test_one_img_from_path(self, img_pre, img_now, evalmode = True):
        if evalmode:
            self.net.eval()
            if self.args.segmentation_detection:
                self.net_rsi.eval()
        if self.args.overlay > 0:
            return self.crop_prediction(img_pre, img_now)
        else:
            return self.padding_prediction(img_pre, img_now)

    def crop_prediction(self, img_pre, img_now):
        img_crop = np.zeros((self.args.overlay, self.args.overlay, self.args.inference_num), dtype = np.float32, order = 'C')
        img_temp_pre = np.zeros((self.args.overlay, self.args.overlay, self.args.image_channel), dtype = np.uint8,
                                order = 'C')
        img_temp_now = np.zeros((self.args.overlay, self.args.overlay, self.args.image_channel), dtype = np.uint8,
                                order = 'C')
        # 记录不为黑色的像素的个数
        ratio = np.ones((self.args.overlay, self.args.overlay), dtype = np.float32, order = 'C')
        img_ratio = np.zeros((self.args.overlay, self.args.overlay, self.args.inference_num), dtype = np.float32,
                             order = 'C')

        for num in range(self.args.inference_num):
            if num == 0:
                result= self.predict_patch(img_pre,
                                                                    img_now, self.args.overlay,
                                                                    self.args.overlay,
                                                                    self.args.image_size)
                img_crop[0:self.args.overlay, 0:self.args.overlay, 0] = result[
                                                                        0:self.args.overlay,
                                                                        0:self.args.overlay]
            
                img_ratio[0:self.args.overlay, 0:self.args.overlay, 0] = ratio[0:self.args.overlay, 0:self.args.overlay]
            else:
                img_crop_up = random.randint(128, self.args.image_size)
                img_crop_left = random.randint(128, self.args.image_size)
                img_temp_pre[0: img_crop_up, 0:img_crop_left, :] = img_pre[0: img_crop_up, 0:img_crop_left, :]
                img_temp_now[0: img_crop_up, 0:img_crop_left, :] = img_now[0: img_crop_up, 0:img_crop_left, :]
                result= self.predict_patch(img_temp_pre,
                                                                    img_temp_now, self.args.image_size,
                                                                    self.args.image_size,
                                                                    self.args.image_size)
                img_crop[0: img_crop_up, 0:img_crop_left, num] = result[
                                                                 0: img_crop_up, 0:img_crop_left]
                img_ratio[0: img_crop_up, 0:img_crop_left, num] = ratio[0: img_crop_up, 0:img_crop_left]
                img_temp_pre[0:(self.args.overlay - img_crop_up), 0:(self.args.overlay - img_crop_left), :] = img_pre[
                                                                                                              img_crop_up:self.args.overlay,
                                                                                                              img_crop_left:self.args.overlay,
                                                                                                              :]
                img_temp_now[0:(self.args.overlay - img_crop_up), 0:(self.args.overlay - img_crop_left), :] = img_now[
                                                                                                              img_crop_up:self.args.overlay,
                                                                                                              img_crop_left:self.args.overlay,
                                                                                                              :]
                result = self.predict_patch(img_temp_pre,
                                                                    img_temp_now, self.args.overlay,
                                                                    self.args.overlay,
                                                                    self.args.image_size)
                img_crop[img_crop_up:self.args.overlay, img_crop_left:self.args.overlay, num] = result[
                                                                                                0:(
                                                                                                        self.args.overlay
                                                                                                        - img_crop_up),
                                                                                                0:(
                                                                                                        self.args.overlay
                                                                                                        - img_crop_left)]
               
                img_ratio[img_crop_up:self.args.overlay, img_crop_left:self.args.overlay,
                num] = ratio[
                       img_crop_up:self.args.overlay,
                       img_crop_left:self.args.overlay]

        img_crop_sum = img_crop.sum(axis = 2)
        img_ratio_sum = img_ratio.sum(axis = 2)
        prediction = img_crop_sum / img_ratio_sum


        # 预测
        prediction = np.where(prediction > self.args.threshold, 1, 0)
        prediction = np.array(prediction, dtype = 'uint8')
        prediction = post_process(prediction, self.args.small_threshold, 1, self.args.interpret_type)

        del img_crop, img_temp_pre, img_temp_now, ratio, img_ratio
        return prediction

    def predict_patch(self, img_pre, img_now, img_row, img_col, step):
        if self.args.interpret_type != 'water' and self.args.interpret_type != 'build':
            img_pre = limit_histogram_equalization(img_pre)
            img_now = limit_histogram_equalization(img_now)
        patch_prediction = np.zeros((img_row, img_col), dtype = np.float32, order = 'C')
        # 分块读取影像，分块预测
        # 逐列提取影像
        for c in range(0, img_col, step):
            # 截取部分影像
            img_part_pre = img_pre[0: img_row, c: c + step, :]
            img_part_now = img_now[0: img_row, c: c + step, :]
            img_part_pre_sum = img_part_pre.sum(axis = 2)
            img_part_now_sum = img_part_now.sum(axis = 2)
            if img_part_pre_sum.max() == 0 and img_part_now_sum.max() == 0:
                patch_prediction[0: img_row, c: c + step] = 0
            else:
                # 分解出三个波段，并且加一个轴
                img_part_pre_r = np.expand_dims(img_part_pre[:, :, 0], 0)
                img_part_pre_g = np.expand_dims(img_part_pre[:, :, 1], 0)
                img_part_pre_b = np.expand_dims(img_part_pre[:, :, 2], 0)
                img_part_now_r = np.expand_dims(img_part_now[:, :, 0], 0)
                img_part_now_g = np.expand_dims(img_part_now[:, :, 1], 0)
                img_part_now_b = np.expand_dims(img_part_now[:, :, 2], 0)
                # 整成多个波段
                img_part_pre_r_16 = img_part_pre_r.reshape((img_col // step, step, step), order = 'C')
                img_part_pre_g_16 = img_part_pre_g.reshape((img_col // step, step, step), order = 'C')
                img_part_pre_b_16 = img_part_pre_b.reshape((img_col // step, step, step), order = 'C')
                img_part_now_r_16 = img_part_now_r.reshape((img_col // step, step, step), order = 'C')
                img_part_now_g_16 = img_part_now_g.reshape((img_col // step, step, step), order = 'C')
                img_part_now_b_16 = img_part_now_b.reshape((img_col // step, step, step), order = 'C')
                # 合并各个波段
                img_part_pre_r_16 = np.expand_dims(img_part_pre_r_16, 1)
                img_part_pre_g_16 = np.expand_dims(img_part_pre_g_16, 1)
                img_part_pre_b_16 = np.expand_dims(img_part_pre_b_16, 1)
                img_part_pre_rgb = np.concatenate((img_part_pre_r_16, img_part_pre_g_16, img_part_pre_b_16), axis = 1)
                img_part_pre_rgb = img_part_pre_rgb.astype(np.float32) / 255.0 * 3.2 - 1.6
                img_part_now_r_16 = np.expand_dims(img_part_now_r_16, 1)
                img_part_now_g_16 = np.expand_dims(img_part_now_g_16, 1)
                img_part_now_b_16 = np.expand_dims(img_part_now_b_16, 1)
                img_part_now_rgb = np.concatenate((img_part_now_r_16, img_part_now_g_16, img_part_now_b_16), axis = 1)
                img_part_now_rgb = img_part_now_rgb.astype(np.float32) / 255.0 * 3.2 - 1.6
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if device.type == 'cuda':
                    img_part_pre_rgb = V(torch.Tensor(img_part_pre_rgb).cuda())
                    img_part_now_rgb = V(torch.Tensor(img_part_now_rgb).cuda())
                else:
                    img_part_pre_rgb = V(torch.Tensor(img_part_pre_rgb))
                    img_part_now_rgb = V(torch.Tensor(img_part_now_rgb))

                # 开始预测
                with torch.no_grad():
                    temp = self.net.forward(img_part_pre_rgb, img_part_now_rgb).reshape(1, img_row,
                                                                                        step).squeeze().cpu(
                    ).data.numpy()

                    patch_prediction[0: img_row, c: c + step] = temp

        return patch_prediction

    def padding_prediction(self, img_pre, img_now):
        if self.args.interpret_type != 'water' and self.args.interpret_type != 'build':
            img_pre = limit_histogram_equalization(img_pre)
            img_now = limit_histogram_equalization(img_now)
        img5_pre, img6_pre = self.padding_process(img_pre)
        img5_now, img6_now = self.padding_process(img_now)

        maska = self.net.forward(img5_pre, img5_now).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img6_pre, img6_now).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        mask3 = np.where(mask3 > self.args.threshold, 1, 0)
        mask3 = np.array(mask3, dtype = 'uint8')
        mask3 = post_process(mask3, self.args.small_threshold, 1, self.args.interpret_type)
        return mask3

    def padding_process(self, img):
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            img5 = V(torch.Tensor(img5).cuda())
            img6 = V(torch.Tensor(img6).cuda())
        else:
            img5 = V(torch.Tensor(img5))
            img6 = V(torch.Tensor(img6))
        return img5, img6

    def load(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            self.net.load_state_dict(torch.load(model_path))
        else:
            self.net.load_state_dict(torch.load(model_path, 'cpu'))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = getargs()
    path = os.path.abspath(sys.argv[0])
    (file_path, temp_filename) = os.path.split(path)

    ##########################
    # img_dir = sys.argv[1]
    # save_dir = sys.argv[2]
    # img_path = img_dir.encode("utf-8").decode("utf-8")
    # save_dir = save_dir.encode("utf-8").decode("utf-8")
    print('\n>>>>>>>>>>>>>>>>>>>>>>>> begin load model >>>>>>>>>>>>>>>>>>>>>>>>>>>')
    time_model_begin = time.time()
    solver = InferenceModel(args)
    model_path = os.path.join(file_path, args.weights, args.resolution, args.interpret_type, args.interpret_type +
                              args.weight_name,
                              args.interpret_type + args.weight_name + '_best.pth')
    if not os.path.exists(model_path):
        print("**************** model not exists *********************************")
    solver.load(model_path)
    print(f'model load cost{time.time() - time_model_begin}s')
    print(">>>>>>>>>>>>>>>>>>>>> model load succeed >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    img_pre_path = args.image_pre_path.encode("utf-8").decode("utf-8")
    img_now_path = args.image_now_path.encode("utf-8").decode("utf-8")
    save_path = os.path.join(args.save_path, args.resolution, args.interpret_type,
                             args.interpret_type + args.weight_name).encode("utf-8").decode("utf-8")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files_pre_path = glob.glob(os.path.join(img_pre_path, '*' + args.image_type))
    files_now_path = glob.glob(os.path.join(img_now_path, '*' + args.image_type))
    time_detect_begin = time.time()
    loop = tqdm(enumerate(files_pre_path), total = len(files_pre_path))
    for index, img_pre in loop:
        if args.overlay > 0 and args.inference_num > 0:
            img_now = files_now_path[index]
            img_array_pre, img_info_pre = read_tiff(img_pre)
            img_array_now, _ = read_tiff(img_now)

            rows = img_array_pre.shape[0]
            cols = img_array_pre.shape[1]

            if rows % args.overlay == 0:
                padding_row = rows
            else:
                padding_row = (rows // args.overlay + 1) * args.overlay
            if cols % args.overlay == 0:
                padding_col = cols
            else:
                padding_col = (cols // args.overlay + 1) * args.overlay
            img_buffer_pre = np.zeros((padding_row, padding_col, args.image_channel), dtype = int, order = 'C')
            img_buffer_now = np.zeros((padding_row, padding_col, args.image_channel), dtype = int, order = 'C')
            img_buffer_pre[0:rows, 0:cols, :] = img_array_pre[:, :, :]
            img_buffer_now[0:rows, 0:cols, :] = img_array_now[:, :, :]
            prediction_buffer = np.zeros((padding_row, padding_col), dtype = int, order = 'C')
            for row in range(padding_row // args.overlay):
                for col in range(padding_col // args.overlay):
                    loop.set_postfix(image = os.path.basename(img_pre), row = row + 1, rows = padding_row // args.overlay,
                                     col = col + 1, cols = padding_col // args.overlay)
                    down = (row + 1) * args.overlay
                    right = (col + 1) * args.overlay
                    up = down - args.overlay
                    left = right - args.overlay
                    img_current_pre = img_buffer_pre[up:down, left:right:]
                    img_current_now = img_buffer_now[up:down, left:right:]
                    img_current_pre_sum = img_current_pre.sum(axis = 2)
                    img_current_pre_sum[img_current_pre_sum > 0] = 1
                    img_current_now_sum = img_current_now.sum(axis = 2)
                    img_current_now_sum[img_current_now_sum > 0] = 1
                    if img_current_pre.max() == 0 and img_current_now.max() == 0:
                        prediction_buffer[up:down, left:right] = 0
                    else:
                        temp_prediction = solver.test_one_img_from_path(img_current_pre, img_current_now)
                        temp_prediction = np.array(temp_prediction, dtype = 'uint8')
                        temp_prediction = post_process(temp_prediction, args.small_threshold, 1, args.interpret_type)
                        prediction_buffer[up:down,
                        left:right] = temp_prediction * img_current_now_sum * img_current_pre_sum
            prediction_buffer = prediction_buffer.astype(np.uint8)
            prediction_buffer = post_process(prediction_buffer, args.small_threshold, 1, args.interpret_type)
            prediction_buffer = prediction_buffer.astype(np.uint8) * 255
            mask_img = prediction_buffer[0:rows, 0:cols]
            mask_img = np.squeeze(mask_img)
            del img_buffer_pre, img_buffer_now, prediction_buffer
        else:
            image_pre = cv2.imdecode(np.fromfile(img_pre, dtype=np.uint8), 1)
            img_now = files_now_path[index]
            image_now = cv2.imdecode(np.fromfile(img_now, dtype=np.uint8), 1)
            h, w, channel = image_pre.shape
            if h % args.image_size == 0:
                padding_h = h
            else:
                padding_h = (h // args.image_size + 1) * args.image_size
            if w % args.image_size == 0:
                padding_w = w
            else:
                padding_w = (w // args.image_size + 1) * args.image_size
            padding_img_pre = np.zeros((padding_h, padding_w, channel), dtype = np.uint8)
            padding_img_pre[0:h, 0:w, :] = image_pre[:, :, :]
            padding_img_now = np.zeros((padding_h, padding_w, channel), dtype = np.uint8)
            padding_img_now[0:h, 0:w, :] = image_now[:, :, :]
            mask_whole = np.zeros((padding_h, padding_w), dtype = np.uint8)
            for i in range(padding_h // args.image_size):
                for j in range(padding_w // args.image_size):
                    loop.set_postfix(image = img_pre.split('/')[-1], h = i + 1,
                                     padding_h = padding_h // args.image_size,
                                     padding_w = padding_w // args.image_size, w = j + 1)
                    crop_pre = padding_img_pre[i * args.image_size:i * args.image_size + args.image_size,
                               j * args.image_size:j * args.image_size + args.image_size, :channel]
                    crop_now = padding_img_now[i * args.image_size:i * args.image_size + args.image_size,
                               j * args.image_size:j * args.image_size + args.image_size, :channel]
                    if crop_pre.max() == 0 and crop_now.max() == 0:
                        pred_img[:, :] = 0
                    else:
                        pred_img = solver.test_one_img_from_path(crop_pre, crop_now)
                    pred_img = pred_img.astype(np.uint8)

                    mask_whole[i * args.image_size:i * args.image_size + args.image_size,
                    j * args.image_size:j * args.image_size + args.image_size] = pred_img[:, :]

            mask_img = mask_whole[0:h, 0:w]
            mask_img = np.array(mask_img, dtype = 'uint8')
            mask_img = post_process(mask_img, args.small_threshold, 1, args.interpret_type)
            mask_img = mask_img.astype(np.uint8) * 255

        img_name = (os.path.basename(img_now)).split(args.image_type)[0]
        raster_path = os.path.join(save_path, img_name + args.label_flag + args.label_type)
        if args.label_type == '.tif' or args.label_type == 'tiff':
            write_tiff(im_data = mask_img, im_geotrans = img_info_pre['geotrans'],
                       im_geosrs = img_info_pre['geosrs'],
                       path_out = raster_path)
            # C = ResetCoord(name, raster_path)
            # C.assign_spatial_reference_by_file()  # 添加空间参考系
            shp_path = os.path.join(save_path, img_name + '.shp')
            ShapeFile(raster_path, shp_path).create_shapefile()
        else:
            cv2.imencode(args.image_type, mask_img)[1].tofile(raster_path)
        # cv2.imwrite(raster_path, mask_img)
    print(f'Detect cost{time.time() - time_detect_begin}s')
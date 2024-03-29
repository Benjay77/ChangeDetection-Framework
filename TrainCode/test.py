"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 测试模块
"""
import argparse
import os
import time

import cv2
import skimage.io
import torch
from skimage.morphology import *
from torch.autograd import Variable
from tqdm import tqdm

from utils.dataset import prepare_data, Dataset
from utils.my_net import MyNet
from utils.common_function import *


def getargs():
    parser = argparse.ArgumentParser(description="Change_Detection_Test")
    parser.add_argument('--segmentation_detection', type=bool, default=True)
    parser.add_argument("--preprocess", type=bool, default=True
                        , help='run prepare_data or not')
    parser.add_argument("--logdir", type=str, default="log", help='path of log files')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='',
                        help='SE_Resnet50/SK_Resnet50')
    parser.add_argument('--loss_function', type=str, default='CACLoss', help='CACLoss')
    parser.add_argument("--kronecker_r1", nargs='+', type=int,
                        default=[10, 20, 30])
    parser.add_argument("--kronecker_r2", nargs='+', type=int,
                        default=[7, 15, 25])
    parser.add_argument("--image_size", type=int, default=256, help='size of image')
    parser.add_argument('--image_type', type=str, default='.tif')
    parser.add_argument('--mask_type', type=str, default='.tif')
    parser.add_argument("--weight_name", type=str, default='202109151535')
    parser.add_argument("--weights", type=str, default="./weights/", help='path of weights files')
    parser.add_argument("--result", type=str, default="./results/", help='path of result files')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/test')
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--use_mutiple_GPU', type=bool, default=True)
    parser.add_argument("--batchsize_per_card", type=int, default=256)
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args


def main():
    # Build model
    time_load_model = time.time()
    print('Loading model ...')
    model = MyNet(args, eval_mode=True, pretrained=True)

    model_path = os.path.join(args.weights, args.weight_name, args.weight_name + '_best.pth')
    model.load(model_path)
    print(f'Loading model cost{time.time() - time_load_model}s\n')

    # load data info
    print('Loading data info ...')
    dataset_test = Dataset('test')
    print(f'dataset_test:{len(dataset_test)}\n')

    if not os.path.exists(args.result + args.weight_name):
        os.makedirs(args.result + args.weight_name)

    loop = tqdm(range(len(dataset_test)))
    for k in loop:
        img_prev_test, img_now_test, _, label_name = dataset_test[k]
        img_prev_test = torch.unsqueeze(img_prev_test, dim=0)
        img_now_test = torch.unsqueeze(img_now_test, dim=0)

        with torch.no_grad():
            img_prev_test, img_now_test = Variable(img_prev_test.cuda()), Variable(
                img_now_test.cuda())
            result_test = model.net.forward(img_prev_test, img_now_test)
            if not args.segmentation_detection:
                result_test = torch.argmax(result_test, dim=1, keepdim=True)
            save_result = (result_test * 255).cpu().numpy().astype(np.uint8)
        # result_test = np.where(result_test > 0.5, 1, 0)
        # result_test = np.array(result_test, dtype='uint8')
        # result_test = binary_erosion(result_test)
        # result_test = binary_dilation(result_test)
        # result_test = binary_opening(result_test)
        # result_test = binary_closing(result_test)
        # result_test = remove_small_holes(result_test, 256)
        # result_test = remove_small_objects(result_test, 256)
        # save_result = result_test * 255
        save_result[save_result > 0] = 255
        save_result = np.squeeze(save_result)
        # save_result = np.transpose(save_result)

        # skimage.io.imsave(os.path.join(args.result, args.weight_name, label_name + '_result'+args.mask_type), save_result)
        cv2.imwrite(os.path.join(args.result, args.weight_name, label_name + '_result' + args.mask_type),
                    save_result)  # jpg
        loop.set_postfix(
            save_result=os.path.join(args.result, args.weight_name, label_name + '_result' + args.mask_type))


if __name__ == "__main__":
    args = getargs()
    if args.preprocess:
        prepare_data(args, False)
    time_detect_begin = time.time()
    main()
    print(f'Detect cost{time.time() - time_detect_begin}s')
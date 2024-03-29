"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 公用函数
"""
import math

import cv2
import gdal
import ogr
import osr
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
from skimage.measure import compare_ssim
from torch.autograd import Variable as V


# 限制直方图均衡化
def limit_histogram_equalization(image):
    image = np.array(image, dtype='uint8')
    # if image.shape[2] == 3:
    #     b, g, r = cv2.split(image)
    # elif image.shape[2] == 4:
    #     b, g, r, _ = cv2.split(image)
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_b = clahe.apply(b)
    clahe_g = clahe.apply(g)
    clahe_r = clahe.apply(r)
    clahe_merge = cv2.merge((clahe_r, clahe_g, clahe_b))
    return clahe_merge


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weights.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Group') != -1:
        # nn.init.uniform(m.weights.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Img = np.uint8(np.clip(Img * 255, 0, 255))
    Img = Img.astype(np.float32) / 255.
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def batch_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Img = np.uint8(np.clip(Img * 255, 0, 255))
    Img = Img.astype(np.float32) / 255.
    Img = np.squeeze(Img)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    Iclean = np.squeeze(Iclean)

    SSIM = 0
    SSIM += compare_ssim(Iclean, Img, data_range=data_range)
    return SSIM


# 填充空洞
def fill_hole(im_in, fill_value):
    # im_floodfill = im_in.copy()
    # h, w = im_in.shape[:2]
    # mask = np.zeros((h + 2, w + 2), np.uint8)
    # cv2.floodFill(im_floodfill, mask, (2, 2), (h, w), fill_value)
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # im_out = cv2.bitwise_or(im_in, im_floodfill_inv)
    # im_out = im_in | im_floodfill_inv
    contours, hierarch = cv2.findContours(im_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im_out = np.zeros(im_in.shape, np.uint8)
    for i in range(len(contours)):
        cnt = contours[i]
        cv2.fillPoly(im_out, [cnt], fill_value)
    return im_out


# 去除小面积区域
def remove_small(img, threshold, fill_value):
    # _,binary = cv2.threshold(img,0.1,1,cv2.THRESH_BINARY)
    contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv_contours = []
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if area < threshold:
            cv_contours.append(contours[i])
    cv2.fillPoly(img, cv_contours, fill_value)
    return img


# 膨胀
def dilation(image, count=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # dilate = cv2.dilate(image, kernel, iterations=1)
    image_dilate = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, count)
    return image_dilate


# 腐蚀
def erode(image, count=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # erode = cv2.erode(image, kernel, iterations=1)
    image_erode = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, count)
    return image_erode


# 形态学梯度
def edge(image):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, se)
    return img_grad


# 开运算
def open_operation(image, count=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_open = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, count)
    return image_open


# 闭运算
def close_operation(image, count=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, count)
    return image_close


def batched_mask_to_box(masks):
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        masks = V(torch.FloatTensor(masks).cuda(), volatile=False)
    else:
        masks = V(torch.FloatTensor(masks), volatile=False)
    # torch.max below raises an error on empty inputs, just skip in this case

    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=device.type)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    masks = masks > 0
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=device.type)[None, :]  # , device = in_height.device
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=device.type)[None, :]  # , device = in_width.device
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]
    out = out.cpu().data.numpy()
    return out


# 后处理
def post_process(prediction, small_threshold, fill_value, interpret_type):
    # 剔除噪音
    prediction = remove_small(prediction, small_threshold, 0)
    if interpret_type != 'build':
        prediction = erode(prediction)
        prediction = dilation(prediction)
        prediction = open_operation(prediction, 3)
        prediction = close_operation(prediction, 3)
        # else:
    batched_mask_to_box(prediction)
    prediction = remove_small(prediction, small_threshold, 0)
    _, prediction = cv2.connectedComponentsWithAlgorithm(prediction, 8, cv2.CV_32S, cv2.CCL_SPAGHETTI)
    prediction = np.where(prediction > 0, fill_value, 0)
    prediction = prediction.astype(np.uint8)
    prediction = fill_hole(prediction, fill_value)
    prediction = prediction.astype(np.uint8)
    return prediction


#  Add geographic information
class ResetCoord:
    def __init__(self, src_tif, dataset):
        self.src_tif = src_tif
        self.dataset = dataset

    def assign_spatial_reference_by_file(self):
        src_ds = gdal.Open(self.src_tif, gdal.GA_ReadOnly)  # gdal.GA_ReadOnly
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjectionRef())
        geo_transform = src_ds.GetGeoTransform()

        # sr = osr.SpatialReference()  # 创建空间参考
        # sr.ImportFromEPSG(4326)  # 定义地理坐标系WGS1984

        dst_ds = gdal.Open(self.dataset, gdal.GA_Update)
        dst_ds.SetProjection(srs.ExportToWkt())
        dst_ds.SetGeoTransform(geo_transform)
        del dst_ds, src_ds


# mask生成shp
class ShapeFile:
    def __init__(self, src_path, dst_path):
        self.src_path = src_path
        self.dst_path = dst_path

    def create_shapefile(self):
        src_file = gdal.Open(self.src_path, gdal.GA_ReadOnly)
        driver = ogr.GetDriverByName("ESRI Shapefile")  # "ESRI Shapefile"
        dst_ds = driver.CreateDataSource(self.dst_path)  # 创建数据源
        pus = src_file.GetProjectionRef()
        spatial_ref = osr.SpatialReference(pus)

        layer = dst_ds.CreateLayer("DstLayer", spatial_ref, ogr.wkbPolygon)

        field = ogr.FieldDefn("value", ogr.OFTInteger)
        layer.CreateField(field)
        src_band = src_file.GetRasterBand(1)

        gdal.Polygonize(src_band, None, layer, 0)
        del dst_ds


def read_tiff(path_in):
    """
    return:
        img: numpy array, exent: tuple, (x_min, x_max, y_min, y_max)
        proj info, and dimentions: (row, col, band)
    """
    rs_data = gdal.Open(path_in)
    im_col = rs_data.RasterXSize
    im_row = rs_data.RasterYSize
    im_bands = rs_data.RasterCount
    img_array = rs_data.ReadAsArray(0, 0, im_col, im_row).astype(np.uint8)
    if im_bands > 1:
        img_array = img_array[0:3, :, :]
        img_array = np.transpose(img_array, (1, 2, 0))

    im_geotrans = rs_data.GetGeoTransform()
    im_proj = rs_data.GetProjection()
    left = im_geotrans[0]
    up = im_geotrans[3]
    right = left + im_geotrans[1] * im_col + im_geotrans[2] * im_row
    bottom = up + im_geotrans[5] * im_row + im_geotrans[4] * im_col
    extent = (left, right, bottom, up)
    espg_code = osr.SpatialReference(wkt=im_proj).GetAttrValue('AUTHORITY', 1)

    img_info = {'geoextent': extent, 'geotrans': im_geotrans,
                'geosrs': espg_code, 'row': im_row, 'col': im_col,
                'bands': im_bands}
    return img_array, img_info


def write_tiff(im_data, im_geotrans, im_geosrs, path_out):
    """
    input:
        im_data: tow dimentions (order: row, col),or three dimentions (order: row, col, band)
        im_geosrs: espg code correspond to image spatial reference system.
    """
    im_data = np.squeeze(im_data)
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) >= 3:
        im_data = np.transpose(im_data, (2, 0, 1))
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path_out, im_width, im_height, im_bands, datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  #
        dataset.SetProjection("EPSG:" + str(im_geosrs))  #
    if im_bands > 1:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset
    else:
        dataset.GetRasterBand(1).WriteArray(im_data)
        del dataset


# 计算图像尺寸归一化参数
def compute_image_normalization(width, height):
    len_width = len(str(width))
    len_height = len(str(height))
    if len_width > 0 and len_height > 0:
        if len_width > len_height:
            result = (int(str(width)[0]) + 1) * math.pow(10, len_width - 1)
        elif len_height > len_width:
            result = (int(str(height)[0]) + 1) * math.pow(10, len_height - 1)
        else:
            if width >= height:
                result = (int(str(width)[0]) + 1) * math.pow(10, len_width - 1)
            else:
                result = (int(str(height)[0]) + 1) * math.pow(10, len_height - 1)
    else:
        result = 0
    return result


def style_transfer(source_image, target_image):
    h, w, c = source_image.shape
    out = []
    # if c==1:
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:, :, i])
        source_image_f_shift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:, :, i])
        target_image_f_shift = np.fft.fftshift(target_image_f)

        change_length = 1
        source_image_f_shift[int(h / 2) - change_length:int(h / 2) + change_length,
        int(h / 2) - change_length:int(h / 2) + change_length] = \
            target_image_f_shift[int(h / 2) - change_length:int(h / 2) + change_length,
            int(h / 2) - change_length:int(h / 2) + change_length]

        source_image_i_fshift = np.fft.ifftshift(source_image_f_shift)
        source_image_if = np.fft.ifft2(source_image_i_fshift)
        source_image_if = np.abs(source_image_if)

        source_image_if[source_image_if > 255] = np.max(source_image[:, :, i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1, 0).swapaxes(1, 2)

    # # 结果中含有>255的值,拉伸或者强制=255效果都不好,但是cv2.imwrite再read效果好
    # # 有时间探究一下原因
    # # 生成数字+字母
    # token = string.ascii_letters + string.digits
    # # 随机选择指定长度随机码
    # token = random.sample(token,15)
    # token_str = ''.join(token)
    # temp_path = "temp_{}.png".format(token_str)
    # if not os.path.exists(temp_path): cv2.imwrite(temp_path, out)
    # out = cv2.imread(temp_path)
    # if os.path.exists(temp_path): os.remove(temp_path)
    out = out.astype(np.uint8)
    return out


# 小图膨胀
class ImagePatch:
    def __init__(self, img, patch_size, edge_overlay):
        self.patch_size = patch_size
        self.edge_overlay = edge_overlay
        self.img = img[:, :, np.newaxis] if len(img.shape) == 2 else img
        self.img_row = img.shape[0]
        self.img_col = img.shape[1]

    def to_patch(self):
        patch_list = []
        start_list = []
        patch_step = self.patch_size - self.edge_overlay
        img_expand = np.pad(self.img, ((self.edge_overlay, patch_step),
                                       (self.edge_overlay, patch_step), (0, 0)), 'constant')
        img_patch_row = (img_expand.shape[0] - self.edge_overlay) // patch_step
        img_patch_col = (img_expand.shape[1] - self.edge_overlay) // patch_step
        for i in range(img_patch_row):
            for j in range(img_patch_col):
                patch_list.append(img_expand[i * patch_step:i * patch_step + self.patch_size,
                                  j * patch_step:j * patch_step + self.patch_size, :])
                start_list.append([i * patch_step - self.edge_overlay, j * patch_step - self.edge_overlay])
        return patch_list, start_list, img_patch_row, img_patch_col

    def higher_patch_crop(self, higher_patch_size, start_list):
        higher_patch_list = []
        radius_bias = higher_patch_size // 2 - self.patch_size // 2
        patch_step = self.patch_size - self.edge_overlay
        img_expand = np.pad(self.img, ((self.edge_overlay, patch_step), (self.edge_overlay, patch_step), (0, 0)),
                            'constant')
        img_expand_higher = np.pad(img_expand, ((radius_bias, radius_bias), (radius_bias, radius_bias), (0, 0)),
                                   'constant')
        start_list_new = list(np.array(start_list) + self.edge_overlay + radius_bias)
        for start_i in start_list_new:
            higher_row_start, higher_col_start = start_i[0] - radius_bias, start_i[1] - radius_bias
            higher_patch = img_expand_higher[higher_row_start:higher_row_start + higher_patch_size,
                           higher_col_start: higher_col_start + higher_patch_size, :]

            higher_patch_list.append(higher_patch)
        return higher_patch_list

    def to_image(self, patch_list, img_patch_row, img_patch_col):
        patch_list = [
            patch[self.edge_overlay // 2:-self.edge_overlay // 2, self.edge_overlay // 2:-self.edge_overlay // 2]
            for patch in patch_list]
        patch_list = [np.hstack((patch_list[i * img_patch_col:i * img_patch_col + img_patch_col]))
                      for i in range(img_patch_row)]
        img_array = np.vstack(patch_list)
        img_array = img_array[self.edge_overlay // 2:self.img_row + self.edge_overlay // 2,
                    self.edge_overlay // 2:self.img_col + self.edge_overlay // 2]

        return img_array

import cv2
import numpy
import torch
import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -%(levelname)s - %(message)s')


def normalize(x: numpy.ndarray, dsize):
    '''
    归一化和重新维度排序
    :param dsize:
    :param x:
    :return:
    '''
    ret = []
    for i in range(x.shape[0]):
        img = x[i, :, :, :]
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
        img = numpy.reshape(img, (img.shape[0], img.shape[1], 1))  # 灰度确实一个通道，标准化
        ret.append(img / 255.0)
    ret = numpy.array(ret)
    ret = torch.tensor(ret, dtype=torch.float32)
    ret = torch.permute(ret, [0, 3, 1, 2])
    logging.debug("PreProcess finished")
    return ret


def test_normalize(image_group, dsize):
    ret = []
    for image in image_group:
        img = cv2.resize(image, dsize, interpolation=cv2.INTER_CUBIC)
        img = numpy.reshape(img, (img.shape[0], img.shape[1], 1))  # 灰度确
        img_float32 = img / 255.0
        ret.append(img_float32)
    ret_numpy = numpy.array(ret)
    ret = torch.tensor(ret_numpy, dtype=torch.float32, device=torch.device('cuda'))
    ret_norm = torch.permute(ret, [0, 3, 1, 2])
    return ret_norm


def reverse_data(x: torch.tensor):
    '''
    反转模型输出为单张图片
    :param x:
    :return:
    '''
    ret = []
    for img in x:
        numpy_image = img.cpu().numpy().transpose((1, 2, 0))
        img_uint8 = (numpy_image * 255.0).astype(numpy.uint8)
        ret.append(img_uint8)
    return numpy.array(ret)


def data_to_image(data: numpy):
    '''

    :param data: (3,1,rows,cols)
    :return: BGR image
    '''
    batch, rows, cols, channels = data.shape
    image = cv2.merge([data[0,:,:,:],data[1,:,:,:],data[2,:,:,:]])
    return image

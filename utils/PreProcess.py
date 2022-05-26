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


def test_normalize(image, dsize):
    ret = []
    img = cv2.resize(image, dsize, interpolation=cv2.INTER_CUBIC)
    img = numpy.reshape(img, (img.shape[0], img.shape[1], 1))  # 灰度确
    ret.append(img / 255.0)
    ret = numpy.array(ret)
    ret = torch.tensor(ret, dtype=torch.float32)
    ret = torch.permute(ret, [0, 3, 1, 2])
    return ret


def reverse_data(x: torch.tensor):
    '''
    反转模型输出为单张图片
    :param x:
    :return:
    '''
    numpy_image = x.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    image_int = (numpy_image * 255.0).astype(int)
    return image_int

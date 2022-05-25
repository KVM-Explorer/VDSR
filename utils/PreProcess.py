import cv2
import torch


def normalize(x: torch.tensor, dsize):
    '''
    归一化和重新维度排序
    :param dsize:
    :param x:
    :return:
    '''
    ret = []
    for i in range(x.size(0)):
        img = x[i, :, :, :]
        cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
        ret.append(img)
    ret = ret / 255.0
    ret = torch.permute(ret, [0, 3, 1, 2])
    print(f"- PreProcess finished")
    return torch.tensor(ret, dtype=torch.float32)

import torch


def normalize(x:torch.tensor):
    '''
    归一化和重新维度排序
    :param x:
    :return:
    '''
    x = x / 255.0
    ret = torch.permute(x,[0,3,1,2])
    return ret

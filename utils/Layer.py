import torch
import torch.nn.functional as F

class Conv():
    def __init__(self):

    def forward(self,x):

    def backward(self,loss):


class Relu():
    def __init__(self):
        self.x = None

    def forward(self,x):
        self.x = x
        return torch.maximum(x, 0)

    def backward(self, delta):
        ret = torch.clamp(delta,min=0)    # 截断小于0的值
        return ret

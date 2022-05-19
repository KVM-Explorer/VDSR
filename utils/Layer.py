import torch
import torch.nn.functional as F

class Conv():
    def __init__(self,kernel ,padding = (0,0),stride = (1,1),bias=0):
        '''
        初始化卷积层
        :param kernel: (num,channels,row,cols) channels 对应上一层通道数 num对应下一层通道数
        :param padding:
        :param stride:
        '''
        self.padding = padding
        self.stride = stride
        self.kernel = torch.rand(*kernel)
        self.bias = bias
        self.gradient_w = torch.zeros(* kernel)
        self.gradient_bias = torch.zeros(1)

    def forward(self,x):
        '''
        多通道前向传播
        :param x:  （batch,width,height,channels)
        :return:
        '''
        input = F.pad(x, [1,1,1,1],
              mode='constant',
              value=0)
        return input

    # def backward(self,loss):


class Relu():
    def __init__(self):
        self.x = None

    def forward(self,x):
        self.x = x
        return torch.maximum(x, 0)

    def backward(self, delta):
        ret = torch.clamp(delta,min=0)    # 截断小于0的值
        return ret

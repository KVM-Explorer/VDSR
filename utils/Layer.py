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
        self.bias = bias # Todo 每个通道对应bias
        self.gradient_w = torch.zeros(* kernel)
        self.gradient_bias = torch.zeros(1)

    def forward(self,x:torch.tensor):
        '''
        多通道前向传播
        :param x:  （batch,width,height,channels)
        :return:
        '''
        output = torch.zeros(*x.Size())
        input = F.pad(x, [self.padding[0],self.padding[0],
                          self.padding[1],self.padding[1]],mode='constant',value=0) # padding 先后顺序不影响

        rows = input.size()[1]
        cols = input.size()[2]
        kernel_row = self.kernel.Size()[2]
        kernel_col = self.kernel.Size()[3]
        for batch in range(x[0]):
            for num in range(self.kernel[0]):
                for i in range(rows-kernel_row+1):
                    for j in range(cols - kernel_col + 1):
                        output[batch,i,j,:] = torch.sum(
                            # Kernel (num,channels,row,cols)
                            input[batch,i+kernel_row,j+kernel_col,:] * self.kernel[num,:,:,:]
                            + self.bias[num]
                        )
        return output

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

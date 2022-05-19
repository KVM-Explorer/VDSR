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
        self.bias = torch.ones(kernel[0])*bias
        self.gradient_w = torch.zeros(* kernel)
        self.gradient_bias = torch.zeros(1)

    def forward(self,x:torch.tensor):
        '''
        多通道前向传播
        :param x:  （batch,channels,rows,cols)
        :return:
        '''
        output = torch.zeros(*x.size())
        input = F.pad(x, [self.padding[0],self.padding[0],
                          self.padding[1],self.padding[1]],mode='constant',value=0) # padding 先后顺序不影响

        rows = input.size()[2]
        cols = input.size()[3]
        kernel_row = self.kernel.size()[2]
        kernel_col = self.kernel.size()[3]

        for batch in range(x.size()[0]):
            for num in range(self.kernel.size()[0]):
                for i in range(rows-kernel_row+1):
                    for j in range(cols - kernel_col+1):
                        output[batch,:,i,j] = torch.sum(
                            # Kernel (num,channels,row,cols)
                            input[batch,:,i:i+kernel_row,j:j+kernel_col] * self.kernel[num,:,:,:]
                            + self.bias[num]
                        )
        return output

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

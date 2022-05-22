import torch
import torch.nn.functional as F


class Conv():
    def __init__(self, kernel, padding=(0, 0), bias=0,):
        '''
        初始化卷积层
        :param kernel: (num,channels,row,cols) channels 对应上一层通道数 num对应下一层通道数
        :param padding:
        :param stride:
        '''
        self.padding = padding
        self.kernel = torch.rand(*kernel)
        self.bias = torch.ones(kernel[0]) * bias
        self.gradient_w = torch.zeros(*kernel)  # 指定参数尺寸
        self.gradient_bias = torch.zeros(self.bias.size())

    def forward(self, x: torch.tensor):
        '''
        多通道前向传播
        :param x:  （batch,channels,rows,cols)
        :return:
        '''
        # Todo add stride
        output = torch.zeros(*x.size())
        padding_mat = F.pad(x, [self.padding[0], self.padding[0],
                                self.padding[1], self.padding[1]], mode='constant', value=0)  # padding 先后顺序不影响

        rows = padding_mat.size()[2]
        cols = padding_mat.size()[3]
        kernel_row = self.kernel.size()[2]
        kernel_col = self.kernel.size()[3]

        for batch in range(x.size()[0]):
            for num in range(self.kernel.size()[0]):
                for i in range(rows - kernel_row + 1):
                    for j in range(cols - kernel_col + 1):
                        output[batch, :, i, j] = torch.sum(
                            # Kernel (num,channels,row,cols)
                            padding_mat[batch, :, i:i + kernel_row, j:j + kernel_col] * self.kernel[num, :, :, :]
                            + self.bias[num]
                        )
        return output

    def backward(self, delta):

        # 初始化
        self.gradient_w = torch.zeros(self.gradient_w.size())
        self.gradient_bias = torch.zeros(self.gradient_bias.size())

        # 求解参数梯度



        # 求解偏置梯度

        # 翻转卷积核
        rot_w = torch.rot90(self.kernel, k=2, dims=[2, 3])  # 旋转180

        kernel_row = self.kernel.size()[2]
        kernel_col = self.kernel.size()[3]
        padding_mat = F.pad(delta, [kernel_row - 1, kernel_row - 1,
                                    kernel_col - 1, kernel_col - 1], mode='constant', value=0)
        rows = 2 * kernel_row - 1 + delta.size[2]
        cols = 2 * kernel_col - 1 + delta.size[3]

        kernel = torch.swapaxes(rot_w, 0, 1)
        # 求解l-1层梯度
        delta_last = torch.zeros((delta.size[0], kernel.size[1], rows, cols))
        for batch in range(delta_last.size()[0]):
            for num in range(self.kernel.size()[0]):
                for i in range(rows - kernel_row + 1):
                    for j in range(cols - kernel_col + 1):
                        delta_last[batch, :, i, j] = torch.sum(
                            # Kernel (num,channels,row,cols)
                            padding_mat[batch, :, i:i + kernel_row, j:j + kernel_col] * kernel[num, :, :, :])


        self.gradient_bias += 1e-9
        self.gradient_w += 1e-9
        return delta_last


class Relu():
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return torch.maximum(x, 0)

    def backward(self, delta):
        ret = torch.clamp(delta, min=0)  # 截断小于0的值
        return ret

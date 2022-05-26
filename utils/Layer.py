import torch
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -%(levelname)s - %(message)s')
class Conv():
    def __init__(self, kernel=(1,1,3,3), padding=(0, 0), bias=0):
        '''
        初始化卷积层
        :param kernel: (num,channels,row,cols) channels 对应上一层通道数 num对应下一层通道数
        :param padding:
        :param stride:
        '''
        self.padding = padding
        self.kernel = torch.rand(*kernel,dtype=torch.float32,device=torch.device('cuda'))
        self.bias = torch.ones(kernel[0],device=torch.device('cuda')) * bias
        self.gradient_w = torch.zeros(*kernel,dtype=torch.float32,device=torch.device('cuda'))  # 指定参数尺寸
        self.gradient_bias = torch.zeros(self.bias.size(),dtype=torch.float32,device=torch.device('cuda'))
        self.batch_size = None
        self.input_mat = None

    def forward(self, x: torch.tensor):
        '''
        多通道前向传播
        :param x:  （batch,channels,rows,cols)
        :return:
        '''
        # logging.debug("开始单次正向传播")
        self.batch_size = x.size()[0]
        self.input_mat = x
        output = torch.zeros(*x.size(),dtype=torch.float32,device=torch.device('cuda'))
        # logging.debug("开始padding填充")
        padding_mat = F.pad(x, [self.padding[0], self.padding[0],
                                self.padding[1], self.padding[1]],
                            mode='constant', value=0)  # padding 先后顺序不影响

        rows = padding_mat.size()[2]
        cols = padding_mat.size()[3]
        kernel_row = self.kernel.size()[2]
        kernel_col = self.kernel.size()[3]

        # logging.debug("正向传播卷积计算")
        for batch in range(x.size()[0]):
            # logging.debug("开始循环batch")
            for num in range(self.kernel.size()[0]):
                for i in range(rows - kernel_row + 1):
                    for j in range(cols - kernel_col + 1):
                        output[batch, :, i, j] = torch.sum(
                            # Kernel (num,channels,row,cols)
                            padding_mat[batch, :, i:i + kernel_row, j:j + kernel_col] * self.kernel[num, :, :, :])\
                                                 + self.bias[num]

        return output

    def backward(self, delta):

        # 初始化

        self.gradient_w = torch.full(self.gradient_w.size(),0.0,device=torch.device('cuda'))
        self.gradient_bias = torch.full(self.gradient_bias.size(),0.0,device=torch.device('cuda'))



        # 翻转卷积核
        rot_w = torch.rot90(self.kernel, k=2, dims=[2, 3])  # 旋转180

        kernel_row = self.kernel.size()[2]
        kernel_col = self.kernel.size()[3]
        kernel = torch.swapaxes(rot_w,0,1)
        padding_mat = F.pad(delta, [kernel_row - 1, kernel_row - 1,
                                    kernel_col - 1, kernel_col - 1], mode='constant', value=0)
        rows = 2 * kernel_row - 2 + delta.size()[2] # 扩充完毕的行
        cols = 2 * kernel_col - 2 + delta.size()[3] # 扩充完毕的列

        # 求解l-1层梯度
        delta_last = torch.zeros((delta.size()[0], kernel.size()[1],
                                  rows - kernel_row + 1 , cols - kernel_col + 1),
                                 device=torch.device('cuda'))

        for batch in range(self.batch_size):  # batch
            for i in range(rows - kernel_row + 1):
                for j in range(cols - kernel_col + 1):
                    delta_last[batch, :, i, j] = torch.sum(
                        # Kernel (num,channels,row,cols)
                        padding_mat[batch, :, i:i + kernel_row, j:j + kernel_col] * kernel)

        delta_last = delta_last[:,:,
                     self.padding[0]:-self.padding[0],
                     self.padding[1]: -self.padding[1]]

        # 求解参数梯度
        # 交换input和delta 通道数目
        swap_input = torch.swapaxes(self.input_mat, 0, 1)
        row_padding = self.kernel.size()[2]//2
        col_padding = self.kernel.size()[3]//2
        padding_swap_input = F.pad(swap_input,
                                   [row_padding,row_padding,col_padding,col_padding],
                                   mode='constant',value=0) # input with padding
        rows_2 = padding_swap_input.size()[2]
        cols_2 = padding_swap_input.size()[3]
        kernel_row_2 = delta.size()[2]
        kernel_col_2 = delta.size()[3]


        for u in range(padding_swap_input.size()[0]):
            for v in range(delta.size()[1]):
                for i in range(rows_2-kernel_row_2+1):
                    for j in range(cols_2-kernel_col_2+1):
                        self.gradient_w[u,v,i,j] = torch.sum(
                            padding_swap_input[u,:,i:i+kernel_row_2,j:j+kernel_col_2]*delta[:,v,:,:]
                        )

        self.gradient_w /= self.batch_size

        # 求解偏置梯度
        self.gradient_bias = torch.sum(torch.sum(torch.sum(delta, dim=-1), dim=-1), dim=0)
        self.gradient_bias /= self.batch_size

        # 防止梯度消失
        self.gradient_bias += 1e-9
        self.gradient_w += 1e-9
        return delta_last


class Relu():
    def __init__(self):
        self.x = None

    def forward(self, x):
        zero = torch.zeros_like(x,device=torch.device('cuda'))
        return torch.maximum(x, zero)

    def backward(self, delta):
        ret = torch.clamp(delta, min=0)  # 截断小于0的值
        return ret

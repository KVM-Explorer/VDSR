# Todo 等待全部反向传播完成后，优化器更新参数
import torch
import utils.Net as Net


def MSE(x, y):
    loss = torch.sum(torch.square(x - y)) / (x.size()[0] * x.size()[2] * x.size()[3])
    delta = (x - y) / (x.size()[2] * x.size()[3])
    return delta, loss


class SGD():
    def __init__(self, loss_function, learning_rate=0.1):
        self.learning_rate = learning_rate
        # self.loss_function = loss_function

    def solve(self, x: torch.tensor, y: torch.tensor):
        # Todo 实现动量SGD
        delta,loss = MSE(x,y)
        return delta,loss

    def update(self, model: Net.VDSR):

        for i in range(len(model.layers)):
            layer = model.layers[i]
            if type(layer) == type(Net.Layer.Conv()):
                # Todo 针对层设置学习率
                model.layers[i].kernel -= model.layers[i].gradient_w * self.learning_rate
                model.layers[i].bias -= model.layers[i].gradient_bias * self.learning_rate

        return model


# Todo complete future
class Adam():
    def __init__(self, loss_function, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def solve(self, x: torch.tensor, y: torch.tensor):
        # Todo delta
        # Todo loss
        delta = None
        loss = None
        return delta, loss

    def update(self, model: Net.VDSR):
        # Todo 更新模型参数

        return model

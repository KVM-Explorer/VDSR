
# Todo update initial learning rate
# Todo 等待全部反向传播完成后，优化器更新参数
class SGD():
    def __init__(self,learning_rate = 0.1):
        self.learning_rate = learning_rate

class Adam():
    def __init__(self,learning_rate = 0.1):
        self.learning_rate = learning_rate
import utils.Layer as Layer
import logging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -%(levelname)s - %(message)s')

class VDSR():
    def __init__(self):
        self.x = input
        self.layers = []
        for i in range(1):
            self.layers.append(Layer.Conv(kernel=(16, 16, 3, 3),padding=(1,1)))
            self.layers.append(Layer.Relu())
        self.layers.append(Layer.Conv(kernel = (16, 1, 3, 3),padding=(1,1)))
        self.layers.append(Layer.Relu())

    def forward(self, initial_data):
        print("Forward")
        x = initial_data
        level = 0
        for t in self.layers:
            x = t.forward(x)
            level += 1
            logging.debug(f"Layer:{level} has been finished")
        output = x + initial_data
        return output

    def backward(self, delta):
        print("Backward")
        x = delta
        level = 0
        for t in self.layers:
            x = t.backward(x)
            level += 1
            logging.debug(f"Layer:{level} has been finished")
        output = x
        return output



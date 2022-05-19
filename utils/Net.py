import utils.Layer as Layer
class VDSR():
    def __int__(self):
        self.x = input
        self.layers = []
        for i in range(20):
            self.layers.append(Layer.Conv())
            self.layers.append(Layer.Relu())

    def forward(self,input):
        x = input
        for t in self.layers:
            x = t.forward(x)
        output = x + input
        return output



    def backward(self,delta):
        x = delta
        for t in self.layers:
            x = t.backward(x)
        output = x
        return output

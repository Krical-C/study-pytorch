import numpy as np
import myNeuron
class NeuralNetwork:
    def __init__(self):
        weights=np.array([0,1])
        bias=0;
        self.h1= myNeuron.Neuron(weights, bias)
        self.h2= myNeuron.Neuron(weights, bias)
        self.o1= myNeuron.Neuron(weights, bias)
    def feedforward(self,x):
        out_h1=self.h1.feedforward(x)
        out_h2=self.h2.feedforward(x)
        out_o1=self.o1.feedforward(np.array([out_h1,out_h2]))

        return out_o1
network=NeuralNetwork()
x=np.array([2,3])
print(network.feedforward(x))
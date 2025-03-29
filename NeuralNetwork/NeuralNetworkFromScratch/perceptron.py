import numpy as np
from fontTools.merge.util import avg_int
from mpmath.functions.signals import sigmoid

#activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#derivation activation function
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


class Perceptron:
    bias: float
    weight: float
    inputs: list[float]
    current_output: float
    current_z: float



    def __init__(self, bias, weight):
        self.bias = bias
        self.weight = weight
        self.inputs = []
        self.current_output = 0
        self.current_z = 0
        self.learning_rate = 0.05

    #def forward function to calculate and adjust weights, values.
    #Linear_activation = True in the output layer to use softmax
    def forward(self, inputs, linear_activation = False):
        #propagation function (Sum function)
        self.inputs = inputs
        sum = 0
        for i in inputs:
            sum += i*self.weight + self.bias
        #create current z value by useing mean of inputs to estimate
        self.current_z = np.mean(inputs) * self.weight + self.bias
        if linear_activation ==True:
            self.current_output = sum
        else:
            # activation function (sigmoid function)
            self.current_output = sigmoid(sum)

        return self.current_output


#adjust bias and weights based on given Target
    def backward(self, target, learning_rate):
        delta = (self.current_z - target) * sigmoid_deriv(self.current_z)
        #use avg from input toadjust weights and bias
        avg_input = np.mean(self.inputs)
        self.weight -= avg_input * learning_rate * delta
        self.bias -= learning_rate * delta
        return self.current_output



#testing
if __name__ == "__main__":
    p = Perceptron(bias=0.5, weight = 0.8)

    # example input
    inputs = [0.2, 0.4, 0.6]
    target = 1.0

    # forward pass
    output = p.forward(inputs)
    print("Vorwärts Output:", output)

    # Backward-Pass
    learning_rate = 0.1
    delta = p.backward(target, learning_rate)
    print("Delta (Fehler):", delta)
    print("Aktualisiertes Gewicht:", p.weight)
    print("Aktualisierter Bias:", p.bias)










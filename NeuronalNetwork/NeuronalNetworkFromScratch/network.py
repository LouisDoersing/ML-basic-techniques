import numpy as np
from typing import List
from transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet import softmax
import torch
from layer import Layer
from perceptron import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class Network:
    input_vector: np.ndarray
    layers: List[Layer]
    output_vector: np.ndarray
    amount_of_mid_layers:int
    output_layer: Layer



    def __init__(self, input_vector: np.ndarray, amount_of_mid_layers, perceptrons_hidden_layer: List[Perceptron]
                 , perceptrons_output_layer: List[Perceptron]):
        self.input_vector = input_vector
        self.amount_of_mid_layers = amount_of_mid_layers
        self.layers = []

        for i in range(amount_of_mid_layers):
            self.layers.append(Layer(perceptrons_hidden_layer))

        self.output_layer = Layer(perceptrons_output_layer)
        self.layers.append(self.output_layer)



    def neuronal_network_forward_learning(self, input_vector):
        current_output = input_vector
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def neuronal_network_backward_run(self, target):
        output_vector = []
        for layer in self.layers:
            output_vector = layer.backward(target)
        return output_vector

    #estimate final output based on softmax
    def last_layer_output(self, layer: Layer):
        z_values = []
        for p in layer.perceptrons:
            z_values.append(p.forward(self.input_vector, linear_activation = True))
        #use softmax on all z values
        z_tensor = torch.tensor(z_values)
        softmax_outputs = torch.softmax(z_tensor, dim=0)
        for i,p in enumerate(layer.perceptrons):
            p.current_output = softmax_outputs[i]
        return softmax_outputs

    # Classification with argmax
    def classify(self):
        softmax_outputs = self.last_layer_output(self.output_layer)
        predicted_index = torch.argmax(softmax_outputs).item()
        #1 == True, 0 == False
        if predicted_index == 1:
            return True, softmax_outputs
        else:
            return False, softmax_outputs








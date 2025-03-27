from typing import List
from perceptron import Perceptron
import numpy as np



class Layer:
    perceptrons:List[Perceptron]

    def __init__(self, perceptrons:list[Perceptron]):
        self.perceptrons = perceptrons



    def forward(self, x):
        outputs = []
        for perceptron in self.perceptrons:
            outputs.append(perceptron.forward(x))
        return outputs

    def backward(self, targets):
        outputs = []
        for i, perceptron in enumerate(self.perceptrons):
            if i < len(targets):
                target_value = targets[i]
            else:
                # Hier könntest du einen Standardwert setzen oder den Fehler anders behandeln
                target_value = 0  # Beispiel: setze 0 als Standardwert
            outputs.append(perceptron.backward(target_value, 0.05))
        return outputs


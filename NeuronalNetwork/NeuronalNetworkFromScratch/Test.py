import copy

import numpy as np
from typing import List
from transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet import softmax
import torch
from layer import Layer
from perceptron import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from network import Network


# Implement cross entropy
def cross_entropy_loss_torch(softmax_outputs, target):
    target_tensor = torch.tensor(target, dtype=torch.float64)
    loss = - torch.sum(target_tensor * torch.log(softmax_outputs + 1e-8))
    return loss


def predict(network, vektor):
    network.input_vector = vektor
    prediction, softmax_outputs = network.classify()
    print(f"Prediction for input Vektor: {prediction}")
    print(f"Softmax: {softmax_outputs}")




# Testing
if __name__ == "__main__":
    #create train and test sets with 3 dimension inputs and 2 Dimension (True/False) output
    X, y = make_classification(n_samples=200, n_features=3, n_redundant=0, n_classes=2, random_state=42,
                               n_informative=3)
    #normalize data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Perceptrons for hidden layers
    perceptron1 = Perceptron(bias=0.5, weight=0.8)
    perceptron2 = Perceptron(bias=0.2, weight=0.7)
    perceptron3 = Perceptron(bias=0.6, weight=1.0)
    perceptrons_mid_layer = [perceptron1, perceptron2, perceptron3]

    # pereptrons for output layer
    perceptron_1_output = Perceptron(bias=0.2, weight=0.7)
    perceptron_2_output = Perceptron(bias=0.2, weight=0.7)
    perceptrons_out_layer = [perceptron_1_output, perceptron_2_output]





    # create Network with 2 Hidden layers
    network = Network(X_train[0], 2, perceptrons_mid_layer, perceptrons_out_layer)

    # Early Stopping Parameter (saves model with lowest loss after n epoch steps)
    patience = 40
    best_loss = 100
    no_improvement = 0
    best_model = None

    epochs = 500
    for epoch in range(epochs):
        # Shuffling data before training
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0.0
        for j in range(len(X_train_shuffled)):
            input_vector = X_train_shuffled[j]
            # One-Hot-Kodierung: [1, 0] for False , [0, 1] für True
            if y_train_shuffled[j] == 0:
                target = np.array([1, 0])
            else:
                target = np.array([0, 1])

            # set input vektor in network
            network.input_vector = input_vector
            # Forward-Pass
            network.neuronal_network_forward_learning(input_vector)
            # Backward-Pass
            network.neuronal_network_backward_run(target)

            # calculate loss from current model
            prediction, softmax_outputs = network.classify()
            loss = cross_entropy_loss_torch(softmax_outputs, target)
            epoch_loss += loss.item()
        #calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(X_train_shuffled)


        # check if curren loss is lower than best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model = copy.deepcopy(network)
            no_improvement = 0
            print("found better model!")
            print(f"Epoch {epoch}: Avg. loss = {avg_epoch_loss}")
        else:
            no_improvement += 1

        # Early Stopping if next n models are not better than current
        if no_improvement >= patience:
            print(f"Early Stopping in epoch: {epoch}")
            break


    #test wit x_test, y_test
    correct = 0
    for i in range(len(X_test)):
        input_vector = X_test[i]
        # set input in network
        network.input_vector = input_vector
        # classify input
        prediction, softmax_outputs = network.classify()
        loss = cross_entropy_loss_torch(softmax_outputs, target)
        print(f"Loss: {loss.item()}")

        # check how ofthen the model is classifying correct
        predicted_label = 1 if prediction else 0
        if predicted_label == y_test[i]:
            correct += 1
    #calculate accuracy
    accuracy = correct / len(X_test)
    print("Accuracy:", accuracy)


    #test cases
    test_one = np.array([1, 2, -3])
    test_two = np.array([1, 1, 1])
    test_three = np.array([-1, -5, 1])
    predict(best_model, test_one)
    predict(best_model, test_two)
    predict(best_model, test_three)
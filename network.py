import math
import random

import numpy as np

class network:
    def __init__(self, input_nodes : int, hidden_nodes : int, hidden_layers : int, output_nodes : int, learning_rate : float):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.hidden_layers = hidden_layers
        self.output_nodes = output_nodes
        self.total_nodes = input_nodes + (hidden_nodes * hidden_layers) + output_nodes
        self.learning_rate = learning_rate

        # set up the arrays
        self.values = np.zeros(self.total_nodes)
        self.expectedValues = np.zeros(self.total_nodes)
        self.thresholds = np.zeros(self.total_nodes)

        # the weight matrix is always square
        self.weights = np.zeros((self.total_nodes, self.total_nodes))

        # set random seed! this is so we can experiment consistently
        random.seed('this_is-going.to be`a|long>one')

        # set initial random values for weights and thresholds
        # this is a strictly upper triangular matrix as there is no feedback
        # loop and there inputs do not affect other inputs
        for i in range(self.input_nodes, self.total_nodes):
            self.thresholds[i] = random.random() / random.random()
            for j in range(i + 1, self.total_nodes):
                self.weights[i][j] = random.random() * 2
    
    def get_hidden_offset(self, layer : int) -> int:
        # calculate the offset of a hidden layer
        return self.input_nodes + (self.hidden_nodes * layer)

    def set_values(self, inputs : np.ndarray, expected):
        # set input values
        assert(len(inputs) == self.input_nodes, 'length of input should match the input nodes')
        for i, val in enumerate(inputs):
            self.values[i] = val

        # set output values
        if hasattr(expected, "__len__"):
            assert(len(expected) == self.output_nodes, 'length of expected should match the output nodes')
            offset = self.get_hidden_offset(self.hidden_layers)
            for i, val in enumerate(expected):
                self.expectedValues[offset + i] = val
        else:
            self.expectedValues[self.total_nodes - 1] = expected

    def process(self):
        # update the hidden layers
        # update the first layer with the inputs
        for i in range(self.input_nodes, self.get_hidden_offset(1)):
            # sum weighted input nodes for each hidden node, compare threshold, apply sigmoid
            weight = 0.0
            for j in range(self.input_nodes):
                weight += self.weights[j][i] * self.values[j]
            weight -= self.thresholds[i]
            self.values[i] = 1 / (1 + math.exp(-weight))

        # update the remaining hidden layers
        if self.hidden_layers > 1:
            for i in range(1, self.hidden_layers):
                offset = self.get_hidden_offset(i)
                for j in range(offset, offset + self.hidden_nodes):
                    # sum weighted input nodes for each hidden node, compare threshold, apply sigmoid
                    weight = 0.0
                    for k in range(offset):
                        weight += self.weights[k][j] * self.values[j]
                    weight -= self.thresholds[j]
                    self.values[j] = 1 / (1 + math.exp(-weight))

        # update the output nodes
        for i in range(self.get_hidden_offset(self.hidden_layers), self.total_nodes):
            # sum weighted hidden nodes for each output node, compare threshold, apply sigmoid
            weight = 0.0
            for j in range(self.input_nodes, self.get_hidden_offset(self.hidden_layers - 1)):
                weight += self.weights[j][i] * self.values[j]
            weight -= self.thresholds[i]
            self.values[i] = 1 / (1 + math.exp(-weight))

    def processErrors(self):
        sumOfSquaredErrors = 0.0

        # we only look at the output nodes for error calculation
        for i in range(self.input_nodes + self.hidden_nodes, self.total_nodes):
            error = self.expectedValues[i] - self.values[i]
            #print error
            sumOfSquaredErrors += math.pow(error, 2)
            outputErrorGradient = self.values[i] * (1 - self.values[i]) * error
            #print outputErrorGradient

            # now update the weights and thresholds
            for j in range(self.input_nodes, self.input_nodes + self.hidden_nodes):
                # first update for the hidden nodes to output nodes (1 layer)
                delta = self.learning_rate * self.values[j] * outputErrorGradient
                # print delta
                self.weights[j][i] += delta
                hiddenErrorGradient = self.values[j] * (1 - self.values[j]) * outputErrorGradient * self.weights[j][i]

                # and then update for the input nodes to hidden nodes
                for k in range(self.input_nodes):
                    delta = self.learning_rate * self.values[k] * hiddenErrorGradient
                    self.weights[k][j] += delta

                # update the thresholds for the hidden nodes
                delta = self.learning_rate * -1 * hiddenErrorGradient
                #print delta
                self.thresholds[j] += delta

            # update the thresholds for the output node(s)
            delta = self.learning_rate * -1 * outputErrorGradient
            self.thresholds[i] += delta
        return sumOfSquaredErrors

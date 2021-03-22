import random
import numpy as np

class Neuron(object):
    def __init__(self, inputs_amount, neuron_number_in_layer, momentum=0.2, learning_rate=0.1, bias=1):
        self.bias = bias
        self.weights = np.zeros(inputs_amount + 1)
        for i in range(inputs_amount + 1):
            self.weights[i] = random.uniform(-0.5,0.5)
        if self.bias:
            self.weights[0] = 1
        self.weight_last_change = np.zeros(inputs_amount + 1)
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.output_value = 0.0
        self.weight_change_factor = 0.0
        self.number = neuron_number_in_layer

    def activation_function(self, stimulus):
        return stimulus

    def derivative(self, x):
        return 1

    def set_output_value(self, value):
        self.output_value = value

    def predict(self, previous_layer):
        summation = 0
        for weight, neuron in zip(self.weights[1:], previous_layer):
            summation += weight * neuron.output_value
        if self.bias:
            summation += self.weights[0]
        self.output_value = self.activation_function(summation)

    def output_layer_factor(self, expected_value):
        self.weight_change_factor = (self.output_value - expected_value) * self.derivative(self.output_value)

    # def hidden_layer_factor(self, next_layer):
    #     summation = 0
    #     for neuron in next_layer:
    #         summation += neuron.weight_change_factor * neuron.weights[self.number]
    #     self.weight_change_factor = summation * self.derivative(self.output_value)

    def update_weights(self, previous_layes):
        for previous_neuron, i in zip(previous_layes, range(1, len(self.weights))):
            weight_change = self.weight_change_factor * self.learning_rate * previous_neuron.output_value + self.momentum * self.weight_last_change[i]
            self.weight_last_change[i] = weight_change
            self.weights[i] -= weight_change
        if self.bias:
            weight_change = self.weight_change_factor * self.learning_rate * 1 + self.momentum * self.weight_last_change[0]
            self.weight_last_change[0] = weight_change
            self.weights[0] -= weight_change
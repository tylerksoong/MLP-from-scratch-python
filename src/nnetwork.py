import numpy as np
from layer import Layer

class Network:
    def __init__(self, layer_sizes):
        self.layers = []

        for i in range(len(layer_sizes)-1):
            if i == layer_sizes -1
                self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

    def calculate_output(self, datapoint):
        output = datapoint.input
        for layer in self.layers:
            output = layer.calculate_output(output)
            layer.output = output

        return output

    
    def loss(self, datapoints, outputs):
        
        # Prevent log(0) by adding a small value (epsilon)
        epsilon = 1e-15
        outputs = np.clip(outputs, epsilon, 1 - epsilon)  # Ensure no 0s or 1s in output
        
        # Calculate cross-entropy loss
        # Assuming data.expected_output is one-hot encoded
        targets = np.array([data.expected_output for data in datapoints])  # Shape: (batch_size, num_classes)
        loss = -np.mean(targets * np.log(outputs))  # Cross-entropy loss formula
        return loss
    
    def backward(self, y_hat, target, learning_rate):
        # Compute gradients for the output layer
        self.layers[-1].calculate_gradients(y_hat, self.layers[-2].preactivation_values, target)
        
        # Propagate gradients backward through hidden layers
        for i in range(len(self.layers) - 2, -1, -1):  # Iterate backward
            dL_dX = self.layers[i+1].input_weight_gradients  # Get the propagated gradient from the next layer
            self.layers[i].calculate_gradients(self.layers[i].output, self.layers[i-1].preactivation_values, dL_dX)

        for layer in self.layers:
            layer.apply_gradients(learning_rate)

    def train(self, datapoints, epochs, learning_rate):
        for epoch in range(epochs):
            y_hat = np.array([self.calculate_output(x) for x in datapoints])
            targets = np.array([x.expected_output for x in datapoints])
            self.backward(y_hat, targets, learning_rate)
            if epoch % 100 == 0:
                print(self.loss(datapoints))
import numpy as np
from layer import Layer

class Network:
    def __init__(self, layer_sizes):
        self.layers = []

        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

    def calculate_output(self, datapoint):
        output = datapoint.input
        for layer in self.layers:
            output = layer.calculate_output(output)

        return self.softmax(output)

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def cost(self, datapoints):
        # Calculate model output
        outputs = np.array([self.calculate_output(data.input) for data in datapoints])  # Shape: (batch_size, num_classes)
        
        # Prevent log(0) by adding a small value (epsilon)
        epsilon = 1e-15
        outputs = np.clip(outputs, epsilon, 1 - epsilon)  # Ensure no 0s or 1s in output
        
        # Calculate cross-entropy loss
        # Assuming data.expected_output is one-hot encoded
        targets = np.array([data.expected_output for data in datapoints])  # Shape: (batch_size, num_classes)
        loss = -np.sum(targets * np.log(outputs)) / len(datapoints)  # Cross-entropy loss formula
        return loss
    
  
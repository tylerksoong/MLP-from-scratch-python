import numpy as np
from layer import Layer

class Network:
    def __init__(self, layer_sizes):
        self.layers = []

        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))

        self.layers[-1].is_output_layer = True

    def softmax(self, logits):
        """
        :param logits: A NumPy array of raw output scores
        :return: A NumPy array displaying probability distribution
        """
        exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def calculate_output(self, datapoints):
        output = np.array([x.input for x in datapoints])
        for layer in self.layers:
            output = layer.calculate_outputs(output)
            layer.output = output

        return np.array([self.softmax(x) for x in output])

    def loss(self, datapoints):

        # Prevent log(0) by adding a small value (epsilon)
        epsilon = 1e-15
        outputs = np.clip(self.calculate_output(datapoints), epsilon, 1 - epsilon)  # Ensure no 0s or 1s in output

        # Calculate cross-entropy loss
        # Assuming data.expected_output is one-hot encoded
        targets = np.array([data.expected_output for data in datapoints])  # Shape: (batch_size, num_classes)
        loss = -np.mean(np.sum(targets * np.log(outputs), axis = 1))  # Cross-entropy loss formula
        return loss


    def backward(self, datapoints, learning_rate):
        # Initialize output and preactivation layers
        y_hat = self.calculate_output(datapoints)
        targets = np.array([x.expected_output for x in datapoints])

        # Compute gradients for the output layer
        self.layers[-1].calculate_gradients(y_hat, self.layers[-2].output, target = targets)
        
        # Propagate gradients backward through hidden layers
        for i in range(len(self.layers) - 2, 0, -1):  # Iterate backward
            dL_dX = self.layers[i+1].input_weight_gradients  # Get the propagated gradient from the next layer
            self.layers[i].calculate_gradients(self.layers[i].output, self.layers[i-1].preactivation_values, self.layers[i+1].weights, dL_dZ_next= dL_dX)

            # Handle the first layer separately
        if len(self.layers) > 1:  # Make sure there's more than one layer
            dL_dX = self.layers[1].input_weight_gradients
            # For the first layer, use the raw input data
            input_data = np.array([x.input for x in datapoints])
            self.layers[0].calculate_gradients(self.layers[0].preactivation_values, input_data, self.layers[1].weights,
                                               dL_dZ_next=dL_dX)

        for layer in self.layers:
            layer.apply_gradients(learning_rate)


    def train(self, datapoints, epochs, learning_rate):
        for epoch in range(epochs):
            y_hat = self.calculate_output(datapoints)
            targets = np.array([x.expected_output for x in datapoints])
            self.backward(datapoints, learning_rate)
            if epoch % 100 == 0:
                print(self.loss(datapoints))
import numpy as np

class Layer:
    def __init__(self, input_length, output_length):
        #xavier initialization
        limit = np.sqrt(6 / (input_length + output_length))
        self.weights = np.random.uniform(-limit, limit, (input_length, output_length))
        self.biases = np.zeros(output_length)
        self.raw_input = None

        self.input_weight_gradients = None
        self.output_weight_gradients = None
        self.bias_gradients = None

    def calculate_output(self, input):
        output = np.dot(input, self.weights) + self.biases
        self.raw_input = output
        return np.array([self.activation_function(x) for x in output])
    
    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        sig = self.activation_function(self, x)
        return sig * (1 - sig)
    
    #dL is derivative of loss function calcuated in network class
    def calculate_gradients(self, previous_raw_input):
        dLdA = self.sigmoid_derivative(self.raw_input)
        dAdZ = np.dot(self.weights, previous_raw_input) #A is activation value, Z = wA + B of previous later
        dLdW 
import numpy as np

class Layer:
    def __init__(self, input_length, output_length, is_output_layer):
        self.is_output_layer = is_output_layer

        #xavier initialization
        limit = np.sqrt(6 / (input_length + output_length))
        self.weights = np.random.uniform(-limit, limit, (input_length, output_length))
        self.biases = np.zeros(output_length)
        self.preactivation_values = None #activation_function(Z = wA + b), A is activation value of previous layer
        self.output = None

        self.input_weight_gradients = None 
        self.output_weight_gradients = None
        self.bias_gradients = None

    def calculate_output(self, input):
        output = np.dot(input, self.weights) + self.biases
        self.preactivation_values = np.array(output)
        return np.array([self.softmax(x) for x in output])
    
    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
    
    #dL is derivative of loss function calcuated in network class
    def calculate_gradients(self, output, previous_preactivation_values, dL_dZ=None, target=None):
        batch_size = output.shape[0]

        if previous_preactivation_values.ndim == 1:
            previous_preactivation_values = previous_preactivation_values.reshape(1, -1)

        dL_dZ = (output - target)

        self.output_weight_gradients = np.dot(previous_preactivation_values.T, dldz) / batch_size
        self.input_weight_gradients = np.dot(dldz, self.weights.T)

        self.bias_gradients = np.sum(dldz, axis=0) / batch_size

    def apply_gradients(self, learning_rate):
        self.weights -= learning_rate * self.output_weight_gradients
        self.biases -= learning_rate * self.bias_gradients
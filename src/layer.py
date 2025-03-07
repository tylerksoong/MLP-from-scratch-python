import numpy as np

class Layer:
    def __init__(self, input_length, output_length):
        self.is_output_layer = False

        #xavier initialization
        limit = np.sqrt(6 / (input_length + output_length))
        self.weights = np.random.uniform(-limit, limit, (input_length, output_length)) #input_length x output_length matrix
        self.biases = np.zeros(output_length) #output_length x 1 matrix
        self.preactivation_values = None #activation_function(Z = wA + b), A is activation value of previous layer
        self.output = None

        self.input_weight_gradients = None 
        self.output_weight_gradients = None
        self.bias_gradients = None

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
            x (numpy.ndarray): Input values

        Returns:
            numpy.ndarray: Sigmoid of input values, bounded between 0 and 1
        """
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.

        Parameters:
            x (numpy.ndarray): Input values (pre-activation values)

        Returns:
            numpy.ndarray: Derivative of sigmoid for the input values
        """
        # Get sigmoid of x
        sig_x = self.sigmoid(x)

        # Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
        return sig_x * (1 - sig_x)

    def calculate_outputs(self, input):
        """

        :param input: A batch_size x 784 matrix
        :return:
        """
        outputs = np.matmul(input, self.weights) + self.biases
        self.preactivation_values = np.array(outputs)
        return np.array([self.sigmoid(x) for x in outputs])

    def calculate_gradients(self, output, previous_activation_values, weights_next = None, dL_dZ_next=None, target=None):
        """
        d
        """
        batch_size = output.shape[0]

        if self.is_output_layer:
            # Compute loss gradient w.r.t. preactivation values (Z) for output layer
            dL_dZ = output - target  # Softmax + Cross-Entropy simplifies to this

        else:
            # Compute loss gradient w.r.t. preactivation values (Z) for hidden layer
            dL_dZ = np.matmul(dL_dZ_next, weights_next.T) * self.sigmoid_derivative(self.preactivation_values) # Backpropagate error

        # Compute gradients for weights, biases, and input
        self.output_weight_gradients = np.matmul(previous_activation_values.T, dL_dZ) / batch_size
        self.bias_gradients = np.mean(dL_dZ, axis=0)

        # Store for backpropagation to previous layer
        self.input_weight_gradients = dL_dZ

    def apply_gradients(self, learning_rate):
        self.weights -= learning_rate * self.output_weight_gradients
        self.biases -= learning_rate * self.bias_gradients



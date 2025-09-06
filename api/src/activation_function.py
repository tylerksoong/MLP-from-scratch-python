from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def function(self, x):
        pass

    def derivative(self, x):
        pass

class Sigmoid(ActivationFunction):
    def function(self, x):
        """
                Sigmoid activation function.

                Parameters:
                    x (numpy.ndarray): Input values

                Returns:
                    numpy.ndarray: Sigmoid of input values, bounded between 0 and 1
                """
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def derivative(self, x):

        """
        Derivative of the sigmoid function.

        Parameters:
            x (numpy.ndarray): Input values (pre-activation values)

        Returns:
            numpy.ndarray: Derivative of sigmoid for the input values
        """
        # Get sigmoid of x
        sig_x = self.function(x)

        # Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
        return sig_x * (1 - sig_x)

class LeakyReLU(ActivationFunction):
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
    def function(self, x, alpha = 0.01):
        """ReLU activation function: f(x) = max(0, x)"""
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x, alpha = 0.01):
        """Derivative of ReLU: f'(x) = 1 for x > 0, else 0"""
        return np.where(x > 0, 1, self.alpha)
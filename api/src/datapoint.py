import numpy as np

class DataPoint:
    def __init__(self, input, expected_output):
        self.input = input.flatten()
        self.expected_output = np.zeros(10)
        self.expected_output[expected_output] = 1

    def get_input(self):
        return self.input

    def get_expected_output(self):
        return self.expected_output
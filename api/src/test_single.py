from nnetwork import Network
from mnistreader import MnistDataloader
from datapoint import DataPoint

import numpy as np
import matplotlib.pyplot as plt
import random

from activation_function import LeakyReLU

nn = Network( layer_sizes=None, activation_func=LeakyReLU(), weight_file='../models/current_model/weights.npz', bias_file='../models/current_model/biases.npz')

probs = nn.calculate_single_output(np.random.random(size=(28,28)).flatten())

for i, prob in enumerate(probs):
    print(f"Class {i}: {prob:.3f}")
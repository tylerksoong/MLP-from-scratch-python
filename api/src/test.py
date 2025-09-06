from nnetwork import Network
from mnistreader import MnistDataloader
from datapoint import DataPoint

import numpy as np
import matplotlib.pyplot as plt
import random

from activation_function import LeakyReLU

input_path = '../data'
training_images_filepath = input_path + '/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = input_path + '/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath = input_path + '/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = input_path + '/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

long_xtest = [np.asarray(x).flatten() for x in x_test]
datapoints = np.array([DataPoint(x,y) for x,y in zip(long_xtest,y_test)])

nn = Network( layer_sizes=None, activation_func=LeakyReLU(), weight_file='../models/current_model/weights.npz', bias_file='../models/current_model/biases.npz')

y_hat = nn.calculate_output(datapoints)
expected_values = [x.expected_output for x in datapoints]

guesses = np.argmax(y_hat, axis= 1)
true = np.argmax(expected_values, axis = 1)

correct = guesses - true
correct = np.array([1 if x == 0 else 0 for x in correct])
print(f'Accuracy: {np.sum(correct) / len(datapoints)}')

wrong_guess_indices = np.where(correct == 0)
right_guess_indices = np.where(correct == 1)

r_index = random.randint(0,10000)
print(f'probabilities of sample {r_index}')
for i, prob in enumerate(y_hat[r_index], 0):
    print(f"Class {i}: {prob:.3f}")

print(f'Label: {true[r_index]}')


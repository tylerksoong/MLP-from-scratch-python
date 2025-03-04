from layer import Layer
from nnetwork import Network
from mnistreader import MnistDataloader
from datapoint import DataPoint

import numpy as np
from array import array
import matplotlib.pyplot as plt
import random

input_path = '../data'
training_images_filepath =  'data/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath =  'data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath =  'data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath =  'data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

long_xtrain = [np.asarray(x).flatten() for x in x_train]
datapoints = np.array([DataPoint(x,y) for x,y in zip(long_xtrain,y_train)])

main_network = Network([784, 100, 100, 10])
print(main_network.calculate_output(datapoints[0]))


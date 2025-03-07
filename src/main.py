
from nnetwork import Network
from mnistreader import MnistDataloader
from datapoint import DataPoint

import numpy as np
import matplotlib.pyplot as plt

input_path = '../MLP-from-scratch-python/data'
training_images_filepath = input_path + '/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = input_path + '/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath = input_path + '/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = input_path + '/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

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


long_xtrain = [np.asarray(x).flatten() for x in x_train]
datapoints = np.array([DataPoint(x,y) for x,y in zip(long_xtrain,y_train)])

nn = Network([784, 100, 100, 100, 10])

num_epochs = 100
batch_size = 100
learning_rate = 0.01

for epoch in range(num_epochs):
    indices = np.random.permutation(len(datapoints))

    # Create mini-batches
    for start_idx in range(0, len(datapoints), batch_size):
        end_idx = min(start_idx + batch_size, len(datapoints))
        batch_indices = indices[start_idx:end_idx]
        batch = [datapoints[i] for i in batch_indices]

        # Train on this batch for one update
        nn.calculate_output(batch)
        nn.backward(batch, learning_rate)

    # Optionally print loss after each epoch
    if epoch % 10 == 0:
        np.savez('../MLP-from-scratch-python/models/weights.npy', nn.layers[0].weights, nn.layers[1].weights, nn.layers[2].weights, nn.layers[3].weights)
        np.savez('../MLP-from-scratch-python/models/biases.npy', nn.layers[0].biases, nn.layers[1].biases,
                 nn.layers[2].biases, nn.layers[3].biases)
        print(f"Epoch {epoch}, Loss: {nn.loss(datapoints)}")


import numpy as np
from matplotlib import pyplot
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

for i in range(9):  
    pyplot.subplot(330 + 1 + i)  # this creates a 3X3 grid
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('Pastel1')) # plucks out the indexed image to display
pyplot.show()
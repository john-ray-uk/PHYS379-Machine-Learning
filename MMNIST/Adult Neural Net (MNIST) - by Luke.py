import numpy as np
import matplotlib.pyplot as plt
import nnfs

from nnfs.datasets import spiral_data
from keras.datasets import mnist

nnfs.init()

### INSTRUCTIONS ###

### Here is the Neural Network for the MNIST dataset. Should run perfectly on all python versions just ensure you install all the required libraries ###
### Make sure to pip install: numpy, nnfs, keras and tensorflow <---- (Not shown in imports above) ###

### IMPORTANT: if you run this on your own laptop please adjust the number of epochs, it is currently set to 1000! ###
### Your laptop will be fine but the code will take ages (like 30 mins) I recommend E = 100 or 500 (E is on line 162)###


# Importing MNIST Dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Convert from 28X28 data to a vector 784 entries long
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

# Squishing into values between 0 and 1 (Normalisation)
train_X = train_X / 255.0
test_X = test_X / 255.0

# OLD TRAINING DATA, NO LONGER USED 

X = [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12]]

X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs,n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
        self.inputs = inputs
    
    # This function updates weights and biases values (derivatives happen within the specific class)

    def backward(self, dvalues):
        self.dw = np.dot(self.inputs.T, dvalues)
        self.db = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        # Modifying a variable, so copy the original variable before change
        self.dinputs = dvalues.copy()
        # If input value is -ve then we set the gradient to 0 (i.e. no impact)
        # This part is the main reason why ReLU is so popular, because the derivative is literally 1 line that just states yes or no
        self.dinputs[self.input <= 0] = 0
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # this "-max" prevents overflow errors
        probabilties = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilties

    # Its honestly really difficult to follow what's happening here but its basically just a dervative matrix (i.e. the Jacobian matrix)

    def backward(self, dvalues):
        # create and empty array with same shape of our inputs values
        self.dinputs = np.empty_like(dvalues)

        for index,  (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y) 
        data_loss = np.mean(sample_losses) # averages the losses for each class aka how wrong it was in each category
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) # This prevents doing log(0) which is undefined 

        # Eliminating the the useless values of the predictions of wrong answers
        # All this does is grab the probabilties for correct answers

        if len(y_true.shape) == 1: # This line compares the vector answer and matrix of predictions
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # This line compares a matrix of vector answers and matrix of predictions
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        labels = len(dvalues[0])
        # Shape thing similar to above, but this time we create a matrix of 0's for false and 1's for true
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true] # np.eye creates identity 2D matrix for any arbitrary shape
        # Gradient
        self.dinputs = -y_true / dvalues
        # Normalise
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalLossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, a_rate=0.01):
        self.a_rate = a_rate

    def update_params(self, layer):
        layer.weights += -self.a_rate * layer.dw
        layer.bias += -self.a_rate * layer.db

# Layer Construction Zone
layer1 = Layer_Dense(784,128)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(128,128)
activation2 = Activation_ReLU()

layer3 = Layer_Dense(128,10)
loss_activation = Activation_Softmax_Loss_CategoricalLossEntropy()

# Optimizer object
optimizer = Optimizer_SGD()

# Training Loop
E = 1000
batch_size = 128

for E in range(E):

    # Shuffle every loop!
    indices = np.arange(len(train_X))
    np.random.shuffle(indices)
    train_X = train_X[indices]
    train_y = train_y[indices]

    for step in range(0, len(train_X), batch_size):

        # Break dataset into 'mini-batches'
        batch_X = train_X[step:step+batch_size]
        batch_y = train_y[step:step+batch_size]

        # Passing through layers
        layer1.forward(batch_X)
        activation1.forward(layer1.output)

        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        layer3.forward(activation2.output)

        # Pass through SoftMax and Loss calculation
        loss = loss_activation.forward(layer3.output, batch_y) # (Neural network output and answer matrix (y))

        # Accuracy Calculations
        Best_Guess = np.argmax(loss_activation.output, axis=1)
        if len(batch_y.shape) == 2:
            batch_y = np.argmax(batch_y, axis=1)

        accuracy = np.mean(Best_Guess == batch_y)


        # Backwards Pass
        loss_activation.backward(loss_activation.output, batch_y)
        layer3.backward(loss_activation.dinputs)
        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        # Optimizer updates
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.update_params(layer3)

    if not E % 100:
        print(f'epoch:{E}, accuracy:{accuracy:.3f}, loss:{loss:.3f}')

# Test time!

layer1.forward(test_X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
loss = loss_activation.forward(layer3.output, test_y)

predictions = np.argmax(loss_activation.output, axis=1)
accuracy = np.mean(predictions == test_y)

print(f"Test Score:{accuracy:.3f}")

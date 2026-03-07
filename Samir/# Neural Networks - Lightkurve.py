# Neural Networks - returns - Phys379
#----------------------------------------------------------

# Imports

'''
- sys for checking versions
- numpy for array manipulation
- matplotlib for plotting
- pandas for data loading
'''
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas
print("Python: ", sys.version)
print("Matplotlib: ", mpl.__version__)
print("Numpy: ", np.__version__)

# Import lightkurve for Kepler lightcurve data
from lightkurve import search_lightcurve


#----------------------------------------------------------

# Implementing Kepler light curve data

# Download light curve for a Kepler target
# You can change the target name to another star
lc_collection = search_lightcurve("Kepler-10", mission="Kepler").download_all()

# Flatten all sectors/segments into one array
flux = np.hstack([lc.flux.value for lc in lc_collection if lc.flux is not None])

# Create a dataframe with daily (cadence) flux values
data = pandas.DataFrame({"flux": flux})

# ----------------------------------------------------------

# Compute 5‑step forward changes directly

forward_steps = 5
data['Return'] = data['flux'].shift(-forward_steps) - data['flux']
data = data.dropna()

# Inputs: previous flux values
inputs = data[['flux']].values

# Targets: 5‑step forward flux change
targets = data[['Return']].values

# ----------------------------------------------------------

# Train/Test Split (Chronological)

split_ratio = 0.8
split_index = int(len(inputs) * split_ratio)
X_train = inputs[:split_index]
X_test = inputs[split_index:]
y_train = targets[:split_index] # Targets for training and testing
y_test = targets[split_index:]
time_index = data.index[split_index:] # Splitting the time index for plotting

# Print dataset sizes
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

#----------------------------------------------------------

# Data Normalization

# Normalize Inputs (training statistics only)
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_std[X_std == 0] = 1e-8 # Prevent division by zero
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Normalize Targets (returns are often small, but normalization helps stability)
y_mean = np.mean(y_train)
y_std = np.std(y_train)
if y_std == 0:
    y_std = 1e-8
y_train = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std


#----------------------------------------------------------

# Layer Class - altered for He initialization

class Layer:
    """
    Fully connected layer.
    Parameters:
    nInputs : int
        Number of input features
    nNeurons : int
        Number of neurons in layer
    """
    def __init__(self, nInputs, nNeurons):
        '''
        Mathematical operation:
        Z = X · W + b
        - Z : output of layer
        - X : input data
        - W : weights
        - b : biases
        '''
        # Connections = weights
        self.weights = np.random.randn(nInputs, nNeurons) * np.sqrt(2/nInputs)
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, learning_rate):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases


#----------------------------------------------------------

# ReLU Activation Class

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


#----------------------------------------------------------

# Loss Classes

class Loss:
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss

class MeanSquaredErrorLoss(Loss):
    def forward(self, yPred, yTrue):
        self.yPred = yPred
        self.yTrue = yTrue
        sampleLosses = (yTrue - yPred) ** 2
        return np.mean(sampleLosses, axis=-1)
    def backward(self):
        n_samples = self.yTrue.shape[0]
        n_outputs = self.yTrue.shape[1]
        self.dinputs = -2 * (self.yTrue - self.yPred) / n_outputs
        self.dinputs /= n_samples

#----------------------------------------------------------

# Network Construction

layer1 = Layer(1, 8)
activation1 = ReLU()
layer3 = Layer(8, 8)
activation3 = ReLU()
layer2 = Layer(8, 1)

#----------------------------------------------------------

# Loss Calculation (MSE)

lossFunc = MeanSquaredErrorLoss()
loss_history = []

#----------------------------------------------------------

# Training Loop

learning_rate = 0.0005
epochs = 1000

for epoch in range(epochs):
    layer1.forward(X_train)
    activation1.forward(layer1.output)
    layer3.forward(activation1.output)
    activation3.forward(layer3.output)
    layer2.forward(activation3.output)
    output = layer2.output

    loss = lossFunc.calculate(output, y_train)
    loss_history.append(loss)

    lossFunc.backward()
    layer2.backward(lossFunc.dinputs, learning_rate)
    activation3.backward(layer2.dinputs)
    layer3.backward(activation3.dinputs, learning_rate)
    activation1.backward(layer3.dinputs)
    layer1.backward(activation1.dinputs, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

#----------------------------------------------------------

# Test Evaluation

layer1.forward(X_test)
activation1.forward(layer1.output)
layer3.forward(activation1.output)
activation3.forward(layer3.output)
layer2.forward(activation3.output)

predictions_scaled = layer2.output
predictions = predictions_scaled * y_std + y_mean

test_mse = np.mean((y_test - predictions) ** 2)
test_rmse = np.sqrt(test_mse)
average_return = np.mean(y_test)
percent_error = (test_rmse / (np.std(y_test)+1e-8)) * 100

print("\nTest MSE:", test_mse)
print("Test RMSE (Flux Change):", test_rmse)
print("Average Return (Test):", average_return)
print("Percent Error:", percent_error, "%")

#----------------------------------------------------------
# PLOTTING
#----------------------------------------------------------

rmse_history = [np.sqrt(l) for l in loss_history]
plt.figure()
plt.plot(rmse_history)
plt.title("Training RMSE vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(time_index, y_test.flatten(), label="True 5‑step Flux Change", color='orange')
plt.plot(time_index, predictions.flatten(), label="Predicted 5‑step Flux Change", color='red', alpha=0.7)
plt.title("True vs Predicted 5‑Step Flux Change")
plt.xlabel("Index")
plt.ylabel("Flux Change")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test.flatten(), predictions.flatten(), alpha=0.5)
plt.plot([-np.max(y_test), np.max(y_test)], [-np.max(y_test), np.max(y_test)], color='red', linestyle='--')
plt.title("True vs Predicted Flux Change (Scatter)")
plt.xlabel("True Change")
plt.ylabel("Predicted Change")
plt.grid(True)
plt.show()


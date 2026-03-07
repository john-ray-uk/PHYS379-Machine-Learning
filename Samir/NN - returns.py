# Neural Networks - returns - Phys379
#----------------------------------------------------------

# Imports

'''
- sys for checking versions
- numpy for array manipulation
- matplotlib for plotting
- yfinance for financial data
'''
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas
import yfinance as yf
print("Python: ", sys.version)
print("Matplotlib: ", mpl.__version__)
print("Numpy: ", np.__version__)


#----------------------------------------------------------

# Implementing yfinance data

# Download stock data
ticker = "NVDA" 
'''
ticker can be any stock symbol, for example:
- 1 - AAPL: Apple Inc.
- 2 - GOOGL: Alphabet Inc. (Google)
- 3 - MSFT: Microsoft Corporation
- 4 - AMZN: Amazon.com, Inc.
- 5 - TSLA: Tesla, Inc.
- 6 - FB: Meta Platforms, Inc. (Facebook)
- 7 - NFLX: Netflix, Inc.
- 8 - NVDA: NVIDIA Corporation
- 9 - JPM: JPMorgan Chase & Co.
- 10 - V: Visa Inc.
'''
period = "10y"
interval = "1d"
data = yf.download(ticker, period=period, interval=interval)
# Keep only relevant columns
data = data[['Open', 'High', 'Low', 'Close']]


# ----------------------------------------------------------

# Compute 5-day forward returns directly

forward_days = 5
data['Return'] = (data['Close'].shift(-forward_days) - data['Close']) / data['Close']
data = data.dropna()

# Inputs: Open, High, Low, Close
inputs = data[['Open', 'High', 'Low', 'Close']].values
# Targets: 5-day forward returns
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
        self.weights = np.random.randn(nInputs, nNeurons) * np.sqrt(2/nInputs) # He initialization - scales weights to improve convergence with ReLU.
        # Neurons = biases
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        '''
        Performs forward propagation.
        z_j = Σ_i (x_i * w_ij) + b_j
        - x_i  : input feature i
        - w_ij : weight connecting input i to neuron j
        - b_j  : bias term for neuron j
        - z_j  : linear combination (pre-activation output)
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, learning_rate):
        '''
        Performs backward propagation and updates weights and biases.
        - dvalues : gradient of loss with respect to layer output
        - learning_rate (L) : step size for parameter updates
        Mathematical operations - gradients:
        - dL/dw_ij = Σ_k (dL/dz_k * dz_k/dw_ij)
        - dL/db_j = Σ_k (dL/dz_k * dz_k/db_j)
        - dL/dx_i = Σ_k (dL/dz_k * dz_k/dx_i)
        Mathematical operations - Gradient Descent:
        - w_ij = w_ij - learning_rate * dL/dw_ij
        - b_j = b_j - learning_rate * dL/db_j
        '''
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        # Gradient Descent update
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases


#----------------------------------------------------------

# ReLU Activation Class

class ReLU:
    """
    ReLU Activation Function: max(0, x)
    ReLU(x) = max(0, x)
    - Introduces nonlinearity
    - Keeps positive signals
    - Zeros negative signals
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


#----------------------------------------------------------

# Loss Base Class

class Loss:
    """
    Base loss class
    """
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss


#----------------------------------------------------------

# Mean Squared Error Loss Class

class MeanSquaredErrorLoss(Loss):
    """
    Mean Squared Error Loss
    - MSE is the variance of the residuals (prediction errors)
    - RMSE is the standard deviation: the average of the squared differences between predicted and true values
    MSE = (1/n) Σ_i (y_true - y_pred)^2
    - y_true : array of true target values
    - y_pred : array of predicted values from the model
    - n      : number of samples
    """
    def forward(self, yPred, yTrue):
        self.yPred = yPred
        self.yTrue = yTrue
        sampleLosses = (yTrue - yPred) ** 2
        return np.mean(sampleLosses, axis=-1)

    def backward(self):
        n_samples = self.yTrue.shape[0]
        n_outputs = self.yTrue.shape[1]
        # dL/dyPred
        self.dinputs = -2 * (self.yTrue - self.yPred) / n_outputs
        self.dinputs = self.dinputs / n_samples


#----------------------------------------------------------

# Network Construction

# First hidden layer: 4 inputs → 8 neurons
layer1 = Layer(4, 8)
activation1 = ReLU()
#second hidden layer: 8 neurons → 8 neurons
layer3 = Layer(8, 8)
activation3 = ReLU()
# Output layer: 8 neurons → 1 output (regression)
layer2 = Layer(8, 1)  # Output = predicted return


#----------------------------------------------------------

# Loss Calculation (MSE)

lossFunc = MeanSquaredErrorLoss()
loss_history = []


#----------------------------------------------------------

# Training Loop

learning_rate = 0.0005
epochs = 1000
for epoch in range(epochs):
    # Forward pass: Input > Layer 1 > ReLU > Layer 2 > Output
    layer1.forward(X_train)
    activation1.forward(layer1.output)
    layer3.forward(activation1.output)
    activation3.forward(layer3.output)
    layer2.forward(activation3.output)
    output = layer2.output

    # Loss
    loss = lossFunc.calculate(output, y_train)
    loss_history.append(loss)

    # Backward pass: Output > Layer 2 > ReLU > Layer 1 > Input
    lossFunc.backward()
    layer2.backward(lossFunc.dinputs, learning_rate)
    activation3.backward(layer2.dinputs)
    layer3.backward(activation3.dinputs, learning_rate)
    activation1.backward(layer3.dinputs)
    layer1.backward(activation1.dinputs, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")


#----------------------------------------------------------

# Test Evaluation - Testing the model on unseen data

layer1.forward(X_test)
activation1.forward(layer1.output)
layer3.forward(activation1.output)
activation3.forward(layer3.output)
layer2.forward(activation3.output)
predictions_scaled = layer2.output

# Convert predictions back to actual returns
predictions = predictions_scaled * y_std + y_mean

# Compute test error
test_mse = np.mean((y_test - predictions) ** 2)
test_rmse = np.sqrt(test_mse)
average_return = np.mean(y_test)
percent_error = (test_rmse / (np.std(y_test)+1e-8)) * 100

print("\nTest MSE:", test_mse)
print("Test RMSE (Return Fraction):", test_rmse)
print("Average Return (Test):", average_return)
print("Percent Error:", percent_error, "%")


#----------------------------------------------------------
#PLOTTING
#----------------------------------------------------------

# Plot 1: Training RMSE vs Epochs
'''
Shows whether the model is learning patterns in price movements.
- If the loss decreases over epochs, the model is learning
- If the loss plateaus, the model may have converged
- If the loss increases, the model may be diverging (too high learning rate)
- since loss is the standard deviation, rmse is the variance
'''

rmse_history = [np.sqrt(l) for l in loss_history]
plt.figure()
plt.plot(rmse_history)
plt.title("Training RMSE vs Epochs (Scaled Units)")
plt.xlabel("Epoch")
plt.ylabel("RMSE (No. standard deviations)")
plt.xticks(rotation=45)
plt.legend(["Training RMSE"])
plt.grid(True)
plt.show()


#----------------------------------------------------------

# Plot 2: True vs Predicted Returns
'''
Shows how well the model's predictions match the actual returns.
- If lines overlap closely, predictions are accurate
- If lines diverge, predictions are poor
- The scale of the plot indicates the magnitude of returns and errors
'''
plt.figure(figsize=(12,5))
plt.plot(time_index, y_test.flatten(), label="True 5-Day Forward Return", color='orange')
plt.plot(time_index, predictions.flatten(), label="Predicted 5-Day Forward Return", color='red', alpha=0.7)
plt.title("5-Day Forward Returns: True vs Predicted")
plt.xlabel("Date")
plt.ylabel("Return (Fractional)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()


#----------------------------------------------------------

# PLOT 3 — Scatter Plot: True vs Predicted Returns
'''
Shows the correlation between true and predicted returns.
- If points cluster around the diagonal line, predictions are accurate
- If points are widely scattered, predictions are poor
- The spread of points indicates the variance of errors
- Outliers indicate specific instances where the model performed poorly
'''
plt.figure(figsize=(8,6))
plt.scatter(y_test.flatten(), predictions.flatten(), alpha=0.5)
plt.plot([-0.1, 0.1], [-0.1, 0.1], color='red', linestyle='--') # Diagonal line for reference
plt.title("True vs Predicted Returns (Scatter)")
plt.xlabel("True Return")
plt.ylabel("Predicted Return")
plt.legend(["Predictions", "Perfect Prediction Line"])
plt.grid(True)
plt.show()

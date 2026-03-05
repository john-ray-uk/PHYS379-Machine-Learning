# Neural Networks - Phys379
#----------------------------------------------------------

'''
Imports:
- sys for checking versions
- numpy for array manipulation
- matplotlib for plotting
- yfinance for financial data (future use)
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

'''
# Test Data (Future -> Stock Regression Data)

# Input layer data (3 samples, 4 features)
inputs = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
])

# Target outputs (continuous values -> % change)
y = np.array([
    [0.5],
    [1.2],
    [-0.3]
])
'''
    
#----------------------------------------------------------

# Implementing yfinance data

# Download stock data
ticker = "AAPL"   # You can change this
'''
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
period = "10y" # choose from: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
interval = "1d" # choose from: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
data = yf.download(ticker, period=period, interval=interval)
# Keep only relevant columns
data = data[['Open', 'High', 'Low', 'Close']]
# Target = Close price 5 days ahead
forward_days = 5
data['Target_Close'] = data['Close'].shift(-forward_days)
data = data.dropna()
inputs = data[['Open', 'High', 'Low', 'Close']].values
targets = data[['Target_Close']].values


# ----------------------------------------------------------

# Train/Test Split (Chronological)

# Split data into training and testing sets (80% train, 20% test)
split_ratio = 0.8
split_index = int(len(inputs) * split_ratio)
# For time series data, we do not shuffle - we take the first 80% as training and the last 20% as testing
X_train = inputs[:split_index]
X_test = inputs[split_index:]
# Targets for training and testing
y_train = targets[:split_index]
y_test = targets[split_index:]
#splitting the time index for plotting
time_index = data.index[split_index:]
# Print dataset sizes
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


#----------------------------------------------------------

# Data Normalization

# Normalize Inputs (using training statistics only)
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_std[X_std == 0] = 1e-8  # Prevent division by zero
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Normalize Targets (CRUCIAL for stability)
y_mean = np.mean(y_train)
y_std = np.std(y_train)
if y_std == 0:
    y_std = 1e-8  # Prevent division by zero
y_train = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std


#----------------------------------------------------------

#Notes
'''

'''

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
        self.weights = np.random.randn(nInputs, nNeurons) * np.sqrt(2/nInputs) # Different initialization for Leaky ReLU
        # Neurons = biases
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        '''
        Performs forward propagation.
        Mathematical operation:
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
        # Gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        # Update parameters - Gradient Descent
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases


#----------------------------------------------------------

# ReLU Activation Class

class ReLU:
    """
    ReLU Activation Function.
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
    Provides method to calculate mean loss across all samples.
    """

    def calculate(self, output, y):
        """
        Computes mean loss across all samples.
        """
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss


#----------------------------------------------------------

# Mean Squared Error Loss Class

class MeanSquaredErrorLoss(Loss):
    """
    Mean Squared Error Loss: tells us how wrong the predictions are
    - it is the standard deviation: the average of the squared differences between predicted and true values
    MSE = (1/n) Σ_i (y_true - y_pred)^2
    - y_true : array of true target values
    - y_pred : array of predicted values from the model
    - n      : number of samples
    """

    def forward(self, yPred, yTrue):
        self.yPred = yPred
        self.yTrue = yTrue
        # Squared differences
        sampleLosses = (yTrue - yPred) ** 2
        # Mean across output neurons
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
# Output layer: 8 neurons → 1 output (regression)
layer2 = Layer(8, 1)


#----------------------------------------------------------

# Loss Calculation (MSE)

lossFunc = MeanSquaredErrorLoss()
loss_history = []


#----------------------------------------------------------

# Training Loop

learning_rate = 0.0005 # Adjusted learning rate for stability
epochs = 1000
for epoch in range(epochs):
    # Forward Pass: Input > Layer 1 > ReLU > Layer 2 > Output
    layer1.forward(X_train)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    output = layer2.output
    # Loss
    loss = lossFunc.calculate(output, y_train)
    loss_history.append(loss)
    # Backward Pass: Output > Layer 2 > ReLU > Layer 1 > Input
    lossFunc.backward()
    layer2.backward(lossFunc.dinputs, learning_rate)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs, learning_rate)
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")


#----------------------------------------------------------

# Final Output

layer1.forward(X_test)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
test_loss = lossFunc.calculate(layer2.output, y_test_scaled)
print("\nFinal Test MSE (scaled):")
print(test_loss)
print("\nFinal Network Output:")
print(layer2.output)
print("\nFinal MSE Loss:")


# ----------------------------------------------------------

# Plot 1: Training RMSE (scaled) vs Epochs
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
plt.show()


# ----------------------------------------------------------

# Test Evaluation

layer1.forward(X_test) # Applies linear transformation: Z = X · W + b
activation1.forward(layer1.output) # Applies ReLu
layer2.forward(activation1.output) # Final linear transformation to get predictions in scaled units
predictions_scaled = layer2.output # Predicts normalised prices

# Convert predictions back to real price units
predictions = predictions_scaled * y_std + y_mean

# Compute 5-day forward returns for Test Set
true_returns = (y_test[1:] - y_test[:-1]) / y_test[:-1] # percentage return for true values
predicted_returns = (predictions[1:] - predictions[:-1]) / predictions[:-1] # percentage return for predicted values

# Align time index for returns
time_index_returns = time_index[1:] # when you compute returns, you lose the first day, so we shift the time index accordingly

# Compute test error in real units
test_mse = np.mean((y_test - predictions) ** 2)
test_rmse = np.sqrt(test_mse)
average_price = np.mean(data['Close'].values[split_index:])
percent_error = (test_rmse / average_price) * 100

print("\nTest MSE:", test_mse)
print("Test RMSE (£):", test_rmse)
print("Average Stock Price (Test):", average_price)
print("Percent Error:", percent_error, "%")

# ----------------------------------------------------------

# Plot 2: True vs Predicted Price
'''
Shows whether predictions track actual returns.
- If points cluster around the diagonal line, predictions are accurate
- If points are widely scattered, predictions are poor
- Outliers indicate specific samples where the model performs poorly
'''
# Forward once more to ensure latest output
time_index = data.index[split_index:]

plt.figure()
plt.plot(time_index, y_test.flatten(), label="True Close Price")
plt.plot(time_index, predictions.flatten(), label="Predicted Close Price")
plt.title("5-Year Daily 5-Day Forward Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price (£)")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# ----------------------------------------------------------

# Plot 3: True vs Predicted Returns
plt.figure(figsize=(12,5))
plt.plot(time_index_returns, true_returns.flatten(), label="True 5-Day Forward Return", color='orange')
plt.plot(time_index_returns, predicted_returns.flatten(), label="Predicted 5-Day Forward Return", color='red', alpha=0.7)
plt.title("5-Day Forward Returns: True vs Predicted")
plt.xlabel("Date")
plt.ylabel("Return (Fractional)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()




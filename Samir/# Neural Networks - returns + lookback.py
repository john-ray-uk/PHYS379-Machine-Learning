# Neural Networks - returns with Lookback and Directional Accuracy
#----------------------------------------------------------

# Imports
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas
import yfinance as yf
print("Python: ", sys.version)
print("Matplotlib: ", mpl.__version__)
print("Numpy: ", np.__version__)

np.random.seed(42)  # For reproducibility of results


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


#----------------------------------------------------------

# Compute 5-day forward returns directly
forward_days = 1
data['Return'] = (data['Close'].shift(-forward_days) - data['Close']) / data['Close']

# Direction target for later use
data['Direction'] = (data['Return'] > 0).astype(int)

data = data.dropna()


#----------------------------------------------------------

# Lookback window
'''
Lookback window allows the model to learn from patterns in past price movements.
lookback = 5 days: model will use the past 5 days of OHLC data to predict the return 5 days in the future
This captures short-term trends and patterns that may influence future returns
'''

lookback = 5  # Number of past days to include as input features

X, y, direction = [], [], []

for i in range(lookback, len(data) - forward_days):
    '''
    Iterates over all rows where we have enough lookback history and a valid forward return:
    - For each row, we take the past x days of data and 'flatten' it into a single feature vector
    - The target 'y' is the forward return for that row
    - The 'direction' is whether the return is pos (1) or neg (0), which is used for directional accuracy
    Working:
    - iloc is used to select the DataFrame based on integer location
    - i-lookback:i picks the last lookback rows before the current row i
    - data[['Open','High','Low','Close']] selects only the relevant columns
    - .values converts the DataFrame slice to a NumPy array
    - .flatten() converts the 2D array (lookback × 4) into a 1D vector of size lookback*4 (here 5*4 = 20)
    '''
    X.append(data[['Open','High','Low','Close']].iloc[i-lookback:i].values.flatten())
    y.append(data['Return'].iloc[i])
    direction.append(data['Direction'].iloc[i])

X = np.array(X)
y = np.array(y).reshape(-1,1)
direction = np.array(direction).reshape(-1,1)


#----------------------------------------------------------
# Train/Test Split (Chronological)

split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]  # Targets for training and testing
y_test = y[split_index:]
direction_test = direction[split_index:]
time_index = data.index[lookback + split_index : lookback + split_index + len(y_test)]  # Adjust index for plotting

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


#----------------------------------------------------------
# Data Normalization

# Normalize Inputs (training statistics only)
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_std[X_std == 0] = 1e-8  # Prevent division by zero
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
        self.weights = np.random.randn(nInputs, nNeurons) * np.sqrt(2/nInputs)  # He initialization
        # Neurons = biases
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        '''
        Performs forward propagation.
        z_j = Σ_i (x_i * w_ij) + b_j
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, learning_rate):
        '''
        Performs backward propagation and updates weights and biases.
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
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


#----------------------------------------------------------
# Loss Classes

class Loss:
    """
    Base loss class
    """
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss

class MeanSquaredErrorLoss(Loss):
    """
    Mean Squared Error Loss
    """
    def forward(self, yPred, yTrue):
        self.yPred = yPred
        self.yTrue = yTrue
        sampleLosses = (yTrue - yPred) ** 2
        return np.mean(sampleLosses, axis=-1)

    def backward(self):
        n_samples = self.yTrue.shape[0]
        n_outputs = self.yTrue.shape[1]
        self.dinputs = -2 * (self.yTrue - self.yPred) / n_outputs
        self.dinputs = self.dinputs / n_samples


#----------------------------------------------------------
# Network Construction
'''
constructing a simple feedforward neural network with:
- Input layer: size = 4 * lookback (20 features for 5 days of OHLC)
- Hidden layer 1: 8 neurons, ReLU activation
- Hidden layer 2: 8 neurons, ReLU activation
- Output layer: 1 neuron (predicted return), no activation (regression)
'''

layer1 = Layer(4*lookback, 8)  # Input layer size adjusted for lookback
activation1 = ReLU()

layer3 = Layer(8, 8)
activation3 = ReLU()

layer2 = Layer(8, 1)  # Output layer


#----------------------------------------------------------
# Abstracted Forward Pass Function

def forward_pass(X):
    """
    Performs a full forward pass through the network.
    """
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer3.forward(activation1.output)
    activation3.forward(layer3.output)
    layer2.forward(activation3.output)
    return layer2.output


#----------------------------------------------------------
# Training Loop#
'''
For each epoch:
- Perform forward pass to get predictions
- Calculate loss using MSE
- Perform backward pass to compute gradients
- Update weights and biases using gradients and learning rate
- Print loss every 100 epochs to monitor training progress
'''

lossFunc = MeanSquaredErrorLoss()
loss_history = []

learning_rate = 0.0005
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    output = forward_pass(X_train)

    # Loss
    loss = lossFunc.calculate(output, y_train)
    loss_history.append(loss)

    # Backward pass
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

'''
Scaled predictions are in terms of standard deviations from the mean return.
To convert back to actual return values, we reverse the normalization:
'''
predictions_scaled = forward_pass(X_test)
predictions = predictions_scaled * y_std + y_mean

# Directional Accuracy
'''
Directional accuracy measures how well the model predicts the direction of price movement (up or down).
- If predicted return > 0, predict "up" (1)
- If predicted return <= 0, predict "down" (0)
- Compare predicted direction to actual direction to compute accuracy
'''
predicted_direction = (predictions > 0).astype(int)
directional_accuracy = np.mean(predicted_direction == direction_test)

# Compute test error
'''
- MSE gives the average squared error in return predictions
- RMSE gives the standard deviation of prediction errors in return units
'''
test_mse = np.mean((y_test - predictions) ** 2)
test_rmse = np.sqrt(test_mse)
average_return = np.mean(y_test)
percent_error = (test_rmse / (np.std(y_test)+1e-8)) * 100

print("\nTest MSE:", test_mse)
print("Test RMSE (Return Fraction):", test_rmse)
print("Average Return (Test):", average_return)
print("Percent Error:", percent_error, "%")
print("Directional Accuracy:", directional_accuracy*100, "%")



#----------------------------------------------------------
# PLOTTING
#----------------------------------------------------------

# Plot 1: Training RMSE vs Epochs
'''
Shows whether the model is learning patterns in price movements.
- If the loss decreases over epochs, the model is learning
- If the loss plateaus, the model may have converged
- If the loss increases, the model may be diverging (too high learning rate)
- Since loss is the standard deviation, RMSE is the variance
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

#------------------------------------------
# Plot 2: True vs Predicted Returns
'''
Shows how well the model's predictions match the actual returns.
- If lines overlap closely, predictions are accurate
- If lines diverge, predictions are poor
- The scale of the plot indicates the magnitude of returns and errors
'''
plt.figure(figsize=(12,5))
plt.plot(time_index, y_test.flatten(), label="True " + str(forward_days) + "-Day Forward Return", color='orange')
plt.plot(time_index, predictions.flatten(), label="Predicted " + str(forward_days) + "-Day Forward Return", color='red', alpha=0.7)
plt.title("True vs Predicted " + str(forward_days) + "-Day Forward Returns: " + str(ticker))
plt.xlabel("Date")
plt.ylabel("Return (Fractional)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

#--------------------------------------------
# Plot 3: Scatter Plot True vs Predicted
'''
Shows the correlation between true and predicted returns.
- If points cluster around the diagonal line, predictions are accurate
- If points are widely scattered, predictions are poor
- The spread of points indicates the variance of errors
- Outliers indicate specific instances where the model performed poorly
'''
plt.figure(figsize=(8,6))
plt.scatter(y_test.flatten(), predictions.flatten(), alpha=0.5)
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')  # Perfect prediction line
plt.title("True vs Predicted Returns (Scatter)")
plt.xlabel("True Return")
plt.ylabel("Predicted Return")
plt.legend(["Predictions","Perfect Prediction Line"])
plt.grid(True)
plt.show()
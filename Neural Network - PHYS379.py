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
import yfinance as yf
print("Python: ", sys.version)
print("Matplotlib: ", mpl.__version__)
print("Numpy: ", np.__version__)

#----------------------------------------------------------

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

#----------------------------------------------------------

#Notes
'''

'''

#----------------------------------------------------------

# Layer Class
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
        self.weights = np.random.randn(nInputs, nNeurons) * 0.01
        # Neurons = biases
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        """
        Performs forward propagation.
        Mathematical operation:
        z_j = Σ_i (x_i * w_ij) + b_j
        - x_i  : input feature i
        - w_ij : weight connecting input i to neuron j
        - b_j  : bias term for neuron j
        - z_j  : linear combination (pre-activation output)
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

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
    Mean Squared Error Loss.
    MSE = (1/n) Σ_i (y_true - y_pred)^2
    - y_true : array of true target values
    - y_pred : array of predicted values from the model
    - n      : number of samples
    """

    def forward(self, yPred, yTrue):
        # Squared differences
        sampleLosses = (yTrue - yPred) ** 2
        # Mean across output neurons
        return np.mean(sampleLosses, axis=-1)

#----------------------------------------------------------

# Network Construction
# First hidden layer: 4 inputs → 5 neurons
layer1 = Layer(4, 5)
activation1 = ReLU()

# Output layer: 5 neurons → 1 output (regression)
layer2 = Layer(5, 1)

#----------------------------------------------------------

# Forward Pass
# Input → Layer 1 → ReLU → Layer 2 → Output
'''
X > Z_1 > A_1 ​> Z_2 ​> y^
- Z_1 : output of layer 1 (pre-activation)
- A_1 : output of activation 1 (post-activation)
- Z_2 : output of layer 2 (final output, linear activation)
'''
layer1.forward(inputs)
activation1.forward(layer1.output)

layer2.forward(activation1.output)

# Final output (linear activation)
output = layer2.output

#----------------------------------------------------------

# Loss Calculation (MSE)
'''
Tells us how wrong the predictions are
    MSE = (1/n) Σ_i (y_true - y_pred)^2
    - y_true : array of true target values
    - y_pred : array of predicted values from the model
    - n      : number of samples
'''
lossFunc = MeanSquaredErrorLoss()
loss = lossFunc.calculate(output, y)

print("Network Output:\n", output)
print("MSE Loss:", loss)
import numpy as np
# Prevents "random" numbers from changing each time this file is loaded
np.random.seed(0)

"""
Layer, MSE, and Neural Network Classes code:

Includes Sigmoid & ReLU Activation functions, a function to train the network, and functions
to find the derivative of the activation functions and use the derivatives to update any
parameters in the network; using a process called "backpropagation".

There are separate classes for the MSE and Layers of the network, which are then called into 
the neural network class.

This code is redesigned to work with an N-input GUI.

"""

class Layer:
    # Defines a default layer with a Sigmoid activation function.
    def __init__(self, inputSize, outputSize, activation='Sigmoid'):
        # Weights are normalised differently for ReLU vs Sigmoid / other activation functions.
        if activation == "ReLU":
            self.weights = np.random.randn(inputSize, outputSize) * np.sqrt(2/inputSize)
            # Initialises the bias as small, but non-zero.
            self.bias = np.ones((1, outputSize)) * 0.01
        else:
            self.weights = np.random.randn(inputSize, outputSize) * np.sqrt(1/inputSize)
            self.bias = np.zeros((1, outputSize))

        # Initialises variables. I/O is set to NoneType, for now.
        self.activationType = activation
        self.input = None
        self.output = None
        self.initInput = None

    # Activation function (varies depending on self.activationType)
    def activate(self, x):
        # Takes an input (x) for backpropagation.
        self.input = x
        # Only uses LINEAR REGRESSION, for now.
        self.initInput = np.dot(x, self.weights) + self.bias
        # Chooses activation function type (defaults to Sigmoid)
        if self.activationType == 'Sigmoid':
            self.output = 1 / (1 + np.exp(-self.initInput))
        elif self.activationType == "ReLU":
            self.output = np.maximum(0, self.initInput)
        elif self.activationType == "Linear":
            self.output = self.initInput
        elif self.activationType == "Tanh":
            self.output = np.tanh(self.initInput)
        else:
            print("Error, incorrect activation input. Set to default method: Sigmoid.")
            self.output = 1 / (1 + np.exp(-self.initInput))
        return self.output

    # Activation derivative function (varies, like previously)
    def activationDerivative(self, x):
        # This is used for gradient descent, with x as the initial input.
        if self.activationType == 'Sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif self.activationType == "ReLU":
            return (x > 0).astype(float)
        elif self.activationType == "Linear":
            return np.ones_like(x)
        elif self.activationType == "Tanh":
            t = np.tanh(x)
            return 1 - t**2
        else:
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)

# Works easiest when in a separate class, like in tutorials.
class MeanSquaredError:
    # Function for forward propagation of the MSE
    def forward(self, yPred, yTrue):
        self.yPred = yPred
        self.yTrue = yTrue
        # Definition of MSE
        sampleLosses = np.mean((yTrue - yPred) ** 2, axis=-1)
        return sampleLosses

    def calculate(self, yPred, yTrue):
        sampleLosses = self.forward(yPred, yTrue)
        return np.mean(sampleLosses)
    # Function for backward propagation of the MSE
    def backward(self):
        nSamples = self.yTrue.shape[0]
        nOutputs = self.yTrue.shape[1] if self.yTrue.ndim > 1 else 1
        # Derivative of MSE
        self.dinputs = 2.0 * (self.yPred - self.yTrue) / (nSamples * nOutputs)
        return self.dinputs
        
# Neural Network class. Both the MSE and Layer classes are called into this.
class NeuralNetwork:
    def __init__(self, layerDetails, learningRate):
        
        # self.layers is a list of tuples (inputSize, outputSize, activation function)
        self.layers = [Layer(detail[0], detail[1], detail[2]) for detail in layerDetails]
        self.lr = learningRate
        # Defined using the earlier MSE class
        self.lossFunc = MeanSquaredError()

    # Predicts a fit by simply applying the chosen activation function to all layers.
    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.activate(output)
        return output

    def training(self, x, y, epochs, batchSize=64, optimiser='ADAM', lr=None, verbose=False):
        """
        Trains over a pre-determined batch size, allowing a choice of optimiser between 
        ADAM (Adaptive Moment Estimation) or SGD (Stochastic Gradient Descent).

        It takes x as inputs (nSamples, nFeatures) and y as outputs (nSamples, nOutputs).

        After running, it outputs the development of the MSE over time, for later plotting.
        """
        # Initialises learning rate if it hasn't been already, and ensures that I/O are the correct shape.
        if lr is None:
            lr = self.lr
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n = x.shape[0]
        # Calls the MSE class
        MSE = MeanSquaredError()
        # This list is used to plot how the MSE changes over time:
        MSEHistory = []

        # Adam momenta variables (used later in the training)
        if optimiser == 'ADAM':
            """
            Mw and Mb are the momenta of the weights and biases, respectively.
            Vw and Vb are the exponentially weighted averages of squared gradients.

            The mathematical derivations of these are discussed in more depth in the report.
            
            """
            Mw = [np.zeros_like(layer.weights) for layer in self.layers]
            Vw = [np.zeros_like(layer.weights) for layer in self.layers]
            Mb = [np.zeros_like(layer.bias) for layer in self.layers]
            Vb = [np.zeros_like(layer.bias) for layer in self.layers]
            # These coefficents are used later, too.
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            """
            The main benefit of the Adam optimiser is that it updates over time. To do this, 
            we use a variable t (for time) that increases by 1 for each successive time that 
            the optimiser is called.
            """
            t = 0

        # Iterate over all the epochs
        for epoch in range(epochs):
            # Randomly shuffle indices (so there are no biases in the training data)
            idx = np.random.permutation(n)
            for start in range(0, n, batchSize):
                """
                The batch index is simply the index list going from 0 to the chosen size 
                of the batch (initialised as 64).
                """
                batchIdx = idx[start:start+batchSize]
                xb = x[batchIdx]
                yb = y[batchIdx]

                # Forward propagation on the batch itself.
                out = xb
                # Applies activation functions on all layers and calculates the MSE on this.
                for layer in self.layers:
                    out = layer.activate(out)
                _ = MSE.calculate(out, yb)  # sets y_pred / y_true inside mse

                # Backward propagation on the batch (using forward pass's values)
                dloss = MSE.backward()
                delta = dloss.copy() # Starting change in error (delta) for the output layer

                # Continues backwards propagation through the layers
                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    # Returns the inital variables for the zeroth layer, and the output of a
                    # previous layer, otherwise.
                    prevOutput = layer.input
                    """
                    Gradients for the weights and bias.
            
                    For the input neuron variables, the gradient is the result of the outputs (prediction from the 
                    inputs of the layer using the chosen activation functions) - the results that are
                    aimed for (y), multiplied by the derivative of the Input neuron in the layer.
            
                    For the weights, the gradient is the dot product of the transpose of the output of 
                    the previous layer, with the gradient of the input neuron.

                    For the bias, the gradient is the sum of all the input gradients in the layer.

                    This is derived in various tutorials online.
                    """
                    gradInput = delta * layer.activationDerivative(layer.initInput)
                    gradW = np.dot(prevOutput.T, gradInput)
                    gradB = np.sum(gradInput, axis=0, keepdims=True)
                    # Updates delta for the next (previous) layer
                    if i > 0:
                        """
                        Delta is updated as the dot product between the gradient of input variables 
                        and the transpose of the weights in the layer.
                        """
                        delta = np.dot(gradInput, layer.weights.T)

                    # Updates SGD optimiser (same code seen in the IBT):
                    if optimiser == 'SGD':
                        layer.weights -= lr * gradW
                        layer.bias -= lr * gradB

                    # Updates Adam optimiser (this is much more complicated):
                    else:
                        # Updates time whenever the Adam optimiser is called (as explained earlier)
                        t += 1
                        # Updates the weights' momenta (this formula is explained in the report)
                        Mw[i] = beta1*Mw[i] + (1 - beta1)*gradW
                        Vw[i] = beta2*Vw[i] + (1 - beta2)*(gradW**2)
                        MwCorrected = Mw[i] / (1 - beta1**t)
                        VWCorrected = Vw[i] / (1 - beta2**t)
                        layer.weights -= lr * MwCorrected / (np.sqrt(VWCorrected) + eps)

                        # Updates the bias' momenta (again this is explained in the report)
                        Mb[i] = beta1 * Mb[i] + (1 - beta1)*gradB
                        Vb[i] = beta2 * Vb[i] + (1 - beta2)*(gradB**2)
                        MbCorrected = Mb[i] / (1 - beta1**t)
                        VbCorrected = Vb[i] / (1 - beta2**t)
                        layer.bias -= lr * MbCorrected / (np.sqrt(VbCorrected) + eps)

            # Outputs prediction / MSE after each epoch. MSE is appended to a list.
            full_out = self.predict(x)
            full_loss = MSE.calculate(full_out, y)
            MSEHistory.append(full_loss)
            # Outputs losses for every 100th epoch, if specified as verbose.
            if verbose and (epoch % 10 == 0 or epoch == epochs-1):
                print(f"Epoch {epoch}, Loss: {full_loss:.6f}")

        return MSEHistory

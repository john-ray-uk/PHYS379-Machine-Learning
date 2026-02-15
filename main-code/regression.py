import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

"""
Linear & Polynomial Regression code

Uses OpenPyXL 3.1.5, NumPy 1.25.2, Pandas 2.3.3, and Matplotlib 3.10.8, with Python version 3.10.19.
If the code does not run correctly, try setting up a Python environment with these precise versions.

OpenPyXL is only required if reading Excel data, like the initial testing data from the IBT.

Imported into the "Main.py" file.

"""
# Main class code for the regression functions, to be called to the Neural Network
class Regression:
  # Standard (unoptimised) Linear Regression code, poorly made by John
  def linear(N,x,y,trainingRate,epochs):
    # Inputs must be taken as NumPy arrays
    X = np.asarray(x,dtype=float)
    Y = np.asarray(y,dtype=float)
    # Set weight, bias, derivatives, loss and final output to be initialised as 0
    w = 0
    b = 0
    dLdW = 0
    dLdB = 0
    l = 0
    output = 0
    # Iterate over the number of chosen epochs
    for i in range(epochs):
        # Can have np.sum(.../N), instead here.
        dLdW = np.mean(-(2*(X)*(Y-(w*X + b))))
        dLdB = np.mean(-(2*(Y-(w*X + b))))
        l = np.mean((Y-(w*X + b))**2)
        w = w - trainingRate*dLdW
        b = b - trainingRate*dLdB
        # Appends these to a list, for debugging purposes (can remove)
        wList.append(w)
        bList.append(b)
    output = w*N + b
    return output,l,w,b
  # Standard Polynomial Regression function
  def polynomialRegression(N,x,y,trainingRate,epochs,degree):
    # As before, inputs must be NumPy arrays
    X = np.asarray(x,dtype=float)
    Y = np.asarray(y,dtype=float)
    # Initialises dLdW, l and output to be 0, and the weights to be an array of zeros
    # Note how no biases are mentioned; these are incorporated as the 0th entry of the weights array.
    dLdW = 0
    l = 0
    output = 0
    w = np.zeros(degree + 1)
    # This creates an array of increasing values of X (e.g. x^0, x, x^2, etc.)
    Xmatrix = np.vander(X,degree+1, increasing=True)
    # Iterate over the number of chosen epochs    
    for i in range(epochs):
        # Takes the dot product between this polynomial matrix and the weights
        preds = np.dot(Xmatrix, w)
        error = Y - preds
        # Same code here as in Linear Regression, just extended to multiple dimensions for inputs.
        dLdW = -2*np.dot(Xmatrix.T, error)/len(Y)
        l = np.mean((error)**2)
        w = w - trainingRate*dLdW
    # Outputs the final dot product of all entries for the chosen number of data points (N) with the weights array.
    output = np.dot(np.array([N**i for i in range(degree + 1)]), w)
    # Returns this output with the final loss and weights.
    return output,l,w[0:]
  

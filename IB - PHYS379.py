# IB - PHYS379

'''
Individual benchmark script for PHYS379
- understand machine learning
- import data
- fit data to model
- send result to project convener
'''

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
#Import training data
import pandas as pd 

#Load data from file
file = pd.read_excel(r'C:\PERSONAL FOLDERS\Samir\Lancs\PHYS379\Advertising.xlsx')


'''
Creating matrices for each parameter, by extracting the relevant columns from the file
iloc - integer location, used to select specific rows and columns by their integer index
[:, n] - selects all rows (:) and the n+1 column
'''

# Extracting columns for each parameter
tv = file.iloc[:, 1].values
radio = file.iloc[:, 2].values
newspaper = file.iloc[:, 3].values
sales = file.iloc[:, 4].values
N = len(tv) # number of data points (tv could be replaced by any parameter - all of equal length)

# Initialising Parameters
alpha = 0.001 # training rate
E = 5000 # number of iterations through the training loop - epochs

# Parameters for cost function
w = 0.0
b = 0.0

# For ease of use in cost funciton
X = radio
Y = sales
X_i = 16.5 # example new advertising value
prediciton = []


'''
Functions - all of which together, when iterated, form a linear regression model
- predict: given an input x, and parameters w and b, predict the output y
- costfunc: calculate the mean squared error cost
- graddesc: update w and b based on the gradient of the cost function
'''

# Predict function: linear model: y = wx + b
def predict(X, w, b):
    return w * X + b

# Cost function: mean squared error
def costfunc(X, Y, w, b): 
    L = np.mean((Y - predict(X, w, b))**2)
    return L

# Gradient descent function: update w and b based on the cost function gradient
def graddesc(X, Y, w, b, alpha):
    dLdW = 0
    dLdB = 0
    dLdW = np.mean((2*(X)*(Y-predict(X, w, b))))
    dLdW = -dLdW
    dLdB = np.mean((2*(Y-predict(X, w, b))))
    dLdB = -dLdB
    w = w - alpha*dLdW
    b = b - alpha*dLdB
    return w, b

# Function to plot the data and the fitted regression line
def plot_regression(X, Y, w, b, X_i=None):
    # Create smooth line for fitted model
    X_line = np.linspace(np.min(X), np.max(X), 200)
    Y_line = predict(X_line, w, b)
    plt.figure()
    # Scatter plot of data
    plt.scatter(X, Y, color='blue', label='Data Points')
    # Regression line
    plt.plot(X_line, Y_line, color='red', label='Fitted Line')
    plt.xlabel("(X)")
    plt.ylabel("(Y)")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()


'''
Running the training function to train the model and make a prediction for the new input value
'''

for h in range(E):
    L = costfunc(X, Y, w, b)
    w, b = graddesc(X, Y, w, b, alpha)
    y_i = predict(X_i, w, b)
    row = [h,w,b,L,y_i]
    prediciton.append(row)
prediciton_df = pd.DataFrame(prediciton,columns=["E","w","b","L","y_i"])
print(prediciton_df)

plot_regression(X, Y, w, b, X_i)
# Imports Python libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy import random
from datetime import date, timedelta
# Imports Neural Network code
from NeuralNetwork import NeuralNetwork
"""
Main code:

Imports Yfinance, NumPy, MatPlotlib, Pandas, Datetime, and the Neural Network code.
Polynomial & Logarithmic regression are not incorporated yet.

"""
# Makes matplotlib graphs look nice :)
mpl.rcParams['font.family'] = 'Times New Roman'
# Gives a timerange of 1 year.
AbsStart = date.today() - timedelta(365)
AbsStart.strftime('%Y-%m-%d')
AbsEnd = date.today() + timedelta(2)
AbsEnd.strftime('%Y-%m-%d')

def stockdata(ticker, start, end):
    Asset = pd.DataFrame(yf.download(ticker, start=start, end=end)['Close'])
    # Gives stock prices when the market closed, for a custom stock over a custom timeframe.
    # Could also choose prices when it opens, or the highest/lowest prices over the day.     
    return Asset # Returns the overall Pandas dataframe.

# Uses Tesla data, as an example
TESLA = stockdata('TSLA', AbsStart, AbsEnd)

# Turns dates and prices into numbers (0 to 1)
prices = TESLA['TSLA'].values
dates = mdates.date2num(TESLA.index)

# Normalization (fixes for Sigmoid function)
scaledPrice = (prices - prices.min()) / (prices.max() - prices.min())
scaledDates = (dates - dates.min()) / (dates.max() - dates.min())

# Reshapes the Neural Network
# Column 1: Normalized Date, Column 2: A dummy variable
inputs = np.column_stack((scaledDates, np.zeros_like(scaledDates)))
targets = scaledPrice
trainingRate = 0.1
iterations = 100000
# Train the Network
NN = NeuralNetwork(trainingRate)
NN.train(inputs, targets, iterations)
print(NN.bias)
print(NN.weights)
# Generates the predictions
predictionsScaled = np.array([NN.predict(i) for i in inputs])

# Unscales the data for plotting
predictionsFinal = predictionsScaled * (prices.max() - prices.min()) + prices.min()

# Plots the graph
plt.figure(figsize=(12, 6))
plt.plot(TESLA.index, prices, label="Actual Tesla Price", color='red', alpha=0.5)
plt.plot(TESLA.index, predictionsFinal, label="NN Prediction Line", color='black', linewidth=3)
plt.title('Tesla Price vs Neural Network Fit')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.gcf().autofmt_xdate() # Fixes up date display
plt.show()

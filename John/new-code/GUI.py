import os
import tkinter as tk
from tkinter import ttk
from NeuralNetwork import *
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy import random
from datetime import datetime, timedelta, date
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)

"""
App class that creates a Tkinter GUI for the code.

Current aims:

- Allow the user to easily change variables for the neural network; like the
learning rate, number of layers, number of neurons per layer, activation functions, etc.
- Create graphs of AI-optimised best fits.
- Allows said graphs to be saved.
- Allow "models" to be saved (as custom .json files).

Additional aims:

- Extend to work with convolutional neural networks.
- Test with other data sets.
- Save application as a .exe.

"""

# To determine the base directory of the .ico file:
basedir = os.path.dirname(__file__)

class app(tk.Tk):
    def __init__(self):
        super().__init__() # Keeps initial conditions set here in other methods like the UI.
        self.title("PHYS379: Neural Network")
        self.geometry("1920x768")
        
        # Defines variables
        self.layersUI = []
        self.learningRate = tk.DoubleVar(value=0.0005)
        self.epoch = tk.IntVar(value=1000)
        self.batchSize = tk.IntVar(value=64)
        self.trainingPercentage = tk.DoubleVar(value=0.8)
        self.percentage = tk.StringVar(value=f"{self.trainingPercentage.get()*100}%")
        self.optimiser = tk.StringVar(value="ADAM")
        self.pathname = tk.StringVar(value="neuralNetwork")
        self.ticker = tk.StringVar(value="AAPL")
        self.chosenPath = tk.StringVar(value='./neuralNetwork.json')
        self.startTime = tk.StringVar(value="2004-12-26")
        self.endTime = tk.StringVar(value="2014-12-26")
        self.forwardDays = tk.IntVar(value=5)
        self.MSE = tk.StringVar(value="")
        self.RMSE = tk.StringVar(value="")
        self.average = tk.StringVar(value="")
        self.percentError = tk.StringVar(value="")
        # Creates frames for the GUI (left-centres the buttons and such)
        blankframe = ttk.Frame(self, padding=0)
        blankframe.pack(side=tk.LEFT, fill=tk.Y,padx=10)
        self.frame = ttk.Frame(self, padding = 1)
        self.frame.pack(side=tk.LEFT, fill=tk.Y,expand=True, padx=5)
        self.UI() # Loads the UI function.

    def UI(self):
        # Container for packed objects
        self.packFrame = tk.Frame(self.frame)
        self.packFrame.pack(side =tk.LEFT, fill=tk.Y, pady=10)
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        # Title in the window
        Title = tk.Label(self.packFrame,text="PHYS379 Neural Network")
        Title.config(font=("Helvetica", 12, "bold", "underline"))
        Title.pack()
        # Subtitle in the window
        SubTitle = tk.Label(self.packFrame,text="by John, Luke, Samir, Alvi, & Adam")
        SubTitle.config(font=("Helvetica", 10))
        SubTitle.pack()       
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        ttk.Label(self.packFrame, text="Learning Rate for model:").pack()
        ttk.Entry(self.packFrame, textvariable=self.learningRate).pack()
        ttk.Label(self.packFrame, text="Number of iterations for model:").pack()
        ttk.Entry(self.packFrame, textvariable=self.epoch).pack()
        ttk.Label(self.packFrame, text="Batch size:").pack()
        ttk.Entry(self.packFrame, textvariable=self.batchSize).pack()
        ttk.Label(self.packFrame, text="Forward days (target shift):").pack()
        ttk.Entry(self.packFrame, textvariable=self.forwardDays).pack()
        ttk.Label(self.packFrame, text="Starting time for Stocks data:").pack()
        tk.Entry(self.packFrame, textvariable=self.startTime).pack()
        ttk.Label(self.packFrame, text="Ending time for Stocks data:").pack()
        tk.Entry(self.packFrame, textvariable=self.endTime).pack()
        ttk.Label(self.packFrame, text="Please insert a Stock Ticker:").pack()
        tk.Entry(self.packFrame, textvariable=self.ticker).pack()
        ttk.Label(self.packFrame, text="Percentage of data used to train the NN:").pack()
        # Creates a scale for training percentage, from 0.05 to 0.95
        def trainingPercentageRound(val):
            val = round(float(val), 2)
            self.trainingPercentage.set(val)
            self.percentage.set(f"{val*100}%")
        ttk.Scale(self.packFrame, variable=self.trainingPercentage, from_ = 0.05, to = 0.95,command=trainingPercentageRound).pack()
        ttk.Label(self.packFrame, textvariable=self.percentage).pack()
        ttk.Label(self.packFrame, text="Choice of optimiser function:").pack()
        ttk.OptionMenu(self.packFrame, self.optimiser, "SGD","SGD", "ADAM" ).pack()
        # Container for dynamic layers
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        self.gridFrame = tk.Frame(self.packFrame)
        self.gridFrame.pack(fill=tk.Y, pady=10)
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        ttk.Button(self.packFrame, text="Add Layer", command=lambda:(self.addLayer(1))).pack()
        ttk.Button(self.packFrame, text="Build & Initialise NN", command=lambda:(self.buildNN(),self.plot())).pack()
        # Initial Layer (Input layer sizing)
        tk.Label(self.gridFrame, text="Neurons").grid(row=0, column=0)
        tk.Label(self.gridFrame, text="Activation").grid(row=0, column=1)
        # default initial layers to something sensible for OHLC -> 1-day forward
        self.addLayer(4)
        self.addLayer(8)
        self.addLayer(4)
        self.addLayer(1)
        # Creates a second right-centred frame for the graphs and output values to be added to.
        RightFrame = ttk.Frame(self, padding=8).pack(side=tk.RIGHT)
        # Then, a plotting subframe is created for the first graph (alongside a second one for the second graph).
        plotframe = ttk.Frame(RightFrame, padding=8)
        plotframe.pack(side=tk.RIGHT, padx=25,pady=50)
        # Creates a figure
        self.fig = plt.Figure()
        # Creates axis to append images
        self.ax = self.fig.add_subplot(111)
        # Enables the figures to be attached to a Tkinter "canvas" that enables Tkinter variables to be used.
        self.canvas = FigureCanvasTkAgg(self.fig, master=plotframe)
        # Fills in the canvas' information
        self.canvas.draw()
        # Unpacks (creates) the widget associated with Tkinter
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas._tkcanvas.pack()
        # Second plotting subframe (inside the RightFrame).
        plotframe2 = ttk.Frame(RightFrame, padding=8)
        plotframe2.pack(side=tk.RIGHT, padx=25,pady=50)
        # Creates a figure
        self.fig2 = plt.Figure()
        # Creates axis to append images
        self.ax2 = self.fig2.add_subplot(111)
        # Enables the figures to be attached to a Tkinter "canvas" that enables Tkinter variables to be used.
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plotframe2)
        # Fills in the canvas' information
        self.canvas2.draw()
        # Unpacks (creates) the widget associated with Tkinter
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas2._tkcanvas.pack()

        self.packFrame2 = ttk.Frame(plotframe)
        self.packFrame2.pack(side=tk.BOTTOM, fill = tk.X, pady=20)

        # This "OutputFrame" holds all the output variables in a grid. The padding was a bit of guesswork (hence the seemingly random values).
        OutputFrame = ttk.Frame(self.packFrame2)
        OutputFrame.pack()
        MSEtext = ttk.Label(OutputFrame, text="Test MSE ($):")
        MSEtext.config(font=("Helvetica", 10,"bold", "underline"))
        MSEtext.grid(row=0, column=0, sticky='w',pady=10, padx=10)
        ttk.Label(OutputFrame, textvariable=self.MSE).grid(row=1, column=0, sticky='w', padx=(30,2))
        RMSEtext = ttk.Label(OutputFrame, text="Test RMSE ($):")
        RMSEtext.config(font=("Helvetica", 10,"bold", "underline"))
        RMSEtext.grid(row=0, column=1, sticky='w',pady=10, padx=10)
        ttk.Label(OutputFrame, textvariable=self.RMSE).grid(row=1, column=1, sticky='w', padx=(38,2))
        AveragePricetext = ttk.Label(OutputFrame, text="Average Price ($):")
        AveragePricetext.config(font=("Helvetica", 10,"bold", "underline"))
        AveragePricetext.grid(row=0, column=2, sticky='w',pady=10, padx=10)        
        ttk.Label(OutputFrame, textvariable=self.average).grid(row=1, column=2, sticky='w', padx=(40,2))
        PercentageErrortext = ttk.Label(OutputFrame, text="Percentage Error:")
        PercentageErrortext.config(font=("Helvetica", 10,"bold", "underline"))
        PercentageErrortext.grid(row=0, column=3, sticky='w',pady=10, padx=10) 
        ttk.Label(OutputFrame, textvariable=self.percentError).grid(row=1, column=3, sticky='w', padx=(45,2))
        

    def addLayer(self, n):
        rowIdx = len(self.layersUI) + 1
        
        neuronEntry = tk.Entry(self.gridFrame, width=10)
        neuronEntry.insert(0, f"{n}")
        neuronEntry.grid(row=rowIdx, column=0, padx=2, pady=2)
        
        ComboBox = ttk.Combobox(self.gridFrame, values=["Sigmoid", "ReLU","Linear","Tanh"], width=10)
        # Defaults last layer to Linear, and any hidden layers to ReLU.
        if len(self.layersUI) == 0:
            ComboBox.set("Sigmoid")
        else:
            ComboBox.set("ReLU")
        ComboBox.grid(row=rowIdx, column=1, padx=2, pady=2)
        
        self.layersUI.append((neuronEntry, ComboBox))

    def buildNN(self):
        # Builds the network by importing from the NeuralNetwork.py file.
        try:
            lr = float(self.learningRate.get())
            architecture = []
            for i in range(len(self.layersUI) - 1):
                in_size = int(self.layersUI[i][0].get())
                out_size = int(self.layersUI[i+1][0].get())
                act = self.layersUI[i+1][1].get()
                architecture.append((in_size, out_size, act))
            # Forces the final layer to Linear (or else there are multiple fits)
            if len(architecture) > 0:
                inputData, outputData, string = architecture[-1]
                architecture[-1] = (inputData, outputData, "Linear")
            self.nn = NeuralNetwork(architecture, lr)
            print(f"Neural network built with {len(architecture)} layers. Architecture: {architecture}")
        except Exception as e:
            print(f"Uh oh: {e}")

    # Main training function
    def training(self, iterations, ticker):
        iterations = int(self.epoch.get()) # Prevents float inputs
        ticker = self.ticker.get()
        forwardDays = int(self.forwardDays.get())
        """
        Parse input dates using Python's datetime library.

        Dates are stripped into the year, month and day input and reassembled as datetime objects.

        """
        startTime = self.startTime.get()
        startTime = datetime.strptime(startTime,'%Y-%m-%d')
        endTime = self.endTime.get()
        endTime = datetime.strptime(endTime,'%Y-%m-%d')
        # Print command for debugging.
        print(f"Start time = {startTime}, end time = {endTime}, ticker = {ticker}, forward days = {forwardDays}")

        # Downloads Yfinance data.
        DATA = yf.download(ticker, start=startTime, end=endTime)
        # Prevents program from running if download is unsuccessful.
        if DATA is None or DATA.empty:
            raise RuntimeError("Downloaded data is empty. Check ticker and dates.")
        # Takes ALL input data (previously it only took the closing data)
        DATA = DATA[['Open','High','Low','Close']].copy()
        DATA['Target_Close'] = DATA['Close'].shift(-forwardDays)
        DATA = DATA.dropna()   # Removes any empty "NaN" rows (shouldn't be a problem anyway)

        inputs = DATA[['Open', 'High', 'Low', 'Close']].values   # Shape: (N,4)
        targets = DATA[['Target_Close']].values                  # Shape: (N,1)

        # train/test split chronological
        splitRatio = self.trainingPercentage.get()
        splitIndex = int(len(inputs) * splitRatio)
        xTrain = inputs[:splitIndex]
        yTrain = targets[:splitIndex]
        xTest  = inputs[splitIndex:]
        yTest  = targets[splitIndex:]
        timeIndex = DATA.index[splitIndex:]

        # Normalise the inputs (x) and the outputs (y) using standard deviation and means
        xMean = np.mean(xTrain, axis=0); xStd = np.std(xTrain, axis=0); xStd[xStd==0]=1e-8
        yMean = np.mean(yTrain, axis=0); yStd = np.std(yTrain, axis=0); yStd[yStd==0]=1e-8

        xTrainNorm = (xTrain - xMean) / xStd
        xAllNorm = (inputs - xMean) / xStd
        yTrainNorm = (yTrain - yMean) / yStd

        # Gets the architecture from the GUI and rebuilds the NN with a correct input size.
        neuronCounts = [int(ne.get()) for (ne, _) in self.layersUI]
        activations   = [ac.get() for (_, ac) in self.layersUI]

        architecture = []
        inFeatures = xTrainNorm.shape[1]
        for i in range(len(neuronCounts) - 1):
            inSize  = inFeatures if i == 0 else neuronCounts[i]
            outSize = neuronCounts[i + 1]
            act = activations[i + 1]
            architecture.append((inSize, outSize, act))
        if len(architecture) > 0:
            architecture[-1] = (architecture[-1][0], architecture[-1][1], "Linear")

        self.nn = NeuralNetwork(architecture, float(self.learningRate.get()))
        print("Rebuilt NN with architecture:", architecture)

        # Trains over a small batch (with Adam as the default optimiser)
        batchSize = int(self.batchSize.get())
        MSEHistory = self.nn.training(xTrainNorm, yTrainNorm, epochs=iterations, batchSize=batchSize, optimiser=self.optimiser.get(), lr=float(self.learningRate.get()), verbose=False)

        # store normalization params for plotting
        self._norm = {'Normalised X': xAllNorm, 'Mean Y': yMean, 'Standard Y': yStd, 'Prices': DATA['Close'].values, 'DATA': DATA, 'Time index': timeIndex, 'Data': DATA}

        return xTrain, inputs, DATA, MSEHistory

    def plot(self):
        # Resets axes / clears graphs.
        self.ax.clear()
        self.ax2.clear()
        # Tries training the data, and outputs an error if not possible.
        try:
            inputs, prices, DATA, MSEHistory = self.training(self.epoch, self.ticker)
        except Exception as e:
            print("Training error:", e)
            return
        xAllNorm = self._norm['Normalised X']
        yMean = self._norm['Mean Y']
        yStd = self._norm['Standard Y']
        DATA = self._norm['Data']

        predictionsScaled = self.nn.predict(xAllNorm)  # Normalised predictions.
        predictionsFinal = predictionsScaled * yStd + yMean  # Unscale the data back to plottable values.

        yPreds = predictionsFinal.reshape(-1)

        # Outputs the change in MSE, if possible (which should always be the case)
        if len(MSEHistory) > 0:
            print(f"Initial MSE: {MSEHistory[0]} and Final MSE: {MSEHistory[-1]}")

        # Determine split index (inputs is first return and the training set)
        splitIndex = len(inputs)

        # True unscaled test targets
        yTest = DATA['Target_Close'].values[splitIndex:] # shape (N_test,)

        # Predicted unscaled test targets
        predTest = yPreds[splitIndex:] # shape (N_test,)

        # Ensures that both are 1D numpy arrays (or else there'll be annoying shape errors)
        predTest = np.array(predTest).reshape(-1)
        yTest = np.array(yTest).reshape(-1)

        # metrics (unscaled / currency units)
        testMSE = np.mean((predTest - yTest) ** 2)
        testRMSE = np.sqrt(testMSE)
        averagePrice = np.mean(yTest)
        percentError = (testRMSE / averagePrice) * 100

        # Update ttk labels by resetting the stringVars.
        self.MSE.set(f"{testMSE:.4f}")
        self.RMSE.set(f"{testRMSE:.4f}")
        self.average.set(f"{averagePrice:.4f}")
        self.percentError.set(f"{percentError:.2f}%")

        # Plots the data
        self.ax.plot(DATA.index, DATA['Close'].values, label=f"Actual {self.ticker.get()} Price", color='red')
        self.ax.plot(DATA.index, predictionsFinal, label="Neural Network fit", color='black', alpha=0.5)
        self.ax.set_title(f'{self.ticker.get()} Price vs Neural Network Fit')
        self.ax.set_xlabel('Date')
        #self.ax.set_xticklabels(r)
        self.ax.set_ylabel('Price ($)')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

        loss = [np.sqrt(l) for l in MSEHistory]

        self.ax2.plot(loss, color='black', alpha=0.5)
        self.ax2.set_title('RMSE loss against epochs')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('RMSE value')
        self.ax2.grid(True)
        self.canvas2.draw()


# Runs the application
if __name__ == "__main__":
    np.random.seed(0)  # Makes the "random" numbers be the same each time the program is run, for repeatability.
    a = app()
    a.iconbitmap(os.path.join(basedir, "icon.ico"))
    a.mainloop()

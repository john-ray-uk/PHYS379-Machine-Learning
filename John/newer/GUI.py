import os
import tkinter as tk
from tkinter import ttk
from NeuralNetwork import *
import yfinance as yf
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy import random
from datetime import datetime, timedelta, date
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from lightkurve import search_lightcurve

"""
App class that creates a Tkinter GUI for the code.

Current aims:

- Allow "models" to be saved (as custom .json files).

Additional aims:

- Extend to work with convolutional neural networks.
- Test with other data sets.

"""

# To determine the base directory of the .ico file:
basedir = os.path.dirname(__file__)
# Makes matplotlib graphs look nice :)
mpl.rcParams['font.family'] = 'Times New Roman'
print("============================================")
print("PHYS379 Project: Neural Network Application")
print("    by John, Luke, Samir, Alvi, & Adam")
print("============================================")
print("")
print("Initialising program...")
print("")
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
        self.mission = tk.StringVar(value="Kepler")
        self.object = tk.StringVar(value="Kepler-10")
        self.dataset = "YFinance"
        self.yearTemp = tk.IntVar(value="2004")
        self.predictTemp = tk.StringVar(value=f"Reinitialise network to show")
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
        RightFrame = ttk.Frame(self, padding=8)
        RightFrame.pack(side=tk.RIGHT)
        # Then, a plotting subframe is created for the first graph (alongside a second one for the second graph).
        plotframe = ttk.Frame(RightFrame, padding=8)
        plotframe.pack(side=tk.RIGHT, padx=25,pady=50)

        self.selectionFrame = ttk.Frame(plotframe)
        self.selectionFrame.pack(side=tk.TOP, fill=tk.X, pady=20)

        InputFrame = ttk.Frame(self.selectionFrame)
        InputFrame.pack()


        SelectionTitle = ttk.Label(InputFrame, text="Select dataset:")
        SelectionTitle.config(font=("Helvetica", 12,"bold", "underline"))
        SelectionTitle.grid(row=0, column=0, sticky='w',pady=10, padx=10)
        ttk.Button(InputFrame, text="Finance", command=lambda:(self.switchDataset("YFinance"))).grid(row=0, column=1, sticky='w',pady=10, padx=10)
        ttk.Button(InputFrame, text="Temperature rise", command=lambda:(self.switchDataset("CSV"))).grid(row=0, column=2, sticky='w',pady=10, padx=10)
        ttk.Button(InputFrame, text="Stellar Light Curves", command=lambda:(self.switchDataset("LightKurve"))).grid(row=0, column=3, sticky='w',pady=10, padx=10)
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
        # Second and third plotting subframes (inside the RightFrame).
        plotframe2 = ttk.Frame(RightFrame, padding=8)
        plotframe2.pack(side=tk.BOTTOM, padx=10,pady=10)
        plotframe3 = ttk.Frame(RightFrame, padding=8)
        plotframe3.pack(side=tk.TOP, padx=10,pady=10)
        # Creates figures
        self.fig2 = plt.Figure(figsize=(5, 3.6))
        self.fig3 = plt.Figure(figsize=(5, 3.6))
        # Creates axes to append images
        self.ax2 = self.fig2.add_subplot(111)
        self.ax3 = self.fig3.add_subplot(111)
        # Enables the figures to be attached to a Tkinter "canvas" that enables Tkinter variables to be used.
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plotframe2)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=plotframe3)
        # Fills in the canvas' information
        self.canvas2.draw()
        self.canvas3.draw()
        # Unpacks (creates) the widget associated with Tkinter
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas2._tkcanvas.pack()
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas3._tkcanvas.pack()

        self.packFrame2 = ttk.Frame(plotframe)
        self.packFrame2.pack(side=tk.BOTTOM, fill = tk.X, pady=20)

        # This "OutputFrame" holds all the output variables in a grid. The padding was a bit of guesswork (hence the seemingly random values).
        OutputFrame = ttk.Frame(self.packFrame2)
        OutputFrame.pack()
        MSEtext = ttk.Label(OutputFrame, text="Test MSE:")
        MSEtext.config(font=("Helvetica", 10,"bold", "underline"))
        MSEtext.grid(row=0, column=0, sticky='w',pady=10, padx=10)
        ttk.Label(OutputFrame, textvariable=self.MSE).grid(row=1, column=0, sticky='w', padx=(30,2))
        RMSEtext = ttk.Label(OutputFrame, text="Test RMSE:")
        RMSEtext.config(font=("Helvetica", 10,"bold", "underline"))
        RMSEtext.grid(row=0, column=1, sticky='w',pady=10, padx=10)
        ttk.Label(OutputFrame, textvariable=self.RMSE).grid(row=1, column=1, sticky='w', padx=(38,2))
        AveragePricetext = ttk.Label(OutputFrame, text="Average Value:")
        AveragePricetext.config(font=("Helvetica", 10,"bold", "underline"))
        AveragePricetext.grid(row=0, column=2, sticky='w',pady=10, padx=10)        
        ttk.Label(OutputFrame, textvariable=self.average).grid(row=1, column=2, sticky='w', padx=(40,2))
        PercentageErrortext = ttk.Label(OutputFrame, text="Percentage Error:")
        PercentageErrortext.config(font=("Helvetica", 10,"bold", "underline"))
        PercentageErrortext.grid(row=0, column=3, sticky='w',pady=10, padx=10) 
        ttk.Label(OutputFrame, textvariable=self.percentError).grid(row=1, column=3, sticky='w', padx=(45,2))

    # Function to clear the GUI upon reloading
    def clearGUI(self,frame):
        # Destroys any subframes...
        for child in frame.winfo_children():
            child.destroy()
        # And clears any layers, unless they're already empty.
        try:
            self.layersUI.clear()
        except Exception:
            self.layersUI = []
            return
        return
    # Function for switching between datasets (pretty intuitive)
    def switchDataset(self,dataset):
        self.dataset = dataset
        self.clearGUI(self.packFrame)
        if dataset == 'YFinance':
            self.YFinance()
        elif dataset == 'LightKurve':
            self.LightKurve()
            return
        elif dataset == 'CSV':
            self.CSV()
            return
        return
    """
    The GUI is recreated when switched between datasets. The easiest way I could think of doing this was by destroying the older
    GUI, and recreating it for each - therefore requiring 3 functions to redefine each dataset's GUI.

    These functions are called YFinance, LightKurve, and CSV - and are shown below.
    
    """
    def YFinance(self):
        """ Copy of earlier code:"""
        if not hasattr(self, "packFrame") or self.packFrame is None:
            self.packFrame = tk.Frame(self.frame)
            self.packFrame.pack(side=tk.LEFT, fill=tk.Y, pady=10)
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        # Different epochs / forward shifts work best for each.
        self.epoch.set(1000)
        self.forwardDays.set(5)
        # Title in the window
        Title = tk.Label(self.packFrame,text="PHYS379 Neural Network Creator")
        Title.config(font=("Helvetica", 12, "bold", "underline"))
        Title.pack()
        # Subtitle in the window
        SubTitle = tk.Label(self.packFrame,text="by John, Luke, Samir, Alvi, & Adam")
        SubTitle.config(font=("Helvetica", 10))
        SubTitle.pack()
        SubSubTitle = tk.Label(self.packFrame,text="YFinance dataset selected:")
        SubSubTitle.config(font=("Helvetica", 10, "underline"))
        SubSubTitle.pack()        
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
        # These layers work well for the YFinance dataset.
        self.addLayer(4)
        self.addLayer(8)
        self.addLayer(4)
        self.addLayer(1)
        return

    def LightKurve(self):
        if not hasattr(self, "packFrame") or self.packFrame is None:
            self.packFrame = tk.Frame(self.frame)
            self.packFrame.pack(side=tk.LEFT, fill=tk.Y, pady=10)
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        self.epoch.set(50)
        self.learningRate.set(0.00005)
        self.forwardDays.set(200)
        # Title in the window
        Title = tk.Label(self.packFrame,text="PHYS379 Neural Network")
        Title.config(font=("Helvetica", 12, "bold", "underline"))
        Title.pack()
        # Subtitle in the window
        SubTitle = tk.Label(self.packFrame,text="by John, Luke, Samir, Alvi, & Adam")
        SubTitle.config(font=("Helvetica", 10))
        SubTitle.pack()
        SubSubTitle = tk.Label(self.packFrame,text="LightKurve dataset selected:")
        SubSubTitle.config(font=("Helvetica", 10, "underline"))
        SubSubTitle.pack()          
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        ttk.Label(self.packFrame, text="Learning Rate for model:").pack()
        ttk.Entry(self.packFrame, textvariable=self.learningRate).pack()
        ttk.Label(self.packFrame, text="Number of iterations for model:").pack()
        ttk.Entry(self.packFrame, textvariable=self.epoch).pack()
        ttk.Label(self.packFrame, text="Batch size:").pack()
        ttk.Entry(self.packFrame, textvariable=self.batchSize).pack()
        ttk.Label(self.packFrame, text="Forward shift (target shift):").pack()
        ttk.Entry(self.packFrame, textvariable=self.forwardDays).pack()
        ttk.Label(self.packFrame, text="Choice of mission:").pack()
        tk.Entry(self.packFrame, textvariable=self.mission).pack()
        ttk.Label(self.packFrame, text="Choice of object:").pack()
        tk.Entry(self.packFrame, textvariable=self.object).pack()
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
        # There are 6 input layers for the LightKurve data.
        self.addLayer(6)
        self.addLayer(8)
        self.addLayer(4)
        self.addLayer(1)
        return
    def CSV(self):
        if not hasattr(self, "packFrame") or self.packFrame is None:
            self.packFrame = tk.Frame(self.frame)
            self.packFrame.pack(side=tk.LEFT, fill=tk.Y, pady=10)
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        self.epoch.set(1000)
        # Title in the window
        Title = tk.Label(self.packFrame,text="PHYS379 Neural Network")
        Title.config(font=("Helvetica", 12, "bold", "underline"))
        Title.pack()
        # Subtitle in the window
        SubTitle = tk.Label(self.packFrame,text="by John, Luke, Samir, Alvi, & Adam")
        SubTitle.config(font=("Helvetica", 10))
        SubTitle.pack()
        SubSubTitle = tk.Label(self.packFrame,text="Met Office datasets selected:")
        SubSubTitle.config(font=("Helvetica", 10, "underline"))
        SubSubTitle.pack()          
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        ttk.Label(self.packFrame, text="Learning Rate for model:").pack()
        ttk.Entry(self.packFrame, textvariable=self.learningRate).pack()
        ttk.Label(self.packFrame, text="Number of iterations for model:").pack()
        ttk.Entry(self.packFrame, textvariable=self.epoch).pack()
        ttk.Label(self.packFrame, text="Batch size:").pack()
        ttk.Entry(self.packFrame, textvariable=self.batchSize).pack()
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
        # A more complex network is needed to compensate for the poorer amount of Met Office data.
        self.addLayer(4)
        self.addLayer(64)
        self.addLayer(32)
        self.addLayer(1)
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        # Cannot change the forward shift for this data, so I added this to compensate.
        ttk.Label(self.packFrame, text="Year to predict temperature rise for:").pack()
        ttk.Entry(self.packFrame, textvariable=self.yearTemp, width=8).pack()
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        ttk.Label(self.packFrame, textvariable=self.predictTemp).pack()
        ttk.Separator(self.packFrame).pack(fill=tk.X, pady=10)
        return
    # Function to add layers to the network, and fill these in the GUI.    
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

    # Function to build the Neural Network itself.
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
                #architecture[-1] = (inputData, outputData, "Linear")
            np.random.seed(0)
            self.nn = NeuralNetwork(architecture, lr)
            print(f"Neural network built with {len(architecture)} layers. Architecture: {architecture}")
            print("")
        except Exception as e:
            print(f"Uh oh: {e}")
            print(f"layers = {self.layersUI} & lr = {lr}")
            print("")

    # Main training function
    def training(self, iterations, ticker):
        iterations = int(self.epoch.get()) # Prevents float inputs
        forwardDays = int(self.forwardDays.get())
        if self.dataset == "YFinance":
            ticker = self.ticker.get()
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
            print("")

            # Downloads Yfinance data.
            print("Downloading YFinance data. Progress:")
            DATA = yf.download(ticker, start=startTime, end=endTime)
            print("")
            # Prevents program from running if download is unsuccessful.
            if DATA is None or DATA.empty:
                raise RuntimeError("Downloaded data is empty. Check ticker and dates.")
            # Takes ALL input data (previously it only took the closing data)
            DATA = DATA[['Open','High','Low','Close']].copy()
            DATA['Return'] = DATA['Close'].shift(-forwardDays)
            DATA = DATA.dropna()   # Removes any empty "NaN" rows (shouldn't be a problem anyway)

            inputs = DATA[['Open', 'High', 'Low', 'Close']].values   # Shape: (N,4)
            targets = DATA[['Return']].values                  # Shape: (N,1)

        elif self.dataset == "LightKurve":
            # Downloads light curve data for the chosen star (this can often take 10-20s or so)
            lcCollection = search_lightcurve(self.object.get(), mission=self.mission.get()).download_all()

            # Initialises the flux and x/y Centroid lists.
            fluxList, cxList, cyList = [], [], []

            # Needed more input variables to improve the neural network's fits. After some research, I found this.
            for lc in lcCollection:
                # Use "flux" as a fallback if "pdcsap_flux" isn't available. "Pdcsap_flux" is pre-calibrated, and better quality. 
                fCol = 'pdcsap_flux' if 'pdcsap_flux' in lc.columns else 'flux'
                
                fluxList.append(lc[fCol].value)
                
                # Safely gets the centroids; sets the data to 0 if not available.
                if 'mom_centr1' in lc.columns:
                    cxList.append(lc['mom_centr1'].value)
                    cyList.append(lc['mom_centr2'].value)
                else:
                    cxList.append(np.zeros(len(lc)))
                    cyList.append(np.zeros(len(lc)))

            # Stacks this data using np.hstack to work with the network in the correct input shape.
            flux = np.hstack(fluxList)
            centroidX = np.hstack(cxList)
            centroidY = np.hstack(cyList)

            # Creates a dataframe with daily (cadence) flux values and the centroid data.
            DATA = pd.DataFrame({"Flux": flux,"X centroid": centroidX,"Y centroid": centroidY})

            # Adds in a 5-cadence-long mean dataset.
            DATA['Mean flux'] = DATA['Flux'].rolling(window=5).mean()

            # Adds in a 5-cadence-long standard deviation dataset.
            DATA['Std Flux'] = DATA['Flux'].rolling(window=5).std()

            # Adds in a rate of flux change dataset.
            DATA['Rate of flux change'] = DATA['Flux'].diff()

            """ All of the above datasets improve the network's accuracy, by providing more relevant parameters."""
            # Errors arise if the LightKurve data is shifted more than 1,000,000 places. I don't know why.
            if self.forwardDays.get() > 1000000:
                self.forwardDays.set(1000000)

            """
            The target data is set to be the input flux data shifted to a relevant degree.
            Due to this, setting the shift to be 0 will lead to an error.
            """
            forwardSteps = int(self.forwardDays.get())
            DATA['Return'] = DATA['Flux'].shift(-forwardSteps)

            # Removes any "NAN" errors in the data, which seems to happen an awful lot.
            DATA = DATA.dropna()

            # Inputs general flux values, as previously stated.
            inputs = DATA[['Flux', 'X centroid', 'Y centroid', 'Mean flux', 'Std Flux', 'Rate of flux change']].values

            # Targets are the shifted flux data.
            targets = DATA[['Return']].values

            # Ensures that these are NumPy arrays.
            inputs = np.asarray(inputs)
            targets = np.asarray(targets)

            # Ensures that they are the correct shape size.
            if inputs.ndim == 1:
                inputs = inputs.reshape(-1, 1)
            if targets.ndim == 1:
                targets = targets.reshape(-1, 1)

            # From looking at LightKurve's documentation, I figured out how to determine the number of shifts in days, which is cool :)
            sampleLc = lcCollection[0]
            timeDiff = sampleLc.time[1].value - sampleLc.time[0].value
            # The time values are usually preset in days, hence the intuitive code.
            totalShiftedTime = forwardSteps * timeDiff
            print(f"Shift of {forwardSteps} cadences is roughly {totalShiftedTime:.2f} days.")

        else:
            """
            Met Office data was imported from here as several .csvs: https://climate.metoffice.cloud/temperature.html#datasets

            I compiled the HadCRUT5, NOAAGlobalTemp, and Berkeley Earth data into one .csv, and took the first dataset as the target, 
            and the other two as inputs. The other datasets had a shorter time-range and couldn't be accurately incorporated.
            
            """
            # Use pandas to read the renamed .csv.
            DATA = pd.read_csv("TempData.csv")

            # Take the years data and reshape into the correct format.
            years = DATA['Year'].astype(float).values.reshape(-1,1)   # Shape of (N,1)
            # Same deal for the NOAAGlobalTemp and Berkeley Earth data.
            NOAATemp = DATA['NOAAGlobalTemp (degC)'].astype(float).values.reshape(-1,1) 
            BerkeleyEarthTemp = DATA['Berkeley Earth (degC)'].astype(float).values.reshape(-1,1) 
            # Normalises years (for easier processing):
            years_mean = years.mean()
            years_std  = years.std() if years.std() != 0 else 1.0
            years_norm = (years - years_mean) / years_std

            # Stack features together, using np.hstack again.
            inputs = np.hstack([years_norm,NOAATemp,BerkeleyEarthTemp])
            DATA['Return'] = DATA['HadCRUT5 (degC)']
            # Sets the target data as the HadCRUT5 dataset (it seemed to look like the mean of the other datasets, to a degree)
            targets = DATA[['Return']].values

        # Split the data into testing/training percentages (using the relevant input variable).
        splitRatio = self.trainingPercentage.get()
        splitIndex = int(len(inputs) * splitRatio)
        # Split the x/y training/testing data according to this.
        xTrain = inputs[:splitIndex]
        yTrain = targets[:splitIndex]
        xTest  = inputs[splitIndex:]
        yTest  = targets[splitIndex:]
        # This is done slightly differently for the Met Office CSV data:
        if self.dataset == "CSV":
            timeIndex = DATA['Year'].astype(int).values[splitIndex:]
        else:
            timeIndex = DATA.index[splitIndex:]

        # Normalises the inputs (x) and the outputs (y) using standard deviation and means:
        xMean = np.mean(xTrain, axis=0)
        xStd = np.std(xTrain, axis=0)
        xStd = np.where(xStd == 0, 1e-8, xStd)
        yMean = np.mean(yTrain, axis=0)
        yStd = np.std(yTrain, axis=0)
        yStd = np.where(yStd == 0, 1e-8, yStd)

        # Normalise the training x/y data, the "xAllNorm" data is used for the predictions fit later on.
        xTrainNorm = (xTrain - xMean) / xStd
        xAllNorm = (inputs - xMean) / xStd
        yTrainNorm = (yTrain - yMean) / yStd

        # Gets the architecture from the GUI and rebuilds the NN with a correct input size.
        neuronCounts = [int(ne.get()) for (ne, _) in self.layersUI]
        activations   = [ac.get() for (_, ac) in self.layersUI]

        architecture = []
        # Input features are, intuitively, the 1st shape of the normalised x training data.
        inFeatures = xTrainNorm.shape[1]
        # Rebuilds the architecture to prevent the number of neuron counts from not equaling the amount of inputs.
        for i in range(len(neuronCounts) - 1):
            inSize  = inFeatures if i == 0 else neuronCounts[i]
            outSize = neuronCounts[i + 1]
            act = activations[i + 1]
            architecture.append((inSize, outSize, act))
        if len(architecture) > 0:
            architecture[-1] = (architecture[-1][0], architecture[-1][1], "Linear")
        # Ensures repeatability in the data.
        np.random.seed(0)
        # Rebuilds the network with the user-provided learning rate.
        self.nn = NeuralNetwork(architecture, float(self.learningRate.get()))
        print("Rebuilt NN with architecture:", architecture)
        print("")

        # Trains over a batch of data, with initial parameters. Verbose is set to true, for debugging via the console menu.
        batchSize = int(self.batchSize.get())
        MSEHistory = self.nn.training(xTrainNorm, yTrainNorm, epochs=iterations, batchSize=batchSize, optimiser=self.optimiser.get(), lr=float(self.learningRate.get()), verbose=True)

        # Stores normalisation parameters for the plotting function.
        if self.dataset == "YFinance":
            self._norm = {'Normalised X': xAllNorm, 'Mean Y': yMean, 'Standard Y': yStd, 'Test Y': yTest, 'Prices': DATA['Close'].values, 'DATA': DATA, 'Time index': timeIndex,'Split Index': splitIndex, 'Data': DATA}
        else:
            self._norm = {'Normalised X': xAllNorm, 'Mean Y': yMean, 'Standard Y': yStd, 'Test Y': yTest,'Test X': xTest, 'DATA': DATA, 'Time index': timeIndex,'Split Index': splitIndex, 'Data': DATA}
        return xTrain, inputs, DATA, MSEHistory

    # Main plotting function:
    def plot(self):
        # Resets axes / clears graphs.
        self.ax.clear()
        self.ax2.clear()
        self.ax3.clear()
        # Tries training the data, and outputs an error if not possible.
        try:
            inputs, prices, DATA, MSEHistory = self.training(self.epoch, self.ticker)
        except Exception as e:
            print("Training error:", e)
            return
        # Recalls the normalisation parameters that were defined earlier. This is simply a workaround to using "global" variables.
        xAllNorm = self._norm['Normalised X']
        yMean = self._norm['Mean Y']
        yStd = self._norm['Standard Y']
        DATA = self._norm['Data']
        predictionsScaled = self.nn.predict(xAllNorm)  # Normalised predictions.
        predictionsFinal = predictionsScaled * yStd + yMean  # Unscales the data back to plottable values.

        # Reshapes the predictions to match input data (for plotting)
        yPreds = predictionsFinal.reshape(-1)

        # Outputs the change in MSE, if possible (which should always be the case)
        print("Untruncated errors and final accuracies:")
        if len(MSEHistory) > 0:
            print(f"Initial MSE: {MSEHistory[0]}$ and Final MSE: {MSEHistory[-1]}$")
            print("")

        # Recalls the split index, with the necessary inputs.
        splitIndex = self._norm.get('Split Index', int(len(inputs) * self.trainingPercentage.get()))

        if self.dataset == "YFinance":
            # Determine split index (inputs is first return and the training set)
            

            # True unscaled test targets
            yTest = DATA['Return'].values[splitIndex:] # shape (N_test,)

            # Predicted unscaled test targets
            predTest = yPreds[splitIndex:] # shape (N_test,)

            # Ensures that both are 1D numpy arrays (or else there'll be annoying shape errors)
            predTest = np.array(predTest).reshape(-1)
            yTest = np.array(yTest).reshape(-1)

            # Provides some nice output accuracies.
            testMSE = np.mean((predTest - yTest) ** 2)
            testRMSE = np.sqrt(testMSE)
            averagePrice = np.mean(yTest)
            percentError = (testRMSE / averagePrice) * 100

            # Prints these nice output accuracies.
            print(f"Test MSE: {testMSE}$")
            print(f"Test RMSE: {testRMSE}$")
            print(f"Average price: {averagePrice}$")
            print(f"Percentage error: {percentError}%")
            print("")

            # Shows these nice output accuracies in the GUI.
            self.MSE.set(f"{testMSE:.4f}")
            self.RMSE.set(f"{testRMSE:.4f}")
            self.average.set(f"{averagePrice:.4f}")
            self.percentError.set(f"{percentError:.2f}%")

            # Plots the YFinance data.
            self.ax.plot(DATA.index, DATA['Close'].values, label=f"Actual {self.ticker.get()} Price", color='red')
            self.ax.plot(DATA.index, predictionsFinal, label="Neural Network fit", color='black', alpha=0.5)
            self.ax.set_title(f'{self.ticker.get()} Price vs Neural Network Fit')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Price ($)')
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

            # Defines the RMSE loss in the network...
            loss = [np.sqrt(l) for l in MSEHistory]
            
            # Then plots this RMSE loss.
            self.ax2.plot(loss, color='black', alpha=0.5)
            self.ax2.set_title('RMSE loss against epochs')
            self.ax2.set_xlabel('Epochs',fontsize=8)
            self.ax2.set_ylabel('RMSE value')
            self.ax2.grid(True)
            self.canvas2.draw()

            # Flattens the data (so that any outliers are more visible). Thanks to Samir for this code.
            Xflatten = DATA['Close'].values
            maxYflatten = predictionsFinal.flatten().max()

            # Plots the flattened data.
            self.ax3.scatter(Xflatten.flatten(), predictionsFinal.flatten(), color="black", s=1)
            self.ax3.plot([0, maxYflatten], [0, maxYflatten], color='red')
            self.ax3.set_title('True vs predicted returns')
            self.ax3.set_xlabel('True return',fontsize=8)
            self.ax3.set_ylabel('Predicted return')
            self.ax3.legend(["Predictions", "Perfect Prediction Line"])
            self.ax3.grid(True)
            self.canvas3.draw()
        
        elif self.dataset == "LightKurve":

            # Same logic as before:
            splitIndex = self._norm.get('Split Index', splitIndex)
            yTest = np.array(self._norm['Test Y']).reshape(-1)
            xTest = np.array(self._norm['Test X']).reshape(-1)
            predTest = yPreds[splitIndex:].reshape(-1)

            # Provides some nice output accuracies.
            testMSE = np.mean((predTest - yTest) ** 2)
            testRMSE = np.sqrt(testMSE)
            averagePrice = np.mean(yTest)
            percentError = (testRMSE / averagePrice) * 100

            # Prints these nice output accuracies.
            print(f"Test MSE: {testMSE}")
            print(f"Test RMSE: {testRMSE}")
            print(f"Average Flux Change: {averagePrice}")
            print(f"Percentage error: {percentError}%")
            print("")

            # Shows these nice output accuracies in the GUI.
            self.MSE.set(f"{testMSE:.4f}")
            self.RMSE.set(f"{testRMSE:.4f}")
            self.average.set(f"{averagePrice:.4f}")
            self.percentError.set(f"{percentError:.2f}%")

            # No dates / years for this data, so I use a created timeIndex instead.
            timeIndex = self._norm.get('Time index')
            if timeIndex is None or len(timeIndex) != len(yTest):
                # Use a range dataset if not working properly.
                timeIndex = np.arange(len(yTest))

            # Plots the actual vs the predicted flux change, aligned to the same axis.
            self.ax.plot(timeIndex, yTest, label="Actual Flux Change", color='red')
            self.ax.plot(timeIndex, predTest, label="Predicted Flux Change", color='black', alpha=0.5)
            self.ax.set_title('True vs Predicted Flux Change')
            self.ax.set_xlabel('Index')
            self.ax.set_ylabel('Flux Change')
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

            # Defines the RMSE loss in the network...
            loss = [np.sqrt(l) for l in MSEHistory]

            # Then plots this RMSE loss.
            self.ax2.plot(loss, color='black', alpha=0.5)
            self.ax2.set_title('RMSE loss against epochs')
            self.ax2.set_xlabel('Epochs',fontsize=8)
            self.ax2.set_ylabel('RMSE value')
            self.ax2.grid(True)
            self.canvas2.draw()

            # Plots this data in a flattened format.
            self.ax3.scatter(yTest, predTest, color='black', s=1)
            self.ax3.plot([min(xTest.min(), predTest.min()), max(xTest.max(), predTest.max())],
                        [min(xTest.min(), predTest.min()), max(xTest.max(), predTest.max())],
                        color='red')
            self.ax3.set_title('True vs predicted Flux Change')
            self.ax3.set_xlabel('True Flux Change')
            self.ax3.set_ylabel('Predicted Flux Change')
            self.ax3.legend(["Predictions", "Perfect Prediction Line"])
            self.canvas3.draw()

        else:
            
            # Same logic as before.
            yTest = DATA['Return'].values[splitIndex:]
            predTest = yPreds[splitIndex:]

            # Ensures that both are 1D numpy arrays (or else there'll be annoying shape errors)
            predTest = np.array(predTest).reshape(-1)
            yTest = np.array(yTest).reshape(-1)

            # Provides some nice output accuracies.
            testMSE = np.mean((predTest - yTest) ** 2)
            testRMSE = np.sqrt(testMSE)
            averagePrice = np.mean(yTest)
            percentError = (testRMSE / averagePrice) * 100

            # Prints these nice output accuracies.
            print(f"Test MSE: {testMSE} degrees centigrade")
            print(f"Test RMSE: {testRMSE} degrees centigrade")
            print(f"Average Temperature: {averagePrice} degrees centigrade")
            print(f"Percentage error: {percentError} %")
            print("")

            # Same logic as before:
            timeIndex = self._norm.get('Time index')
            if timeIndex is None or len(timeIndex) != len(yTest):
                timeIndex = np.arange(len(yTest))

            # Shows these nice output accuracies in the GUI.
            self.MSE.set(f"{testMSE:.4f}")
            self.RMSE.set(f"{testRMSE:.4f}")
            self.average.set(f"{averagePrice:.4f}")
            self.percentError.set(f"{percentError:.2f}%")
            inputs = DATA['Year'].values

            # Plots the temperature change data
            self.ax.plot(DATA['Year'], DATA['NOAAGlobalTemp (degC)'], label=f"Actual Temperature", color='red') #timeIndex, yTest
            self.ax.plot(DATA['Year'], predictionsFinal, label="Neural Network fit", color='black', alpha=0.5) #timeIndex, predTest,
            self.ax.set_title(f'Temperature vs Neural Network Fit')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Temperature Change oC')
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

            # Defines the RMSE loss in the network...
            loss = [np.sqrt(l) for l in MSEHistory]

            # Then plots this RMSE loss.
            self.ax2.plot(loss, color='black', alpha=0.5)
            self.ax2.set_title('RMSE loss against epochs')
            self.ax2.set_xlabel('Epochs',fontsize=8)
            self.ax2.set_ylabel('RMSE value')
            self.ax2.grid(True)
            self.canvas2.draw()

            # Same logic as before. Flattens the data, and plots it.
            Xflatten = DATA['NOAAGlobalTemp (degC)'].values
            maxYflatten = predictionsFinal.flatten().max()
            minYflatten = predictionsFinal.flatten().min()

            self.ax3.scatter(Xflatten.flatten(), predictionsFinal.flatten(), color="black", s=1)
            self.ax3.plot([minYflatten, maxYflatten], [minYflatten, maxYflatten], color='red')
            self.ax3.set_title('True vs predicted temperature')
            self.ax3.set_xlabel('True temperature',fontsize=8)
            self.ax3.set_ylabel('Predicted temperature')
            self.ax3.legend(["Predictions", "Perfect Prediction Line"])
            self.ax3.grid(True)
            self.canvas3.draw()

            # Shows the given temperature change for a set year (and prints this to more significant figures)
            self.predictTemp.set(f"Fit Value for {self.yearTemp.get()} predicted to be: {yPreds[int(self.yearTemp.get()-1850)]:.4f} °C")

            print(f"Fit value in {self.yearTemp.get()} predicted to be: {yPreds[int(self.yearTemp.get()-1850)]:.4f} °C")


# Runs the application
if __name__ == "__main__":
    np.random.seed(0)  # Makes the "random" numbers be the same each time the program is run, for repeatability.
    a = app()
    # Adds a cool logo :)
    a.iconbitmap(os.path.join(basedir, "icon.ico"))
    a.mainloop()

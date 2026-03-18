import numpy as np
import matplotlib.pyplot as plt

"""
Temperature Data code:

As a note, this code will NOT work just as it is -  as this is designed to work within my GUI code.
The first bit (up to line 28) shows how to download and read the dataset. Simply plot inputs against targets.

The later code shows how I plot the data, but this will be different for your own neural networks.
"""

DATA = pd.read_csv(resource_path("TempData.csv"))

# Take the years data and reshape into the correct format.
years = DATA['Year'].astype(float).values.reshape(-1,1)   # Shape of (N,1)
# Same deal for the NOAAGlobalTemp and Berkeley Earth data.
NOAATemp = DATA['NOAAGlobalTemp (degC)'].astype(float).values.reshape(-1,1) 
BerkeleyEarthTemp = DATA['Berkeley Earth (degC)'].astype(float).values.reshape(-1,1) 
# Normalises years (for easier processing):
yearsMean = years.mean()
yearsStd  = years.std() if years.std() != 0 else 1.0
yearsNorm = (years - yearsMean) / yearsStd
# Stack features together, using np.hstack again.
inputs = np.hstack([years_norm,NOAATemp,BerkeleyEarthTemp])
DATA['Return'] = DATA['HadCRUT5 (degC)']
# Sets the target data as the HadCRUT5 dataset (it seemed to look like the mean of the other datasets, to a degree)
targets = DATA[['Return']].values

"""
Plotting data (this makes more sense in my GUI.py file)
xStd, yStd, yTrainNorm, timeIndex, self._norm[...], etc. are defined in the GUI.py file.

I recommend either using / modifying my GUI & NeuralNetwork code to plot this or using the import code above and fitting it to your own code
(the latter option is MUCH easier).

"""
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
yTest = DATA['Return'].values[splitIndex:]
predTest = yPreds[splitIndex:]

# Ensures that both are 1D numpy arrays (or else there'll be annoying shape errors)
predTest = np.array(predTest).reshape(-1)
yTest = np.array(yTest).reshape(-1)
xTest = self._norm['Test X']
xTest = np.array(xTest).reshape(-1)
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
self.ax3.scatter(Xflatten.flatten(), predictionsFinal.flatten(), color="black", s=1)
self.ax3.plot([0, maxYflatten], [0, maxYflatten], color='red')
self.ax3.set_title('True vs predicted temperature')
self.ax3.set_xlabel('True temperature',fontsize=8)
self.ax3.set_ylabel('Predicted temperature')
self.ax3.legend(["Predictions", "Perfect Prediction Line"])
self.ax3.grid(True)
self.canvas3.draw()

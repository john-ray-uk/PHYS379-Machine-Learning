import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
"""
Linear Regression code
Uses OpenPyXL 3.1.5, NumPy 1.25.2, Pandas 2.3.3, and Matplotlib 3.10.8, with Python version 3.10.19.
If the code does not run correctly, try setting up a Python environment with these precise versions.
"""
mpl.rcParams['font.family'] = 'Times New Roman'
# Reads Advertising.xlsx file
alldata = pd.read_excel(r'C:\PERSONAL FOLDERS\Samir\Lancs\PHYS379\Advertising.xlsx')
# Gets the radio data bit, ignores the header (only reads the first 200 data points)
radiodata = alldata['radio'].head(200)
salesdata = alldata['sales'].head(200)
# Append values to the lists below.
sales = []
radio = []
linear = []
indices = []

for index,value in radiodata.items():
    radio.append(value)
    indices.append(index)
for index,sale in salesdata.items():
    sales.append(sale)
# Prints number of data points, N
N = len(alldata)
#N = 1
print(f"Number of data points: {N}")
"""
INPUT PARAMETERS:
"""
trainingRate = 0.001
epochs = 5002 # ACTUAL value is - 2 this value, somehow.
w = 0
b = 0
wList = []
bList = []

# Standard Linear Regression function (needs fixing for weirder datasets)
def linearRegression(N,x,y,w,b,trainingRate,epochs):
    X = np.asarray(x,dtype=float)
    Y = np.asarray(y,dtype=float)
    dLdW = 0
    dLdB = 0
    l = 0
    newValue = 0
    for i in range(epochs):
        # Can have np.sum(.../N), instead. Not sure what's better.
        dLdW = np.mean(-(2*(X)*(Y-(w*X + b))))
        dLdB = np.mean(-(2*(Y-(w*X + b))))
        l = np.mean((Y-(w*X + b))**2)
        w = w - trainingRate*dLdW
        b = b - trainingRate*dLdB
        wList.append(w)
        bList.append(b)
    newValue = w*N + b
    return newValue,l

radioValue,l = linearRegression(N,radio,sales,w,b,trainingRate,epochs)
#stockValue,l = linearRegression(N,stocks,sales,w,b,trainingRate,epochs)
# TV and Newspaper sales' datasets are trash :(

# Function for R^2 error / coefficient of determination -> gives VERY bad results for the radio sales data
def accuracy(y,expectedy):
    # Used to find errors in Linear regression
    Y = np.asarray(y,dtype=float)
    expY = np.asarray(expectedy,dtype=float)
    Rsquarederror = 1 - (np.sum((expY-Y)**2)/np.sum((Y-np.mean(Y))**2))

    return Rsquarederror

# Calls the above R^2 error function, for the radiodata
radioError = accuracy(sales,radioValue)

print(f"R^2 error is {radioError}")
print("                            ")
print(f"Unprocessed radio sales data: {radio}")
print("                            ")
print(f"Linear regressed y radio datapoint: {radioValue}")
print("                            ")
print(f"Radio data mean-squared error: {l}")
print("                            ")

'''
# Radio sales data graph (this data is the most reliable / functional, it seems)
plt.scatter(indices, radio, color="red", label="Radio data")
plt.axline((0,0), (200, radioValue), color='black', linewidth=2, label="Linear Regression")
plt.legend()
plt.show()
'''
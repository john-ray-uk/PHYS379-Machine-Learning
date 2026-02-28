import numpy as np
from keras.datasets import mnist
from sklearn import linear_model

### This is the Linear regression code for MNIST ###
### It sucks SUPER hard but is slightly better than expected tbf, do not use for anything except for comparisons ###

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Flatting into vectors
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

# Squishing into values between 0 and 1 (Normalisation)
train_X = train_X / 255.0
test_X = test_X / 255.0

def linearising(train_X, train_y, test_X, test_y):
    trained_model = linear_model.LinearRegression()
    trained_model.fit(train_X,train_y)

    y_predicted = trained_model.predict(test_X)

    loss = np.mean((y_predicted - test_y)**2)
    
    y_predicted_round = np.clip(np.round(y_predicted), 0, 9)

    accuracy = np.mean(y_predicted_round == test_y)

    return y_predicted, loss, accuracy

predictions, loss, acc = linearising(train_X, train_y, test_X, test_y)

print(predictions, loss, acc)
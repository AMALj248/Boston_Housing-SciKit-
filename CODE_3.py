#this uses the toy library  which is inbluilt in scikit library




#IMPORTING THE REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#LOADING THE DATASET TO REQUIRED FUNCTIONS
#using the toy dataset which is inbuilt in sklearn library
from sklearn.datasets import load_boston
boston = load_boston()
data = pd.DataFrame(boston.data)
data.coloumns = boston.feature_names
data.head(10)
data['Price'] = boston.target
data.head()

# showing the boston dataset in tabular cloumn
data.describe()

#information about the boston datatypes
data.info()

#INPUT DATASET
x = boston.data

#OUTPUT DATASET
y=boston.target

#SPLITTING THE DATA USING SIMPLE TEST TRAIN SLIT OF DATA
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x ,y , test_size = 0.2, random_state=0)

#printing the dimensions of splitted data
print("x_train shape :", x_train.shape)
print("x_test shape : ", x_test.shape)
print("y_train shape :",y_train.shape)
print("y_test shape :", y_test.shape)

#applying linear regression model to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test results
y_pred = regressor.predict(x_test)

#plotting the scatter plot  between y_test and y_predicited
plt.scatter(y_test, y_pred, c='green')
plt.xlabel("Price: in $1000 s")
plt.ylabel("predicted value ")
plt.title("True value vs predicted value : Linear Regression ")
plt.show()


#Result from the MULTI LINEAR REGRESSION MODEL
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(" Mean Square Error : ", mse)
m =math.sqrt(mse)
print (m)

#Mean absolute error
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred- y_test))))



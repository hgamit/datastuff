# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 02:09:50 2019

@author: hmnsh
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the training set
dataset_train = pd.read_csv('./marketdata/zsh20.csv')
#dataset_train = dataset_train[dataset_train["Volume"] != 0]

dataset_train.Time = pd.to_datetime(dataset_train.Time.str.replace('D', 'T'))
dataset_train = dataset_train.sort_values('Time')

dataset_train['Year'] = dataset_train['Time'].dt.year
dataset_train['Month'] = dataset_train['Time'].dt.month
dataset_train['Day'] = dataset_train['Time'].dt.day

dataset_train.set_index('Time', inplace=True)
print(dataset_train.shape)

training_set = dataset_train.iloc[:, 1:7].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 15 timesteps and 1 output - we use last 15 prices to predict next. 
#This takes data from 15th row onwards
X_whole = []
y_whole = []

sequence_size = 4
for i in range(sequence_size, len(training_set_scaled)):
    X_whole = np.append(X_whole, training_set_scaled[i-sequence_size:i, 2])
    #X_whole = np.append(X_whole, training_set_scaled[i, 6])
    #X_whole = np.append(X_whole, training_set_scaled[i, 7])
    y_whole.append(training_set_scaled[i, 2])

sz = training_set_scaled.shape[0]-sequence_size
X_whole, y_whole = np.array(X_whole.reshape(sz,sequence_size)), np.array(y_whole)


#Train -valid and Test split in time order
X_train = X_whole[0:420,:].copy()
X_test = X_whole[420:,:].copy()

y_train = y_whole[0:420].copy()
y_test = y_whole[420:].copy()


# make a persistence forecast
def persistence(last_ob, n_seq):
	return [last_ob for i in range(n_seq)]
 
# evaluate the persistence model
def make_forecasts(train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = persistence(X[-1], n_seq)
		# store the forecast
		forecasts.append(forecast)
	return forecasts
 
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = test[:,(n_lag+i)]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
 
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue fig = plt.figure(figsize=(12,8))
	fig = plt.figure(figsize=(12,8))
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s][0]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()


from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
# configure
n_lag = 1
n_seq = 3
n_test = len(X_test)
# make forecasts
forecasts = make_forecasts(X_train, X_test, n_lag, n_seq)
# evaluate forecasts
evaluate_forecasts(X_test, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(DataFrame(X_whole), forecasts, n_test+2)

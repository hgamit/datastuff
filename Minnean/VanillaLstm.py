# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:49:00 2019

@author: hmnsh
"""

# univariate lstm example
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
print(os.getcwd())

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


def nextpred(model, X_valid, n_steps, num_of_pred):
    #predicting on new data
    xw=X_valid[-1,:].copy()
    xw=xw.reshape(1,-1)
    xw=np.reshape(xw,(xw.shape[0], xw.shape[1], 1))    
    y_pred_w = []
    
    for i in range(0,num_of_pred):
        
        y_pred1=model.predict(xw)
        y_pred_w = np.append(y_pred_w,y_pred1)
        
        #print("Before", xw)
    #    print(xwvap)
    #    print(xvchange)
        #print(y_pred1)
        
        if n_steps>1:
            for j in range(n_steps-1):
                xw[:,j]=xw[:,j+1]
            
        xw[:,n_steps-1]=y_pred1
    
        #print("After",xw)
    #    print(xwvap)
    #    print(xvchange)
        
    return y_pred_w

# load dataset
# Importing the training set
dataset_train = pd.read_csv('C:/Users/hmnsh/repos/datastuff/Minnean/marketdata/zsh20.csv')
#dataset_train = dataset_train[dataset_train["Volume"] != 0]

dataset_train.Time = pd.to_datetime(dataset_train.Time.str.replace('D', 'T'))
dataset_train = dataset_train.sort_values('Time')

dataset_train.set_index('Time', inplace=True)
print(dataset_train.shape)

# define input sequence
training_set = dataset_train.iloc[:, 3:4].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
raw_seq = sc.fit_transform(training_set)

# choose a number of time steps
n_steps = 3
# split into samples
X_whole, y_whole = split_sequence(raw_seq, n_steps)

#Train -valid and Test split in time order
X_train = X_whole[0:362,:].copy()
X_valid = X_whole[362:420,:].copy()
X_test = X_whole[420:,:].copy()

y_train = y_whole[0:362].copy()
y_valid = y_whole[362:420].copy()
y_test = y_whole[420:].copy()

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features))))
model.add(Bidirectional(LSTM(30, activation='relu', return_sequences=True)))
model.add(Bidirectional(LSTM(10, activation='relu')))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=["mean_squared_error"])
# fit model
hist = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=200, verbose=0)

plt.figure(figsize=(14,12))
plt.suptitle('Training Evaluation', fontsize=24)

plt.subplot(2,2,1)
#Plotting Training history
print(hist.history.keys())

# Visualising the results
plt.plot(hist.history['loss'], color = 'blue',  label = 'train_loss')
plt.plot(hist.history['val_loss'], color = 'red',  label = 'val_loss')
plt.title('Losses')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
#plt.show()
plt.subplot(2,2,2)
# Visualising the results
plt.plot(hist.history['mean_squared_error'], color = 'blue',  label = 'train_mean_squared_error')
plt.plot(hist.history['val_mean_squared_error'], color = 'red',  label = 'val_mean_squared_error')
plt.title('mean_squared_error')
plt.xlabel('epoch')
plt.ylabel('mean_squared_error')
plt.legend()
plt.show()

# demonstrate prediction
yhat = model.predict(X_test, verbose=0)
real_stock_price  = sc.inverse_transform(y_test)
predicted_stock_price  = sc.inverse_transform(yhat)

#raw preds
y_raw_price = nextpred(model, X_valid, n_steps, X_test.shape[0])
predicted_raw_price = sc.inverse_transform(y_raw_price.reshape(-1, 1))

from sklearn.metrics import mean_absolute_error
from math import sqrt
# report performance
from sklearn.metrics import mean_squared_error
print("sqrt mean_squared_error: ", sqrt(mean_squared_error(real_stock_price, predicted_stock_price)))
print("mean_squared_error: ", mean_squared_error(real_stock_price, predicted_stock_price))
mean_absolute_error = mean_absolute_error(real_stock_price, predicted_stock_price)
print("mean_absolute_error: ", mean_absolute_error)

fig = plt.figure(figsize=(12,8))
# Visualising the results
plt.plot(real_stock_price, color = 'red',  marker='o', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue',  marker='o', label = 'Predicted Stock Price')
plt.plot(predicted_raw_price, color = 'green',  marker='o', label = 'Predicted Raw Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

#real_stock_price, predicted_stock_price
SS_Residual = sum((real_stock_price-predicted_stock_price)**2)
SS_Total = sum((real_stock_price-np.mean(real_stock_price))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(real_stock_price)-1)/(len(real_stock_price)-X_train.shape[1]-1)
print ("R Squared:", r_squared, "\nAdjusted R Squared:", adjusted_r_squared)
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 01:42:02 2019

@author: hmnsh
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


sequence_size = 2
# Importing the training set
dataset_train = pd.read_csv('./marketdata/zsh20.csv') #, nrows=120
#dataset_train = dataset_train[dataset_train["Volume"] != 0]
#dataset_train["vwap"] = ((dataset_train["High"] + dataset_train["Low"] + dataset_train["Last"])/3) * dataset_train["Volume"]
#nt = dataset_train["Volume"].groupby(dataset_train.index // sequence_size).sum()


training_set = dataset_train.iloc[:, 2:8].values

dataset_train.Time = pd.to_datetime(dataset_train.Time.str.replace('D', 'T'))
dataset_train = dataset_train.sort_values('Time')
dataset_train.set_index('Time', inplace=True)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 15 timesteps and 1 output - we use last 15 prices to predict next. 
#This takes data from 15th row onwards
X_whole = []
y_whole = []
sequence_size = 2
for i in range(sequence_size, len(training_set_scaled)):
    X_whole = np.append(X_whole, training_set_scaled[i-sequence_size:i, 0])
    vwap = 0
    vol = 0
    for j in range(sequence_size):
        if (j+i<training_set_scaled.shape[0]):
            vwap += (((np.sum(training_set_scaled[j+i, 0:3]))/3) * training_set_scaled[j+i, 4])
            vol += training_set_scaled[j+i, 4]
    if vwap !=0  and vol != 0:
        X_whole = np.append(X_whole, vwap/vol)
        y_whole.append(training_set_scaled[i, 0])

sz = training_set_scaled.shape[0]-sequence_size
X_whole, y_whole = np.array(X_whole.reshape(sz,sequence_size+1)), np.array(y_whole)
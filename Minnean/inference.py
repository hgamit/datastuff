# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:39:40 2019

@author: hmnsh
"""


# Recurrent Neural Network

#https://www.youtube.com/watch?v=zwqwlR48ztQ

# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the training set
dataset_train = pd.read_csv('./marketdata/zsh20.csv')
#dataset_train = pd.read_csv('./marketdata/zsk20_daily_price-history-11-04-2019.csv')
                            #soybean/zsk20_daily_price-history-10-25-2019.csv')
#dataset_train = dataset_train[dataset_train["Volume"] != 0]

dataset_train.Time = pd.to_datetime(dataset_train.Time.str.replace('D', 'T'))
dataset_train = dataset_train.sort_values('Time')
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

X_whole_vchange = []
y_whole_vchange = []


X_whole_wvap = []
y_whole_wvap = []

sequence_size = 1
for i in range(sequence_size, len(training_set_scaled)):
    X_whole = np.append(X_whole, training_set_scaled[i-sequence_size:i, 2])
    #X_whole = np.append(X_whole, training_set_scaled[i, 6])
    #X_whole = np.append(X_whole, training_set_scaled[i, 7])
    vwap = 0
    vol = 0
    vchg = 0
    for j in range(sequence_size):
        if (j+i<training_set_scaled.shape[0]):
            vwap += (((np.sum(training_set_scaled[j+i, 0:3]))/3) * training_set_scaled[j+i, 4])
            vchg += (((training_set_scaled[j+i, 3])) * training_set_scaled[j+i, 4])
            vol += training_set_scaled[j+i, 4]
    #if vwap !=0  and vol != 0:
    X_whole = np.append(X_whole, vwap/vol)
    X_whole = np.append(X_whole, vchg/vol)
    y_whole.append(training_set_scaled[i, 2])

sz = training_set_scaled.shape[0]-sequence_size
X_whole, y_whole = np.array(X_whole.reshape(sz,sequence_size+2)), np.array(y_whole)

for i in range(sequence_size, len(X_whole)):
    X_whole_wvap = np.append(X_whole_wvap,X_whole[i-sequence_size:i, sequence_size])
    X_whole_wvap = np.append(X_whole_wvap,X_whole[i, 0])
    X_whole_wvap = np.append(X_whole_wvap,X_whole[i, sequence_size+1])
    y_whole_wvap.append(X_whole[i, sequence_size])
    
    X_whole_vchange = np.append(X_whole_vchange,X_whole[i-sequence_size:i, sequence_size+1])
    X_whole_vchange = np.append(X_whole_vchange,X_whole[i, 0])
    X_whole_vchange = np.append(X_whole_vchange,X_whole[i, sequence_size])
    y_whole_vchange.append(X_whole[i, sequence_size+1])

sz1 = X_whole.shape[0]-sequence_size
X_whole_wvap, y_whole_wvap = np.array(X_whole_wvap.reshape(sz1,sequence_size+2)), np.array(y_whole_wvap)
X_whole_vchange, y_whole_vchange = np.array(X_whole_vchange.reshape(sz1,sequence_size+2)), np.array(y_whole_vchange)

from keras.models import model_from_json
def load_model(model_name):
    # load json and create model
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+".h5")
    print("Loaded model from disk")
    return loaded_model



soy_close = load_model("zsh20_1000epoch_Rsqr99_oct25")
soy_vwap = load_model("zsh20_1000epoch_Rsqr99_vwap_oct25")
soy_vchange = load_model("./models25oct/zsh20_1500epoch_Rsqr91_vchange")

#soy_close = load_model("zsk20_1000epoch_Rsqr99")
#soy_vwap = load_model("zsk20_1000epoch_Rsqr99_vwap")
#soy_vchange = load_model("zsk20_1500epoch_Rsqr91_vchange")

#soy_close = load_model("zsn20_1000epoch_Rsqr99")
#soy_vwap = load_model("zsn20_1000epoch_Rsqr99_vwap")
#soy_vchange = load_model("zsn20_1500epoch_Rsqr91_vchange")



#predicting on new data
xw=X_whole[-1,:].copy()
xw=xw.reshape(1,-1)
xw=np.reshape(xw,(xw.shape[0], xw.shape[1], 1))

xwvap = X_whole_wvap[-1,:].copy()
xwvap=xwvap.reshape(1,-1)
xwvap=np.reshape(xwvap,(xwvap.shape[0], xwvap.shape[1], 1))

xvchange=X_whole_vchange[-1,:].copy()
xvchange=xvchange.reshape(1,-1)
xvchange=np.reshape(xvchange,(xvchange.shape[0], xvchange.shape[1], 1))

y_pred_w = []

num_of_pred = 35

for i in range(0,num_of_pred):
    
    y_pred1=soy_close.predict(xw)
    test_data = np.zeros(shape=(len(y_pred1), 6) )
    test_data[:,2] = y_pred1
    y_transformed  = sc.inverse_transform(test_data)[:,2]
    y_pred_w = np.append(y_pred_w,y_transformed)
    
    
    y_pred2=soy_vwap.predict(xwvap)
    
    
    y_pred3=soy_vchange.predict(xvchange)
    
    print(xw)
    print(xwvap)
    print(xvchange)
    print(y_pred1,y_pred2,y_pred3)
    
    for j in range(sequence_size):
        xw[:,j]=xw[:,j+1]
        xwvap[:,j]=xwvap[:,j+1]
        xvchange[:,j]=xvchange[:,j+1]
        
    xw[:,sequence_size-1]=y_pred1
    xw[:,sequence_size]=y_pred2
    xw[:,sequence_size+1]=y_pred3
    
    xwvap[:,sequence_size-1]=y_pred2
    xwvap[:,sequence_size]=y_pred1
    xwvap[:,sequence_size+1]=y_pred3
    
    xvchange[:,sequence_size-1]=y_pred3
    xvchange[:,sequence_size]=y_pred1
    xvchange[:,sequence_size+1]=y_pred2
    
    print(xw)
    print(xwvap)
    print(xvchange)
    
print(y_pred_w)


#10.87240385




    

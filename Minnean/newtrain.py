# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 02:49:02 2019

@author: hmnsh
"""

# Part 1 - Data Preprocessing

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

sequence_size = 12
for i in range(sequence_size, len(training_set_scaled)):
    X_whole = np.append(X_whole, training_set_scaled[i-sequence_size:i, 2])
    #X_whole = np.append(X_whole, training_set_scaled[i, 6])
    #X_whole = np.append(X_whole, training_set_scaled[i, 7])
    y_whole.append(training_set_scaled[i, 2])

sz = training_set_scaled.shape[0]-sequence_size
X_whole, y_whole = np.array(X_whole.reshape(sz,sequence_size)), np.array(y_whole)


#Train -valid and Test split in time order
X_train = X_whole[0:362,:].copy()
X_valid = X_whole[362:420,:].copy()
X_test = X_whole[420:,:].copy()

y_train = y_whole[0:362].copy()
y_valid = y_whole[362:420].copy()
y_test = y_whole[420:].copy()

# Part 2 - Building the RNN

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))
#X_whole = np.reshape(X_whole, (X_whole.shape[0], X_whole.shape[1], 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

from keras import optimizers

adm = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
# Compiling the RNN
regressor.compile(optimizer = adm, loss = 'mean_squared_error' , metrics=["mean_squared_error"])

# Fitting the RNN to the Training set #, validation_data=(X_valid, y_valid)
hist = regressor.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs = 200, batch_size = 32, verbose=0)


#regressor.save_weights("zns20_400epoch_Rsqr98.h5")
#print("Saved model to disk")
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")

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


from keras.models import model_from_json
# serialize model to JSON
model_json = regressor.to_json()
with open("regressor.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("regressor.h5")
print("Saved model to disk")


#create empty table with 6 fields to inverse
test_data = np.zeros(shape=(len(y_test), 6) )
test_data[:,2] = y_test
real_stock_price = sc.inverse_transform(test_data)[:,2]
real_stock_price.shape[0]

#from keras.models import model_from_json
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



soy_close = load_model("regressor")

#soy_vwap = load_model("zsh20_1000epoch_Rsqr99_vwap_oct25")
#soy_vchange = load_model("./models25oct/zsh20_1500epoch_Rsqr91_vchange")

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

y_pred_w = []

num_of_pred = real_stock_price.shape[0]

for i in range(0,num_of_pred):
    
    y_pred1=soy_close.predict(xw)
    test_data = np.zeros(shape=(len(y_pred1), 6) )
    test_data[:,2] = y_pred1
    y_transformed  = sc.inverse_transform(test_data)[:,2]
    y_pred_w = np.append(y_pred_w,y_transformed)
    
    #print("Before", xw)
#    print(xwvap)
#    print(xvchange)
    #print(y_pred1)
    
    if sequence_size>1:
        for j in range(sequence_size-1):
            xw[:,j]=xw[:,j+1]
        
    xw[:,sequence_size-1]=y_pred1

    #print("After",xw)
#    print(xwvap)
#    print(xvchange)
    
print(y_pred_w)
predicted_stock_price = y_pred_w
predicted_stock_price.shape[0]


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

import datetime
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import haversine

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('./marketdata/zsh20.csv')
#dataset_train = dataset_train[dataset_train["Volume"] != 0]
training_set = dataset_train.iloc[:, 4:5].values

dataset_train.Time = pd.to_datetime(dataset_train.Time.str.replace('D', 'T'))
dataset_train = dataset_train.sort_values('Time')
dataset_train.set_index('Time', inplace=True)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 15 timesteps and 1 output - we use last 15 prices to predict next. 
#This takes data from 15th row onwards
X_whole = []
y_whole = []
sequence_size = 2
for i in range(sequence_size, len(training_set_scaled)):
    X_whole.append(training_set_scaled[i-sequence_size:i, 0])
    y_whole.append(training_set_scaled[i, 0])
X_whole, y_whole = np.array(X_whole), np.array(y_whole)

#Train -valid and Test split in time order
X_train = X_whole[0:362,:].copy()
X_valid = X_whole[362:420,:].copy()
X_test = X_whole[420:,:].copy()

y_train = y_whole[0:362].copy()
y_valid = y_whole[362:420].copy()
y_test = y_whole[420:].copy()


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))


print('Training and making predictions')
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'max_depth': 6, 
    'learning_rate': 0.1,
    'verbose': 0, 
    'metric': 'rmsle'}

n_estimators = 60

n_iters = 5
preds_buf = []
err_buf = []
d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid)
watchlist = [d_valid]

model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)
preds = model.predict(X_test)

real_stock_price = sc.inverse_transform(y_test.reshape(-1, 1))
# Part 3 - Making the predictions and visualising the results
predicted_stock_price = sc.inverse_transform(preds.reshape(-1, 1))

from sklearn.metrics import mean_absolute_error
from math import sqrt
# report performance
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
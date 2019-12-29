#https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/
#https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/


# univariate multi-step vector-output stacked lstm example
# Importing the libraries
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional,Dropout
from keras.layers import Dense
from keras.regularizers import L1L2
from numpy import hstack, vstack, newaxis
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.Session(config=config)
%matplotlib inline

def qk(y_pred, y):
    return cohen_kappa_score(y_pred, y, weights='quadratic')

def plot_eval(real_stock_price, predicted_stock_price):
    print("sqrt mean_squared_error: ", sqrt(mean_squared_error(real_stock_price, predicted_stock_price)))
    print("mean_squared_error: ", mean_squared_error(real_stock_price, predicted_stock_price))
    mean_absolute_er = mean_absolute_error(real_stock_price, predicted_stock_price)
    print("mean_absolute_error: ", mean_absolute_er)
    #print("quadratic_cohen_kappa_score: ", qk(predicted_stock_price, real_stock_price))
    #real_stock_price, predicted_stock_price
    SS_Residual = sum((real_stock_price-predicted_stock_price)**2)
    SS_Total = sum((real_stock_price-np.mean(real_stock_price))**2)
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adjusted_r_squared = 1 - (1-r_squared)*(len(real_stock_price)-1)/(len(real_stock_price)-X_train.shape[1]-1)
    print ("R Squared:", r_squared, "\nAdjusted R Squared:", adjusted_r_squared)
    out_arr = np.subtract(predicted_stock_price, real_stock_price)
    fig = plt.figure(figsize=(12,8))
    # Visualising the results
    plt.plot(real_stock_price, color = 'red',  marker='o', label = 'Real Stock Price')
    plt.plot(predicted_stock_price, color = 'blue',  marker='o', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def add_datepart(df, fldnames, drop=False, time=False, errors="raise"):	
    if isinstance(fldnames,str):
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64
        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        #targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[n] = getattr(fld.dt, n.lower())
        df['Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop: df.drop(fldname, axis=1, inplace=True)
    df.drop(['Elapsed','Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'], axis=1, inplace=True)
        

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	last_row = []
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			last_row = sequence[i:end_ix]
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y), last_row


def calc_feat(sequence, n_steps_in, n_steps_out, num_of_features=3):
    X= []
    size = len(sequence)
    skips = n_steps_in -1
    sz = size-skips
    for i in range(0,size-1): ##loop through each row in sequence
        vwap = 0
        vol = 0
        vchg = 0
        #print("i:", i)
        
        for j in range(n_steps_in): ## vwap rows loop
            #print("j:",j, " sz", sz, " size", size, " i", i)
            #print("High, Low, Close", sequence[i, 0:3])
            #print("Vol", sequence[i, 4])
            #print("Change", sequence[i, 3])
            vwap += (((np.sum(sequence[i, 0:3]))/3) * sequence[i, 4])
            vchg += (((sequence[i, 3])) * sequence[i, 4])
            vol += sequence[i, 4]
            i = i+1
            #print("vwap:", vwap, " vchg", vchg)
        #if vwap !=0  and vol != 0:
        #print("vol out:", vol)
        #print("---------------------------")
        X = np.append(X, vwap/vol) #vwap
        X = np.append(X, vchg/vol) #Change
        #X = np.append(X, sequence[i, 4]) #volume
        #X = np.append(X, sequence[i, 5]) #openInt
        print(sequence[i, 6])
        X = np.append(X, sequence[i, 6]) #DTWEXB
        #X = np.append(X, sequence[i, 7]) #Year
        #X = np.append(X, sequence[i, 8]) #Mnt
        #X = np.append(X, sequence[i, 9]) #Week
        #X = np.append(X, sequence[i, 10]) #DayofMonth
        #X = np.append(X, sequence[i, 11]) #Dayofweek
        #X = np.append(X, sequence[i, 12]) #Dayofyear
        if (i == size):
            break
        i = i-skips    


    X = np.array(X.reshape(sz, num_of_features))
    print(X.shape)
    last_feat = X[-n_steps_out,:]
    X = X[0:-n_steps_out,:]
    print(X.shape)
    #X = X[:-(n_steps_out), :]
    return X, last_feat


def calc_pred(sequence, n_steps_in, n_steps_out):
    last_row = sequence[-(n_steps_in):]
    X= []
    k = 0 # row itreator
    size = len(last_row) + 1
    for i in range(n_steps_in, size):
        vwap = 0
        vol = 0
        vchg = 0
        for j in range(n_steps_in):
            if (k<i):
                print ("row ",last_row[k, 0:3])
                print ("vol ",last_row[k, 4])
                print ("chg ",last_row[k, 4])
                vwap += (((np.sum(last_row[k, 0:3]))/3) * last_row[k, 4])
                vchg += (((last_row[k, 3])) * last_row[k, 4])
                print ("vwap ",vwap)
                print ("vchg ",vchg)
                vol += last_row[k, 4]
                k = k+1
        
        X = np.append(X, vwap/vol) #vwap
        X = np.append(X, vchg/vol) #Change
    
    num_of_features = 2 #num_of_features
    sz = size-(n_steps_in)
    #X = np.array(X.reshape(sz, num_of_features)) 
    pred_seq = last_row[:,2]
    pred_seq = np.concatenate((pred_seq , X), axis=0)
    return pred_seq

# load dataset
# Importing the training set
dataset_train = pd.read_csv('C:/Users/hmnsh/repos/datastuff/Minnean/marketdata/zsh20_daily_price-history-12-13-2019.csv' ) #, nrows=10
#dataset_train = dataset_train[dataset_train["Volume"] != 0]

dataset_dxy = pd.read_csv('C:/Users/hmnsh/repos/datastuff/Minnean/marketdata/DTWEXBDEC13.csv')

dataset_dxy = dataset_dxy.replace(".", np.nan).fillna(method='ffill')
dataset_train = pd.merge(dataset_train, dataset_dxy,  left_on='Time', right_on='DATE', how='left')
dataset_train.Time = pd.to_datetime(dataset_train.Time.str.replace('D', 'T'))
dataset_train = dataset_train.sort_values('Time')
dataset_train.drop(['DATE'], inplace=True, axis=1)
#dataset_train = dataset_train.astype({"Volume": np.float64, "Open Int": np.float64})
# choose a number of time stepscls
n_steps_in, n_steps_out = 5, 1
add_datepart(dataset_train, 'Time')
dataset_train.set_index('Time', inplace=True)
print(dataset_train.shape)

# define input sequence
training_feat = dataset_train.iloc[:, 1:15].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (-1, 1))
training_feat_scaled = sc.fit_transform(training_feat)

#X_feat = calc_feat(training_feat, n_steps_in, n_steps_out)
X_feat, pred_feat = calc_feat(training_feat_scaled, n_steps_in, n_steps_out)
#pred_seq= calc_pred(training_feat_scaled, n_steps_in, n_steps_out)
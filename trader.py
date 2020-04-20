# -*- coding: utf-8 -*-
"""
modul: trader
modul author: Christoph Doerr

http://kaushik316-blog.logdown.com/posts/1964522
https://github.com/bukosabino/ta/blob/master/ta/volume.py

"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import os.path
#from selenium import webdriver #For Prediction
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import utils as np_utils

import calculateIndicators as indicators
import utils as utils
import evaluateIndicators as eva
import model_utils as model_utils

stock_data_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/data/stock_data/'
indicator_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/data/indicators/'
safe_fig_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/data/figures/'
safe_model_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/data/models/'
market = ['GSPC', 'DJI', 'Nasdaq', 'MSCIWORLD', 'Lyxor600' ]
stock_name = ['CCL', 'NEE', 'TSM', 'ISRG', 'CGC', 'BEP', 'ASML', 'RDS-B', 'APA', 'DAI']
start_date = date(2000, 1, 1)
end_date = date(2019, 4, 9)
# end_date = date.today()
forecast_time = 10
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
forcasted_date = end_date + timedelta(days=forecast_time)

# for stock in stock_name:
#     stock_data = ut.loadStockData(stock_data_path, stock) 
#     start_idx, end_idx, idx_forcasted, stock_dates, number_years = ut.findDate(stock_data, start_date, end_date, forcasted_date)
#     eva.evaluateAllIndicators(stock_data, stock_dates, start_idx, end_idx,  stock, safe_fig_path, True)
# total_return_percentage, annual_return = ut.getStockPerformance(apa, start_idx, end_idx, number_years, rounded=True)
# real_return = ut.calculateRealReturn(annual_return, start_date, end_date, number_years)


if os.path.exists('{}{}_indicators.csv'.format(indicator_path, 'APA')):
    apa = utils.loadStockData(indicator_path, 'APA')
else:
    apa = utils.loadStockData(stock_data_path, 'APA', indicators=False)
    apa = indicators.calculateIndicators(apa)
    utils.safeIndicators(apa, indicator_path, 'APA')

start_idx, end_idx, forcast_idx, stock_dates, number_years = utils.findDate(apa, start_date, end_date, forcasted_date)
apa_normalized = model_utils.normalizeIndicators(apa)
train_data, test_data, predict_data = model_utils.splitData(apa_normalized, start_idx, end_idx, forcast_idx)
X_train, Y_train, X_test, Y_test, X_predict, Y_predict = model_utils.getXY(train_data, test_data, predict_data)
# X_train, Y_train, X_test, Y_test, X_predict, Y_predict = model_utils.prepareDataforLTSM()

X_train = model_utils.prepareDataforLTSM(X_train)
Y_train = model_utils.prepareDataforLTSM(Y_train, Y_data=True)
X_test = model_utils.prepareDataforLTSM(X_test)
Y_test = model_utils.prepareDataforLTSM(Y_test, Y_data=True)
X_predict = model_utils.prepareDataforLTSM(X_predict, sample_length=len(X_predict))
Y_predict = model_utils.prepareDataforLTSM(Y_predict,  sample_length=len(X_predict), Y_data=True)

number_epochs = 15
batch_size = 1
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(500, input_shape=(X_train.shape[1],X_train.shape[2]), batch_size = batch_size, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dense(2,activation='softmax')
# ])



# model = model_utils.trainModel(model, X_train, Y_train, X_test, Y_test, batch_size, number_epochs)
# model_utils.safeModel(model, safe_model_path, number_epochs, batch_size)
model_name = 'lstm4_15_1'
model = model_utils.loadModel(safe_model_path, model_name)
predict_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(500, input_shape=(X_predict.shape[1],X_predict.shape[2]), batch_size = batch_size, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
    tf.keras.layers.Dense(2,activation='softmax')
])

# copy weights
old_weights = model.get_weights()
predict_model.set_weights(old_weights)
model_prediction = predict_model.predict(X_predict)
apa['prediction'] = np.full((len(apa['Adj Close']),1), -1)
apa.loc[(end_idx+1):forcast_idx, 'prediction'] = model_prediction[0,:,1]

print(model_prediction[0,:,1])
# fig = plt.figure(figsize=(15,12))   
# ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=3)
# ax0.plot_date(stock_dates[end_idx+1:forcast_idx],apa['Close'][end_idx+1:forcast_idx], color = 'blue', label = 'Close', linestyle ='-', markersize = 0)
# ax1 = ax0.twinx()
# ax1.plot_date(stock_dates[end_idx+1:forcast_idx],apa['daily_label'][end_idx+1:forcast_idx], color = 'red', label = 'Label', linestyle ='', marker = 'X')
# ax1.plot_date(stock_dates[end_idx+1:forcast_idx],apa['prediction'][end_idx+1:forcast_idx], color = 'green', label = 'Model', linestyle ='', marker = '*')
# ax1.set_title(r'Prediction of Stock Price')
# ax1.set_xlabel('Date')
# ax0.set_ylabel('$')
# plt.grid()
# lines, labels = ax0.get_legend_handles_labels()
# lines2, labels2 = ax1.get_legend_handles_labels()
# ax0.legend(lines + lines2, labels + labels2, loc=1)
# plt.show()

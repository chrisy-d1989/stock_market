# -*- coding: utf-8 -*-
"""
modul: 
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
forecast_time = 50
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
    apa = indicators.calculateIndicators(apa)
    utils.safeIndicators(apa, indicator_path, 'APA')

start_idx, end_idx, forcast_idx, stock_dates, number_years = utils.findDate(apa, start_date, end_date, forcasted_date)
apa_normalized = model_utils.normalizeIndicators(apa)
train_data, test_data, predict_data = model_utils.splitData(apa_normalized, start_idx, end_idx, forcast_idx)
X_train, Y_train = model_utils.getXY(train_data)
X_test, Y_test = model_utils.getXY(test_data)
X_predict, Y_predict = model_utils.getXY(predict_data)

# k = 3
# threshold= .5
# number_epochs = 11
# batch_size = 32

# input_shape = len(X_train[0])
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(50, activation='elu', input_shape=(input_shape,)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(50, activation='elu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])

# model.compile(
#   optimizer='adam',
#   loss='sparse_categorical_crossentropy',
#   metrics=['accuracy'],
# )

# model.fit(
#   X_train, # training data
#   Y_train, # training targets
#   epochs=number_epochs,
#   batch_size=batch_size,
# )

# model.evaluate(
#   X_test,
#   Y_test
# )

# model_utils.safeModel(model, safe_model_path, number_epochs, batch_size)
# model_name = 1
# model_utils.loadModel(safe_model_path, model_name, number_epochs, batch_size,)
# #print(model.predict(X_predict))
# print('DONE')

# # apa = ind.calculateIndicators(apa)
# ut.safeIndicators(apa, safe_path, stock_name[1])
# apa = ind.calculateBollingerBands(apa)











# apa.dropna(inplace=True)
# apa['Prediction'] = apa['Adj Close'].shift(-1)

# start_idx, end_idx, forecast, stock_dates = findDate(apa, start_date, end_date, forcasted_date)

# X = np.array(apa.drop(['Prediction'], 1))
# X = np.delete(X, 0, 1)
# Y = np.array(apa['Prediction'])
# Y = Y[start_idx:end_idx]
# X = preprocessing.scale(X[start_idx:end_idx])
# X_prediction = X[-forecast_time:]
# X_train, X_test, Y_train, Y_test =train_test_split(X[:-forecast_time], Y[:-forecast_time], test_size=0.5)
# stock_close = np.array(apa['Adj Close'])

# #Further Test Data
# Z = np.array(apa.drop(['Prediction'], 1))
# Z = np.delete(Z, 0, 1)
# Z = preprocessing.scale(Z[end_idx:end_idx + 200])

# if(np.sum(np.isnan(X_train))):
#     imp.fit(X_train)    
#     X_train = imp.transform(X_train)
# if(np.sum(np.isnan(Y_train))):
#     Y_train = np.where(np.isnan(Y_train), np.ma.array(Y_train, 
#                mask = np.isnan(Y_train)).mean(axis = 0), Y_train)  

# #Performing the Regression on the training data
# clf = LinearRegression()
# clf.fit(X_train, Y_train)
# prediction = clf.predict(X_prediction)
# prediction_Z = clf.predict(Z)
# score = clf.score(X_train, Y_train)
# print(score)

# # Calculate evaluation values
# difference_absolute = np.array([])
# difference_percentage = np.array([])
# for i in range(forecast_time):
#     difference_absolute = np.append(difference_absolute, abs(stock_close[end_idx-forecast_time+i] - prediction[i]))
#     difference_percentage = np.append(difference_percentage, ((stock_close[end_idx-forecast_time+i] - prediction[i])/stock_close[end_idx-forecast_time+i])*100)
  
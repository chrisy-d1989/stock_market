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
safe_model_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/data/models/stock/'
checkpoint_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/data/models/stock/checkpoints/'
market = ['GSPC', 'DJI', 'Nasdaq', 'MSCIWORLD', 'Lyxor600' ]
stock_name = ['CCL', 'NEE', 'TSM', 'ISRG', 'CGC', 'BEP', 'ASML', 'RDS-B', 'APA', 'DAI']
start_date = date(2010, 1, 6)
end_date = date(2018, 4, 9)
# end_date = date.today()
forecast_time = 10
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
forcasted_date = end_date + timedelta(days=forecast_time)
from_csv = False
# for stock in stock_name:
#     stock_data = ut.loadStockData(stock_data_path, stock) 
#     start_idx, end_idx, idx_forcasted, stock_dates, number_years = ut.findDate(stock_data, start_date, end_date, forcasted_date)
#     eva.evaluateAllIndicators(stock_data, stock_dates, start_idx, end_idx,  stock, safe_fig_path, True)
# total_return_percentage, annual_return = ut.getStockPerformance(apa, start_idx, end_idx, number_years, rounded=True)
# real_return = ut.calculateRealReturn(annual_return, start_date, end_date, number_years)

#make sure first symbol is the one you want to predict
symbols = ["APA", "ISRG", "BEP", 'ASML', 'RDS-B', 'CCL', 'NEE', 'TSM']
# symbols = ["APA", "ISRG"]
if os.path.exists('{}{}_indicators.csv'.format(indicator_path, 'alldata_normalized')):
    print('loading normalized indicator data ...')
    data_normalized = utils.loadStockData(indicator_path, 'alldata_normalized')
    print('... done loading normalized indicator data !!!')
else:
    if os.path.exists('{}{}_indicators.csv'.format(indicator_path, 'alldata')):
        print('loading all indicator data ...')
        data = utils.loadStockData(indicator_path, 'alldata')
        from_csv = True
        print('... done loading all indicator data !!!')
    else:
        print('Loading stock data ...')
        for i in range(len(symbols)):
            if os.path.exists('{}{}_indicators.csv'.format(indicator_path, symbols[i])):
                indicator = utils.loadStockData(indicator_path, symbols[i])
                indicator.columns = indicator.columns.map(lambda x : x+'_'+symbols[i] if x !='Date' else x)
            else:
                stock_data = utils.loadStockData(stock_data_path, symbols[i], indicators=False) 
                indicator = indicators.calculateIndicators(stock_data)
                indicator.columns = indicator.columns.map(lambda x : x+'_'+symbols[i] if x !='Date' else x)
                utils.safeIndicators(indicator, indicator_path, symbols[i])        
            if i == 0:
                start_idx, end_idx, forcast_idx, stock_dates, number_years = utils.findDate(indicator, start_date, end_date, \
                                                                                            forcasted_date)
                indicator.Date = pd.to_datetime(indicator.Date)
                data = indicator
            else:
                data = utils.combinePandasSeries(data, indicator, symbols[i])   
        utils.safeIndicators(data, indicator_path, 'alldata')
        print('... done loading stock data !!!')
    print('Normalizing indicator data ...')
    start_idx, end_idx, forcast_idx, stock_dates, number_years = utils.findDate(data, start_date, end_date, \
                                                                            forcasted_date, from_csv)
    data_normalized = model_utils.normalizeIndicators(data[start_idx:forcast_idx].reset_index())
    utils.safeIndicators(data_normalized, indicator_path, 'alldata_normalized')        
    print('... done normalizing indicator data !!!')

train_data, test_data, predict_data = model_utils.splitData(data_normalized, forecast_time, shuffle=False)
X_train, Y_train, X_test, Y_test, X_predict, Y_predict = model_utils.getXY(train_data, test_data, predict_data, symbols)
# X_train, Y_train, X_test, Y_test, X_predict, Y_predict = model_utils.prepareDataforLTSM()
X_train, batch_size = model_utils.prepareDataforLTSM(X_train)
Y_train, _ = model_utils.prepareDataforLTSM(Y_train, Y_data=True)
X_test, batch_size_test = model_utils.prepareDataforLTSM(X_test)
Y_test, _ = model_utils.prepareDataforLTSM(Y_test, Y_data=True)
X_predict, batch_size_predict = model_utils.prepareDataforLTSM(X_predict, sample_length=len(X_predict))
Y_predict, _ = model_utils.prepareDataforLTSM(Y_predict,  sample_length=len(X_predict), Y_data=True)

number_epochs = 2
# batch_size = 300
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), stateful=True, batch_size = batch_size,\
                         dropout=0.2, recurrent_dropout=0.1, return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(1028, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(512, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(256, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(128, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(64, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
    tf.keras.layers.Dense(2,activation='softmax')
])
# model, history = model_utils.trainModel(model, X_train, Y_train, X_test, Y_test, checkpoint_path, batch_size, number_epochs,\
#                                         monitor='loss', schedule=True, stopping=True, plateau=True, checkpoint=False)
model, history = model_utils.trainLSTMModel(model, X_train, Y_train, X_test, Y_test, checkpoint_path, batch_size, number_epochs,\
                                        monitor='loss', schedule=True, stopping=True, plateau=True, checkpoint=False)
model_utils.plotModelPerformance(history)

# model_utils.safeModel(model, safe_model_path, batch_size, history)
# val_loss, val_acc= model.evaluate(X_test, Y_test)
# print('val_loss:', val_loss)
# print('val_acc:', val_acc)
# model = model_utils.loadModel(safe_model_path, model_name)
# predict = model.predict(X_predict)

# fig = plt.figure(figsize=(15,12))   
# ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=3)
# ax0.plot_date(stock_dates[start_idx:end_idx],apa_normalized['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle ='-', markersize = 0)
# ax1 = ax0.twinx()
# ax1.plot_date(stock_dates[start_idx:end_idx],apa_normalized['Adj Close'][start_idx:end_idx], color = 'red', label = 'Label', linestyle ='-',markersize = 0)
# # ax1.plot_date(stock_dates[end_idx+1:forcast_idx],apa['prediction'][end_idx+1:forcast_idx], color = 'green', label = 'Model', linestyle ='', marker = '*')
# ax1.set_title(r'Prediction of Stock Price')
# ax1.set_xlabel('Date')
# ax0.set_ylabel('$')
# plt.grid()
# lines, labels = ax0.get_legend_handles_labels()
# lines2, labels2 = ax1.get_legend_handles_labels()
# ax0.legend(lines + lines2, labels + labels2, loc=1)
# plt.show()
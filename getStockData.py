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

#from selenium import webdriver #For Prediction
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

import indicators as ind
import utils as ut
import evaluationIndicators as eva



data_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/data/'
market = ['GSPC', 'DJI', 'Nasdaq', 'MSCIWORLD', 'Lyxor600' ]
stock_name = ['APA', 'BA']
  
start_date = date(2019, 1, 15)
end_date = date(2020, 3, 23)
# end_date = date.today()
forecast_time = 100
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
forcasted_date = end_date + timedelta(days=forecast_time)

apa = ut.readcsv(data_path, stock_name[1])
start_idx, end_idx, idx_forcasted, stock_dates, number_years = ut.findDate(apa, start_date, end_date, forcasted_date)
# total_return_percentage, annual_return = ut.getStockPerformance(apa, start_idx, end_idx, number_years, rounded=True)

apa = ind.calculateAverageDirectionalIndex(apa)
eva.evaluateADX(apa, stock_dates, start_idx, end_idx,  stock_name[1])
   
# fig1 = plt.figure(figsize=(15,12))
# plt.plot_date(stock_dates[start_idx:end_idx],apa['Adj Close'][start_idx:end_idx], color = 'red', label = 'Close', linestyle = '-', markersize = 0)
# plt.plot_date(stock_dates[start_idx:end_idx],apa['lower_bound'][start_idx:end_idx], color = 'blue', label = 'BBLow', linestyle = '-', markersize = 0)
# plt.plot_date(stock_dates[start_idx:end_idx],apa['upper_bound'][start_idx:end_idx], color = 'blue', label = 'BBHigh', linestyle = '-', markersize = 0)
# plt.title(r'APA Stock Prediction')
# plt.xlabel('Date')
# plt.ylabel('$')
# plt.legend()

# fig2 = plt.figure(figsize=(15,12))
# #plt.plot_date(stock_dates[start:end],apa['High'][start:end], label = 'High')
# #plt.plot_date(stock_dates[start:end],apa['Low'][start:end], label = 'Low')
# plt.plot_date(stock_dates[end_idx-forecast_time:end_idx],difference_absolute, label = r'absolute', linestyle = '-', markersize = 0)
# plt.plot_date(stock_dates[end_idx-forecast_time:end_idx],difference_percentage, label = r'%', linestyle = '-', markersize = 0)
# plt.title(r'APA Stock Prediction')
# plt.xlabel('Date')
# plt.ylabel('$')
# plt.legend()
# plt.show()


# print(apa.loc[600])


# apa = readcsv(data_path, 'APA')
# aal = readcsv(data_path, 'AAL_dt')
# start_idx, end_idx, forecast, stock_dates, number_yrs = findDate(apa, start_date, end_date, forcasted_date)
# # apa.columns = ['date', 'open', 'high', 'low', 'Adj Close', 'Adj Close', 'volume']
# # apa.insert(0, '', apa.reset_index())
# print(apa.head(2))
# print(aal.head(2))
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
  
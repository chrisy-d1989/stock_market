# -*- coding: utf-8 -*-
"""

modul: help functions
modul author: Christoph Doerr

"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from sklearn import preprocessing
import matplotlib.pyplot as plt

def loadStockData(stock_data_path, stock):
    """ 
    Input data_path: path to folder that holds stock data
    Input stock: stock symbol
    Return stock_data: pandas series with stock data
    """
    stock_data = pd.read_csv(stock_data_path +  stock + '.csv')
    return stock_data

def findDate(stock, start_date, end_date, forcasted_date):
    """ 
    Input data_path: path to folder that holds stock data
    Input stock: stock symbol
    Return stock_data: pandas series with stock data
    """
    dates = np.array(stock['Date'])
    stock_dates = np.array([])
    idx_start = 0   
    idx_end = 0
    idx_forcasted = 0
    while idx_start == 0 or idx_end == 0:
        for i in range(len(dates)):
            date = datetime.strptime(dates[i], '%Y-%m-%d').date()
            if(date == start_date):
                idx_start = i
            if(date == end_date):
                idx_end = i
            if(date == forcasted_date):
                idx_forcasted = i
        if idx_start == 0:
            start_date = start_date + timedelta(days=1)
        if idx_end == 0:
            end_date = end_date + timedelta(days=1)
             
    stock_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
    number_years = int(-(start_date - end_date).days / 365)
    return (idx_start, idx_end, idx_forcasted, stock_dates, number_years)

def annual_return(total_return, number_years, rounded=True):
    """ 
    Input data_path: path to folder that holds stock data
    Input stock: stock symbol
    Return stock_data: pandas series with stock data
    """
    num = ((total_return ** (1/number_years)) - 1) * 100
    return np.round(num, decimals=2) if rounded else num

def getStockPerformance(stock, start_idx, end_idx, number_years, rounded=True):
    """ 
    Function to calculate the performance of a stock over a given time frame
    Input stock: pandas series with stock data holding Adj Close column
    Input start_idx: start index of evaluation
    Input end_idx: end index of evaluation
    Input number_years: number of years of evaluation
    Input rounded: round annual return
    Return total_return_percentage: pandas series with stock data
    Return annual return: pandas series with stock data
    """
    total_return = (stock['Adj Close'][end_idx]- stock['Adj Close'][start_idx])/stock['Adj Close'][start_idx] + 1
    total_return_percentage = total_return * 100
    num = ((total_return ** (1/number_years)) - 1) * 100
    annual_return = np.round(num, decimals=2) if rounded else num
    return (total_return_percentage, annual_return)

def getMarketPerformance(path, market, start_idx, end_idx, number_years):
    """ 
    Function to calculate the performance of the market over a given time frame
    Input stock: pandas series with stock data holding Adj Close column
    Input start_idx: start index of evaluation
    Input end_idx: end index of evaluation
    Input number_years: number of years of evaluation
    Return market_peformance: pandas series holding percentage of market retunr annual market return
    """
    return_market = np.array([])
    return_market_percentage = np.array([])
    anual_return_market = np.array([])
    for index in market:
        index = loadStockData(path, index)
        total_return = (index['Adj Close'][end_idx]- index['Adj Close'][start_idx])/index['Adj Close'][start_idx] + 1
        return_market = np.append(return_market, total_return) 
        return_market_percentage = np.append(return_market_percentage, total_return * 100)
        anual_return_market = np.append(anual_return_market, annual_return(total_return, number_years, rounded=True))
    
    market_performance = pd.DataFrame([return_market_percentage, anual_return_market], columns=market, index = ['return_percentage', 'anual_return'])
    return (market_performance)

def calculateRealReturn(annual_return, start_date, end_date, number_years):
    real_return = np.array([])
    inflation_history = pd.DataFrame(np.transpose(np.array([5.0, 4.5, 2.6, 1.8, 1.3, 2.0, 0.9, 0.6, 1.4, 2.0, 1.3, 1.1, 1.7, 1.5, 1.6, 2.3, 2.6, 0.3, 1.1, 2.1, 2.0, 1.4, 1.0, 0.5, 0.5, 1.5, 1.8, 1.4, 0.4])), index = np.transpose(np.arange(1991, 2020, 1)), columns = ['inflation_rate'])
    for i in range(0, number_years):
        real_return = np.append(real_return, ((1 + annual_return/100)/(1 + inflation_history.loc[start_date.year + i]/100)))
    return real_return

def safeIndicators(stock, safe_path, stock_name):
    """ 
    Function to safe the calculated indicators into a csv file
    Input stock: pandas series with stock data holding Adj Close column
    Input safe_path: path to indicator folder
    Input stock_name: symbol of the stock
    """
    stock.to_csv('{}{}_indicators.csv'.format(safe_path, stock_name), index = False)

def loadIndicators(indicator_path, stock_name):
    """ 
    Function to safe the calculated indicators into a csv file
    Input stock: pandas series with stock data holding Adj Close column
    Input safe_path: path to indicator folder
    """
    stock_data = pd.read_csv(indicator_path +  stock_name + '_indicators.csv')
    return stock_data

def standardizeIndicators(stock):
    for key, value in stock.iteritems():
        if key == 'Date':
            continue
        elif (key == 'daily_label'):
                mean = 0
                std = 1
        else:
            mean = value.mean()
            std = value.std()
        stock.loc[:,key] = (stock.loc[:, key] - mean) / std
    return (stock)

def normalizeIndicators(stock):
    for key, value in stock.iteritems():
        if key == 'Date':
            continue
        else:
            x = stock[[key]].values.astype(float)
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            stock[key] = pd.DataFrame(x_scaled)
    return (stock)

def saveFigure(safe_fig_path, stock_name, indicator):
    plt.savefig(safe_fig_path + stock_name + '_' + indicator + '.png')
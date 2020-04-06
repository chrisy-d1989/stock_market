# -*- coding: utf-8 -*-
"""

modul: help functions
modul author: Christoph Doerr

"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

def readcsv(data_path, stock):
    """ 
    Input data_path: path to folder that holds stock data
    Input stock: stock symbol
    Return stock_data: pandas series with stock data
    """
    stock_data = pd.read_csv(data_path +  stock + '.csv')
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

def getMarketPerformance(path, market, start_date, forcasted_date, end_date):
    return_market = np.array([])
    return_market_percentage = np.array([])
    anual_return_market = np.array([])
    for index in market:
        index = readcsv(path, index)
        start_idx, end_idx, forecast, stock_dates, number_years = findDate(index, start_date, end_date, forcasted_date)
        total_return = (index['Adj Close'][end_idx]- index['Adj Close'][start_idx])/index['Adj Close'][start_idx] + 1
        return_market = np.append(return_market, total_return) 
        return_market_percentage = np.append(return_market_percentage, total_return * 100)
        anual_return_market = np.append(anual_return_market, annual_return(total_return, number_years, rounded=True))
    
    market_performance = pd.DataFrame([return_market_percentage, anual_return_market], columns=market, index = ['return_percentage', 'anual_return'])
    return (market_performance)


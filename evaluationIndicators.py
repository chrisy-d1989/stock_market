# -*- coding: utf-8 -*-
"""

modul: evaluation of indicators
modul author: Christoph Doerr

"""
import matplotlib.pyplot as plt


def evaluateBollingerBands(stock, stock_dates, start_idx, end_idx, stock_name):   
    for idx, row in stock.iterrows():
        if stock['Adj Close'].iloc[idx] <= stock['lower_bound'].iloc[idx]:
            recent_low = idx
        if stock['Adj Close'].iloc[idx] >= stock['upper_bound'].iloc[idx]:
            recent_high = idx   
    # print(stock['Date'].iloc[recent_low], stock['Date'].iloc[recent_high])
    fig = plt.figure(figsize=(15,12))
    plt.plot_date(stock_dates[start_idx:end_idx],stock['Adj Close'][start_idx:end_idx], color = 'red', label = 'Close', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['moving_avg_20'][start_idx:end_idx], color = 'green', label = 'Close', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['lower_bound'][start_idx:end_idx], color = 'blue', label = 'BBLow', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['upper_bound'][start_idx:end_idx], color = 'blue', label = 'BBHigh', linestyle = '-', markersize = 0)
    plt.title(r'{} Stock over Time'.format(stock_name))
    plt.xlabel('Date')
    plt.ylabel('$')
    plt.grid()
    plt.legend()    
    return(recent_low, recent_high)

def evaluateTrix(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Adj Close'][start_idx:end_idx], color = 'red', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['Trix_14'][start_idx:end_idx], color = 'red', label = 'Trix', linestyle = '-', markersize = 0)
    ax0.set_title(r'APA over Time')
    ax1.set_title(r'{} Trix Indicator'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')
    plt.grid()
    plt.legend()
    
def evaluateIchimikuIndex(stock, stock_dates, start_idx, end_idx, stock_name):
    fig,ax = plt.subplots(1,1,sharex=True,figsize = (20,9))
    ax.plot_date(stock_dates[start_idx:end_idx],stock['Adj Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax.plot_date(stock_dates[start_idx:end_idx],stock['senkou_span_a'][start_idx:end_idx], color = 'green', label = 'SpanA', linestyle = '-', markersize = 0)
    ax.plot_date(stock_dates[start_idx:end_idx],stock['senkou_span_b'][start_idx:end_idx], color = 'red', label = 'SpanB', linestyle = '-', markersize = 0) 
    # use the fill_between call of ax object to specify where to fill the chosen color
    # pay attention to the conditions specified in the fill_between call
    ax.fill_between(stock_dates[start_idx:end_idx],stock['senkou_span_a'][start_idx:end_idx],stock['senkou_span_b'][start_idx:end_idx],where = stock['senkou_span_a'][start_idx:end_idx] >= stock['senkou_span_a'][start_idx:end_idx], color = 'lightgreen')
    ax.fill_between(stock_dates[start_idx:end_idx],stock['senkou_span_a'][start_idx:end_idx],stock['senkou_span_b'][start_idx:end_idx],where = stock['senkou_span_a'][start_idx:end_idx] < stock['senkou_span_b'][start_idx:end_idx], color = 'lightcoral')
    plt.legend(loc=0)
    plt.grid()
    plt.show()

def evaluateWilliamsR(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Adj Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['williamsR'][start_idx:end_idx], color = 'blue', label = 'williamsR', linestyle = '-', markersize = 0)
    ax1.axhline(y=-20, color='red', alpha=0.3)
    ax1.axhline(y=-80, color='red', alpha=0.3)
    ax0.set_title(r'APA over Time')
    ax1.set_title(r'WilliamsR Indicator'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    # place a text box in upper left in axes coords
    textstr = ['Overbought', 'Oversold']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.95, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.grid()
    plt.legend()
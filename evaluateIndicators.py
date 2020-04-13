# -*- coding: utf-8 -*-
"""

modul: evaluation of indicators
modul author: Christoph Doerr

"""

import matplotlib.pyplot as plt
import numpy as np
import calculateIndicators as ind
import utils as ut

""" Indicators showing the trend of a stock """
def evaluateTrix(stock, stock_dates, start_idx, end_idx, stock_name):  
    stock = ind.calculateTrix(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Adj Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['Trix_14'][start_idx:end_idx], color = 'green', label = 'Trix', linestyle = '-', markersize = 0)
    ax0.set_title(r'{} Trix Indicator over time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax0.grid()
    ax1.grid()
    ax0.legend()
    ax1.legend()
    
def evaluateIchimikuIndex(stock, stock_dates, start_idx, end_idx, stock_name):
    stock = ind.calculateIchimokuIndex(stock)
    fig,ax = plt.subplots(1,1,sharex=True,figsize = (20,9))
    ax.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax.plot_date(stock_dates[start_idx:end_idx],stock['senkou_span_a'][start_idx:end_idx], color = 'green', label = 'SpanA', linestyle = '-', markersize = 0)
    ax.plot_date(stock_dates[start_idx:end_idx],stock['senkou_span_b'][start_idx:end_idx], color = 'red', label = 'SpanB', linestyle = '-', markersize = 0) 
    ax.fill_between(stock_dates[start_idx:end_idx],stock['senkou_span_a'][start_idx:end_idx],stock['senkou_span_b'][start_idx:end_idx],where = stock['senkou_span_a'][start_idx:end_idx] >= stock['senkou_span_a'][start_idx:end_idx], color = 'lightgreen')
    ax.fill_between(stock_dates[start_idx:end_idx],stock['senkou_span_a'][start_idx:end_idx],stock['senkou_span_b'][start_idx:end_idx],where = stock['senkou_span_a'][start_idx:end_idx] < stock['senkou_span_b'][start_idx:end_idx], color = 'lightcoral')
    ax2 = ax.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax.legend()
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc=1)
    ax.set_title(r'{} Ichimuku Index over Time'.format(stock_name))
    ax.set_xlabel('Date')
    ax.set_ylabel('$')

def evaluateKST(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateKST(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['signal_KST'][start_idx:end_idx], color = 'green', label = 'KST', linestyle = '-', markersize = 0)
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_title(r' {} KST Indicator over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    ax0.grid()
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    # place a text box in upper left in axes coords
    textstr = ['Crossover']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.5, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.grid()

def evaluateDPO(stock, stock_dates, start_idx, end_idx, stock_name):
    stock = ind.calculateDetrendedPriceOscillator(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['detrended_price_oscillator'][start_idx:end_idx], color = 'blue', label = 'DPO', linestyle = '-', markersize = 0)
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_title(r'{} Detrended Price Oscillator over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()    
    
def evaluateCCI(stock, stock_dates, start_idx, end_idx, stock_name):  
    stock = ind.calculateCommodityChannelIndex(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['CCI'][start_idx:end_idx], color = 'blue', label = 'CCI', linestyle = '-', markersize = 0)
    ax1.axhline(y=-100, color='green')
    ax1.axhline(y=100, color='green')
    ax1.axhspan(-100, 100, alpha=0.5, color='lightgreen')
    ax0.set_title(r'{} Commodity Channel Index over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()    
    
def evaluateADX(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateAverageDirectionalIndex(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=4)
    ax1 = plt.subplot2grid((10, 1), (4, 0), rowspan=6)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['adx'][start_idx:end_idx], color = 'k', label = 'ADX', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['-dm'][start_idx:end_idx], color = 'red', label = '-DM', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['+dm'][start_idx:end_idx], color = 'green', label = '+DM', linestyle = '-', markersize = 0)
    ax1.axhline(y=25, color='green', alpha=0.3)
    ax1.axhline(y=20, color='red', alpha=0.3)
    ax1.set_title(r'{} ADX Indicator over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    ax0.grid()
    ax1.grid()
    # place a text box in upper left in axes coords
    textstr = ['Strong', 'Weak']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax0.text(0.01, 0.8, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax0.text(0.01, 0.25, textstr[1], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    
def evaluateArronIndicator(stock, stock_dates, start_idx, end_idx, stock_name): 
    stock = ind.calculateAroonIndicator(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['aroon_down'][start_idx:end_idx], color = 'green', label = 'down', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['aroon_up'][start_idx:end_idx], color = 'magenta', label = 'up', linestyle = '-', markersize = 0)
    ax0.set_title(r'{} Aroon Indicator over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()    

""" Indicators caluclating the Volume Trend of the Stock """
def evaluateMoneyFlowVolume(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateMoneyFlowVolume(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['Money_Flow_Volume'][start_idx:end_idx], color = 'green', label = 'Money Flow Index', linestyle = '-', markersize = 0)
    ax1.fill_between(stock_dates[start_idx:end_idx],stock['Money_Flow_Volume'][start_idx:end_idx],np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])),where = stock['Money_Flow_Volume'][start_idx:end_idx] >= np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])), color = 'lightgreen')
    ax1.fill_between(stock_dates[start_idx:end_idx],stock['Money_Flow_Volume'][start_idx:end_idx],np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])),where = stock['Money_Flow_Volume'][start_idx:end_idx] <= np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])), color = 'tomato')
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_title(r'{} Money Flow Index over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()      

    
def evaluateForceIndex(stock, stock_dates, start_idx, end_idx, stock_name): 
    stock = ind.calculateForceIndex(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['ForceIndex'][start_idx:end_idx], color = 'green', label = 'Force Index', linestyle = '-', markersize = 0)
    ax0.set_title(r'{} Force Index over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()     
    
def evaluateEaseofMovement(stock, stock_dates, start_idx, end_idx, stock_name): 
    stock = ind.calculateEaseOfMovement(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['ease_movement'][start_idx:end_idx], color = 'green', label = 'EOM', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['sma_ease_movement'][start_idx:end_idx], color = 'gold', label = 'SMA_EOM', linestyle = '-', markersize = 0)
    ax0.set_title(r'{} Ease of Movement Trend over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()       
    
def evaluateVolumePriceTrend(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateVolumePriceTrend(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['volume_price_trend'][start_idx:end_idx], color = 'gold', label = 'vpt', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['vpt_signal'][start_idx:end_idx], color = 'green', label = 'vpt_ma', linestyle = '-', markersize = 0)
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_title(r'{} Volume Price Trend over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()        

""" Indicators calculating the volatility of the stock"""
def evaluateAverageTrueRange(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateAverageTrueRange(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
  
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['avg_true_range'][start_idx:end_idx], color = 'green', label = 'ATR', linestyle = '-', markersize = 0)
    ax0.set_title(r'{} Average True Range Indicator ofver time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')  
    ax0.grid()
    ax0.grid()
    ax0.legend()
    ax1.legend()

def evaluateBollingerBands(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateMovingAverages(stock)
    stock = ind.calculateBollingerBands(stock)
    for idx, row in stock.iterrows():
        if stock['Adj Close'].iloc[idx] <= stock['lower_bound'].iloc[idx]:
            recent_low = idx
        if stock['Adj Close'].iloc[idx] >= stock['upper_bound'].iloc[idx]:
            recent_high = idx   
    fig = plt.figure(figsize=(15,12))
    plt.plot_date(stock_dates[start_idx:end_idx],stock['Adj Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['moving_avg_20'][start_idx:end_idx], color = 'green', label = 'MA_20', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['lower_bound'][start_idx:end_idx], color = 'lightgreen', label = 'BBLow', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['upper_bound'][start_idx:end_idx], color = 'lightgreen', label = 'BBHigh', linestyle = '-', markersize = 0)
    plt.title(r'{} Stock over Time with Bollinger Bands'.format(stock_name))
    plt.xlabel('Date')
    plt.ylabel('$')
    plt.grid()
    plt.legend()    
    return(recent_low, recent_high)

def evaluateKeltnerChannel(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateKeltnerChannel(stock)
    for idx, row in stock.iterrows():
        if stock['Adj Close'].iloc[idx] <= stock['keltner_channelL'].iloc[idx]:
            recent_low = idx
        if stock['Adj Close'].iloc[idx] >= stock['keltner_channelL'].iloc[idx]:
            recent_high = idx   
    fig = plt.figure(figsize=(15,12))
    plt.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['keltner_channelM'][start_idx:end_idx], color = 'darkgreen', label = 'EMA_20', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['keltner_channelL'][start_idx:end_idx], color = 'lightgreen', label = 'KCLow', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['keltner_channelU'][start_idx:end_idx], color = 'lightgreen', label = 'KCHigh', linestyle = '-', markersize = 0)
    plt.fill_between(stock_dates[start_idx:end_idx],stock['keltner_channelL'][start_idx:end_idx],stock['keltner_channelU'][start_idx:end_idx],where =stock['keltner_channelU'][start_idx:end_idx] >= stock['keltner_channelL'][start_idx:end_idx], color = 'lightgreen')
    plt.title(r'{} Stock over Time with Keltner Channel'.format(stock_name))
    plt.xlabel('Date')
    plt.ylabel('$')
    plt.grid()
    plt.legend()    
    return(recent_low, recent_high)

    
""" Indicators calculating the Momentum of the Stock"""    
def evaluateRelativeStrengthIndex(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateRelativeStrengthIndex(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['RSI'][start_idx:end_idx], color = 'green', label = 'RSI', linestyle = '-', markersize = 0)
    ax1.axhline(y=30, color='green', alpha=0.3)
    ax1.axhline(y=70, color='red', alpha=0.3)
    ax0.set_title(r'{} Relative Strength Indicator over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    ax0.grid()
    # place a text box in upper left in axes coords
    textstr = ['Overbought', 'Oversold']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.90, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.text(0.01, 0.25, textstr[1], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()  

def evaluateStochasticK(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateStochasticK(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())   
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['%K'][start_idx:end_idx], color = 'green', label = '%K', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['%D'][start_idx:end_idx], color = 'gold', label = '%D', linestyle = '-', markersize = 0)
    ax1.axhline(y=20, color='green', alpha=0.3)
    ax1.axhline(y=80, color='red', alpha=0.3)
    ax0.set_title(r'{} Stochastc %K/D Indicator over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    ax0.grid()
    # place a text box in upper left in axes coords
    textstr = ['Overbought', 'Oversold']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.8, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.text(0.01, 0.25, textstr[1], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()  
    
def evaluateWilliamsR(stock, stock_dates, start_idx, end_idx, stock_name):   
    stock = ind.calculateWilliamsR(stock)
    fig = plt.figure(figsize=(15,12))
    fig.subplots_adjust(hspace=0)
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (6, 0), rowspan=4)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Adj Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax2 = ax0.twinx()
    ax2.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax2.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax2.set_ylim(0, 1*stock['Volume'].max())   
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['williamsR'][start_idx:end_idx], color = 'green', label = 'williamsR', linestyle = '-', markersize = 0)
    ax1.axhline(y=-20, color='red', alpha=0.3)
    ax1.axhline(y=-80, color='green', alpha=0.3)
    ax0.set_title(r'{} WilliamsR Indicator over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    # place a text box in upper left in axes coords
    textstr = ['Overbought', 'Oversold']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.8, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.text(0.01, 0.25, textstr[1], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    ax1.legend()
    ax0.grid()
    ax1.grid()  

def evaluateAllIndicators(stock, stock_dates, start_idx, end_idx, stock_name, safe_fig_path, safe_fig = False):
    stock = ind.calculateIndicatorsforEvaluation(stock)
    fig1 = plt.figure(figsize=(15,12), facecolor='black')
    plt.style.use('dark_background')
    fig1.subplots_adjust(hspace=0)
    ax_RSI = plt.subplot2grid((14, 1), (0, 0), rowspan=4)
    ax_close = plt.subplot2grid((14, 1), (4, 0), rowspan=6)
    ax_CCI = plt.subplot2grid((14, 1), (10, 0), rowspan=4)  
    vol_factor = 4
   
    #Plot Close and Bollinger Bands
    ax_close.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax_close.plot_date(stock_dates[start_idx:end_idx],stock['moving_avg_20'][start_idx:end_idx], color = 'green', label = 'Moving Average 20', linestyle = '-', markersize = 0)
    ax_close.plot_date(stock_dates[start_idx:end_idx],stock['lower_bound'][start_idx:end_idx], color = 'lightgreen', label = 'BBLow', linestyle = '-', markersize = 0)
    ax_close.plot_date(stock_dates[start_idx:end_idx],stock['upper_bound'][start_idx:end_idx], color = 'lightgreen', label = 'BBHigh', linestyle = '-', markersize = 0)
    ax_close.fill_between(stock_dates[start_idx:end_idx],stock['lower_bound'][start_idx:end_idx],stock['upper_bound'][start_idx:end_idx],where =stock['upper_bound'][start_idx:end_idx] >= stock['lower_bound'][start_idx:end_idx], color = 'lightgreen', alpha=0.5)
    ax_vol = ax_close.twinx()
    ax_vol.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax_vol.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax_vol.set_ylim(0, vol_factor*stock['Volume'].max())    
    #Plot RSI
    ax_RSI.plot_date(stock_dates[start_idx:end_idx],stock['RSI'][start_idx:end_idx], color = 'green', label = 'RSI', linestyle = '-', markersize = 0)
    ax_RSI.axhline(y=30, color='green', alpha=1.0)
    ax_RSI.axhline(y=70, color='red', alpha=1.0)
    # place a text box in upper left in axes coords
    textstr = ['Overbought', 'Oversold']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_RSI.text(0.01, 0.90, textstr[0], transform=ax_RSI.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax_RSI.text(0.01, 0.25, textstr[1], transform=ax_RSI.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    #Plot CCI
    ax_CCI.plot_date(stock_dates[start_idx:end_idx],stock['CCI'][start_idx:end_idx], color = 'darkgreen', label = 'CCI', linestyle = '-', markersize = 0)
    ax_CCI.axhline(y=-100, color='lightgreen')
    ax_CCI.axhline(y=100, color='lightgreen')
    ax_CCI.axhspan(-100, 100, alpha=0.5, color='lightgreen')  
    ax_RSI.set_title(r'{} Relative Strength, Bollinger Bands, Commodity Channel Indicators over Time'.format(stock_name))
    ax_close.set_ylabel('$')    
    lines, labels = ax_close.get_legend_handles_labels()
    lines2, labels2 = ax_vol.get_legend_handles_labels()
    ax_close.legend(lines + lines2, labels + labels2, loc='northwest')
    ax_CCI.legend()
    ax_RSI.legend()
    ax_RSI.grid()
    ax_close.grid() 
    ax_CCI.grid() 
    ax_close.tick_params(axis='y', colors='blue')
    ax_vol.tick_params(axis='y', colors = 'gold')
    
    ut.saveFigure(safe_fig_path, stock_name, 'BollingerBands')
    
    fig2 = plt.figure(figsize=(15,12), facecolor='black')
    plt.style.use('dark_background')
    fig2.subplots_adjust(hspace=0)
    ax_WR = plt.subplot2grid((14, 1), (0, 0), rowspan=4)
    ax_close = plt.subplot2grid((14, 1), (4, 0), rowspan=6)
    ax_MF = plt.subplot2grid((14, 1), (10, 0), rowspan=4)  
    
    #Plot Close and Ichimuku
    ax_close.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax_close.plot_date(stock_dates[start_idx:end_idx],stock['senkou_span_a'][start_idx:end_idx], color = 'green', label = 'SpanA', linestyle = '-', markersize = 0)
    ax_close.plot_date(stock_dates[start_idx:end_idx],stock['senkou_span_b'][start_idx:end_idx], color = 'red', label = 'SpanB', linestyle = '-', markersize = 0) 
    ax_close.fill_between(stock_dates[start_idx:end_idx],stock['senkou_span_a'][start_idx:end_idx],stock['senkou_span_b'][start_idx:end_idx],where = stock['senkou_span_a'][start_idx:end_idx] >= stock['senkou_span_a'][start_idx:end_idx], color = 'lightgreen')
    ax_close.fill_between(stock_dates[start_idx:end_idx],stock['senkou_span_a'][start_idx:end_idx],stock['senkou_span_b'][start_idx:end_idx],where = stock['senkou_span_a'][start_idx:end_idx] < stock['senkou_span_b'][start_idx:end_idx], color = 'lightcoral')
    ax_vol = ax_close.twinx()
    ax_vol.plot_date(stock_dates[start_idx:end_idx],stock['Volume'][start_idx:end_idx], color = 'orange', alpha = 0.4, label = 'Volume', linestyle = '-', markersize = 0)
    ax_vol.fill_between(stock_dates[start_idx:end_idx], np.zeros(len(stock['Volume'][start_idx:end_idx])), stock['Volume'][start_idx:end_idx], color='orange', alpha=.4)
    ax_vol.set_ylim(0, vol_factor*stock['Volume'].max())    
    #Plot Williams R
    ax_WR.plot_date(stock_dates[start_idx:end_idx],stock['williamsR'][start_idx:end_idx], color = 'green', label = 'williamsR', linestyle = '-', markersize = 0)
    ax_WR.axhline(y=-20, color='red', alpha=1.0)
    ax_WR.axhline(y=-80, color='green', alpha=1.0)
    # place a text box in upper left in axes coords
    textstr = ['Overbought', 'Oversold']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_WR.text(0.01, 0.8, textstr[0], transform=ax_WR.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax_WR.text(0.01, 0.25, textstr[1], transform=ax_WR.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    #Plot Money Flow
    ax_MF.plot_date(stock_dates[start_idx:end_idx],stock['Money_Flow_Volume'][start_idx:end_idx], color = 'green', label = 'Money Flow Index', linestyle = '-', markersize = 0)
    ax_MF.fill_between(stock_dates[start_idx:end_idx],stock['Money_Flow_Volume'][start_idx:end_idx],np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])),where = stock['Money_Flow_Volume'][start_idx:end_idx] >= np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])), color = 'lightgreen')
    ax_MF.fill_between(stock_dates[start_idx:end_idx],stock['Money_Flow_Volume'][start_idx:end_idx],np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])),where = stock['Money_Flow_Volume'][start_idx:end_idx] <= np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])), color = 'tomato')
    ax_MF.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax_close.set_ylabel('$')    
    ax_WR.set_title(r'{} WilliamsR, Ichimuku, Money Flow Indicator over Time'.format(stock_name)) 
    lines, labels = ax_close.get_legend_handles_labels()
    lines2, labels2 = ax_vol.get_legend_handles_labels()
    ax_close.legend(lines + lines2, labels + labels2, loc='northwest')
    ax_MF.legend()
    ax_WR.legend()
    ax_WR.grid()
    ax_close.grid() 
    ax_MF.grid() 
    ax_close.tick_params(axis='y', colors='blue')
    ax_vol.tick_params(axis='y', colors = 'gold')
    
    ut.saveFigure(safe_fig_path, stock_name, 'Ichimoku')
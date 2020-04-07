# -*- coding: utf-8 -*-
"""

modul: evaluation of indicators
modul author: Christoph Doerr

"""
import matplotlib.pyplot as plt
import numpy as np

def evaluateBollingerBands(stock, stock_dates, start_idx, end_idx, stock_name):   
    for idx, row in stock.iterrows():
        if stock['Adj Close'].iloc[idx] <= stock['lower_bound'].iloc[idx]:
            recent_low = idx
        if stock['Adj Close'].iloc[idx] >= stock['upper_bound'].iloc[idx]:
            recent_high = idx   
    fig = plt.figure(figsize=(15,12))
    plt.plot_date(stock_dates[start_idx:end_idx],stock['Adj Close'][start_idx:end_idx], color = 'red', label = 'Close', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['moving_avg_20'][start_idx:end_idx], color = 'green', label = 'MA_20', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['lower_bound'][start_idx:end_idx], color = 'blue', label = 'BBLow', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['upper_bound'][start_idx:end_idx], color = 'blue', label = 'BBHigh', linestyle = '-', markersize = 0)
    plt.title(r'{} Stock over Time'.format(stock_name))
    plt.xlabel('Date')
    plt.ylabel('$')
    plt.grid()
    plt.legend()    
    return(recent_low, recent_high)

def evaluateKeltnerChannel(stock, stock_dates, start_idx, end_idx, stock_name):   
    for idx, row in stock.iterrows():
        if stock['Adj Close'].iloc[idx] <= stock['keltner_channelL'].iloc[idx]:
            recent_low = idx
        if stock['Adj Close'].iloc[idx] >= stock['keltner_channelL'].iloc[idx]:
            recent_high = idx   
    print(stock['Date'].iloc[recent_low], stock['Date'].iloc[recent_high])
    fig = plt.figure(figsize=(15,12))
    plt.plot_date(stock_dates[start_idx:end_idx],stock['Adj Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['keltner_channelM'][start_idx:end_idx], color = 'darkgreen', label = 'EMA_20', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['keltner_channelL'][start_idx:end_idx], color = 'lightgreen', label = 'KCLow', linestyle = '-', markersize = 0)
    plt.plot_date(stock_dates[start_idx:end_idx],stock['keltner_channelU'][start_idx:end_idx], color = 'lightgreen', label = 'KCHigh', linestyle = '-', markersize = 0)
    plt.fill_between(stock_dates[start_idx:end_idx],stock['keltner_channelL'][start_idx:end_idx],stock['keltner_channelU'][start_idx:end_idx],where =stock['keltner_channelU'][start_idx:end_idx] >= stock['keltner_channelL'][start_idx:end_idx], color = 'lightgreen')
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
    ax1.axhline(y=-80, color='green', alpha=0.3)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'WilliamsR Indicator'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    # place a text box in upper left in axes coords
    textstr = ['Overbought', 'Oversold']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.8, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.text(0.01, 0.25, textstr[1], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax0.grid()
    ax1.grid()
    plt.legend()
    
def evaluateStochasticK(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['%K'][start_idx:end_idx], color = 'blue', label = '%K', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['%D'][start_idx:end_idx], color = 'magenta', label = '%D', linestyle = '-', markersize = 0)
    ax1.axhline(y=20, color='green', alpha=0.3)
    ax1.axhline(y=80, color='red', alpha=0.3)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'Stochastc %K/D Indicator')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    ax0.grid()
    # place a text box in upper left in axes coords
    textstr = ['Overbought', 'Oversold']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.8, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.text(0.01, 0.25, textstr[1], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.grid()
    plt.legend()
    
def evaluateAverageTrueRange(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot(1,1,1)  
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1 = ax0.twinx()
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['avg_true_range'][start_idx:end_idx], color = 'green', label = 'ATR', linestyle = '-', markersize = 0)
    ax0.set_title(r'Average True Range Indicator {} over Time'.format(stock_name))
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')  
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    ax0.tick_params(axis='y', colors='blue')
    ax1.tick_params(axis='y', colors = 'green')   

def evaluateKST(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['signal_KST'][start_idx:end_idx], color = 'blue', label = 'KST', linestyle = '-', markersize = 0)
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'KST Indicator')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    ax0.grid()
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    # place a text box in upper left in axes coords
    textstr = ['Crossover']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.5, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.grid()

def evaluateDPO(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['detrended_price_oscillator'][start_idx:end_idx], color = 'blue', label = 'DPO', linestyle = '-', markersize = 0)
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'Detrended Price Oscillator')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    ax0.grid()
    ax1.grid()    
    
def evaluateCCI(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['CCI'][start_idx:end_idx], color = 'blue', label = 'CCI', linestyle = '-', markersize = 0)
    ax1.axhline(y=-100, color='green')
    ax1.axhline(y=100, color='green')
    ax1.axhspan(-100, 100, alpha=0.5, color='lightgreen')
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'Commodity Channel Index')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    ax0.grid()
    ax1.grid()

def evaluateArronIndicator(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['aroon_down'][start_idx:end_idx], color = 'green', label = 'down', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['aroon_up'][start_idx:end_idx], color = 'magenta', label = 'up', linestyle = '-', markersize = 0)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'Aroon Indicator')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    ax0.grid()
    ax1.grid()

def evaluateMoneyFlow(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['Money_Flow_Volume'][start_idx:end_idx], color = 'green', label = 'MoneyFlow', linestyle = '-', markersize = 0)
    ax1.fill_between(stock_dates[start_idx:end_idx],stock['Money_Flow_Volume'][start_idx:end_idx],np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])),where = stock['Money_Flow_Volume'][start_idx:end_idx] >= np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])), color = 'lightgreen')
    ax1.fill_between(stock_dates[start_idx:end_idx],stock['Money_Flow_Volume'][start_idx:end_idx],np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])),where = stock['Money_Flow_Volume'][start_idx:end_idx] <= np.zeros(len(stock['Money_Flow_Volume'][start_idx:end_idx])), color = 'red')
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'Money Flow Index')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    ax0.grid()
    ax1.grid()    

def evaluateVolumePriceTrend(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['volume_price_trend'][start_idx:end_idx], color = 'magenta', label = 'vpt', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['vpt_signal'][start_idx:end_idx], color = 'green', label = 'vpt_ma', linestyle = '-', markersize = 0)
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'Volume Price Trend')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    ax0.grid()
    ax1.grid()  

def evaluateEaseofMovement(stock, stock_dates, start_idx, end_idx, stock_name): 
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['ease_movement'][start_idx:end_idx], color = 'green', label = 'EOM', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['sma_ease_movement'][start_idx:end_idx], color = 'magenta', label = 'SMA_EOM', linestyle = '-', markersize = 0)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'Ease of Movement Trend')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    ax0.grid()
    ax1.grid() 
    
def evaluateForceIndex(stock, stock_dates, start_idx, end_idx, stock_name): 
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['ForceIndex'][start_idx:end_idx], color = 'green', label = 'ForceI', linestyle = '-', markersize = 0)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'Force Index')
    ax1.set_xlabel('Date')
    ax1.axhline(y=0, linestyle='--', color='tomato', alpha=0.5)
    ax0.set_ylabel('$')   
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)
    ax0.grid()
    ax1.grid() 

def evaluateRelativeStrengthIndex(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['RSI'][start_idx:end_idx], color = 'blue', label = 'RSI', linestyle = '-', markersize = 0)
    ax1.axhline(y=30, color='green', alpha=0.3)
    ax1.axhline(y=70, color='red', alpha=0.3)
    ax0.set_title(r'{} over Time'.format(stock_name))
    ax1.set_title(r'Relative Strength Indicator')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    ax0.grid()
    # place a text box in upper left in axes coords
    textstr = ['Overbought', 'Oversold']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.95, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.text(0.01, 0.25, textstr[1], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.grid()
    plt.legend() 
    
def evaluateADX(stock, stock_dates, start_idx, end_idx, stock_name):   
    fig = plt.figure(figsize=(15,12))
    ax0 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
    ax1 = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
   
    ax1.plot_date(stock_dates[start_idx:end_idx],stock['Close'][start_idx:end_idx], color = 'blue', label = 'Close', linestyle = '-', markersize = 0)
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['adx'][start_idx:end_idx], color = 'k', label = 'ADX', linestyle = '-', markersize = 0)
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['-dm'][start_idx:end_idx], color = 'red', label = '-DM', linestyle = '-', markersize = 0)
    ax0.plot_date(stock_dates[start_idx:end_idx],stock['+dm'][start_idx:end_idx], color = 'green', label = '+DM', linestyle = '-', markersize = 0)

    ax0.axhline(y=25, color='green', alpha=0.3)
    ax0.axhline(y=20, color='red', alpha=0.3)
    ax1.set_title(r'{} over Time'.format(stock_name))
    ax0.set_title(r'ADX Indicator')
    ax1.set_xlabel('Date')
    ax0.set_ylabel('$')   
    ax0.grid()
    # place a text box in upper left in axes coords
    textstr = ['Strong', 'Weak']
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax0.text(0.01, 0.8, textstr[0], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax0.text(0.01, 0.25, textstr[1], transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax1.grid()
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=1)




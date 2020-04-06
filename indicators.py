# -*- coding: utf-8 -*-
"""

modul: financial indicators
modul author: Christoph Doerr

"""

from datetime import datetime, date, timedelta

#import yfinance as yf
#yf.pdr_override()
import pandas as pd
import numpy as np

#from iexfinance import Stock
#from iexfinance import get_historical_data


""" Indicators showing the trend of a stock """
def calculateMovingAverages(stock):   
    """
    Moving Average, Exponential MV and MA Convergence Divergence (MACD)
    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.
    https://www.investopedia.com/terms/m/movingaverage.asp
    https://www.investopedia.com/terms/e/ema.asp
    https://www.investopedia.com/terms/m/macd.asp 
    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with simple moving average, exponential moving average, 
    moving average convergence, moving average convergence histogramm
    """
    periods = [10,20,50,100,200]
    #Calculate moving average
    for period in periods:
       stock['moving_avg_{}'.format(period)] = stock['Adj Close'].rolling(window=period).mean()
    #Calculate exponential moving average
    for period in periods:
       stock['e_moving_avg_{}'.format(period)] = stock['Adj Close'].ewm(span=period).mean() 
    #Calculate moving average convergence
    emaslow = stock['Adj Close'].ewm(span=26, min_periods = 1).mean()
    emafast = stock['Adj Close'].ewm(span=12, min_periods = 1).mean()
    stock['moving_average_convergence'] = emafast - emaslow
    #Calculate Moving Average Convergence Histogramm
    macd_signal = stock['moving_average_convergence'].ewm(span=12, min_periods = 1).mean()
    stock['moving_average_convergence_histogramm'] =  stock['moving_average_convergence'] - macd_signal
    return stock

def calculateTrix(stock, n = 14):
    """Trix (TRIX)
    Shows the percent rate of change of a triple exponentially smoothed moving
    average. A negative Trix Index shows a oversold market a negative Trix index shows a 
    overbought index
    https://www.investopedia.com/terms/t/trix.asp    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with trix indicator
    
    """
    EX1 = stock['Adj Close'].ewm(span=n, min_periods=n - 1).mean()
    EX2 = EX1.ewm(span=n, min_periods=n - 1).mean()
    EX3 = EX2.ewm(span=n, min_periods=n - 1).mean()
    i = 0
    Trix = [0]
    while i + 1 <= stock.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        Trix.append(ROC)
        i = i + 1
    stock['Trix_{}'.format(n)] = Trix
    return stock

def calculateIchimokuIndex(stock):
    """Ichimoku Kinkō Hyō (Ichimoku)
    The overall trend is up when price is above the cloud, down when price is below the cloud, 
    and trendless or transitioning when price is in the cloud.
    https://www.investopedia.com/terms/i/ichimoku-cloud.asp
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Ichimoku Kinko Hyo Indicator
    """
    # Tenkan-Sen Nine Day Perios
    period9_high = stock['High'].rolling(window= 9, min_periods=0).max()
    period9_low = stock['Low'].rolling(window= 9, min_periods=0).min()
    stock['tenkan_sen'] = (period9_high + period9_low) /2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    period26_high = stock['High'].rolling(window= 26, min_periods=0).max()
    period26_low = stock['Low'].rolling(window= 26, min_periods=0).min()
    stock['kijun_sen'] = (period26_high + period26_low) / 2
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    stock['senkou_span_a'] = ((stock['tenkan_sen'] + stock['kijun_sen']) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    period52_high = stock['High'].rolling(window= 52, min_periods=0).max()
    period52_low = stock['Low'].rolling(window= 52, min_periods=0).min()
    stock['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
    return stock

def calculateKST(stock):
    """KST Oscillator (KST Signal)
    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst
   
    Input: pandas series with High, Low, Adj Close and Close columns over time
    """
    ROC1 = stock['Adj Close'].rolling(window=10).mean()
    ROC2 = stock['Adj Close'].rolling(window=15).mean()
    ROC3 = stock['Adj Close'].rolling(window=20).mean()
    ROC4 = stock['Adj Close'].rolling(window=30).mean()
    RCMA1 = ROC1.rolling(window=10).mean()
    RCMA2 = ROC2.rolling(window=10).mean()
    RCMA3 = ROC3.rolling(window=10).mean()
    RCMA4 = ROC4.rolling(window=15).mean()
    stock['KST'] = RCMA1 + RCMA2*2 + RCMA3*3 + RCMA4*4
    return stock

def calculateDetrendedPriceOscillator(stock):     
    """Detrended Price Oscillator (DPO)
    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci
    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with DPO Indicator
    """
    
    sma = stock['Adj Close'].rolling(window=20).mean()
    for index, row in stock.iterrows():
        DPO = stock['Adj Close'].iloc[index - int((20/2)) + 1] -  sma.iloc[index]
        stock.at[index, 'detrended_price_oscillator'] =  DPO
    return stock

def calculateCommodityChannelIndex(stock):
    """Commodity Channel Index (CCI)
    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with CCI Indicator
    """
    
    tp = (stock['High'] + stock['Low'] + stock['Adj Close']) / 3 
    moving_mean = tp.rolling(window=21).mean()
    moving_std = tp.rolling(window=21).std()
    stock['CCI'] =(tp - moving_mean) / (0.015 * moving_std)
    return stock 

def calculateAverageDirectionalIndex(stock, n=14):
    """Average Directional Movement Index (ADX)
    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).
    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.
    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with ADX Indicator
    """
    
    cs = stock['Close'].shift(1)
    pdm = pd.Series(np.amax([stock['High'], cs], axis=0))
    pdn = pd.Series(np.amin([stock['Low'], cs], axis=0))
    tr = pdm - pdn
  
    trs_initial = np.zeros(n)
    trs = np.zeros(len(stock['Adj Close']) - (n-1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)

    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1]/float(n)) + tr[n+i]

    up = stock['High'] - stock['High'].shift(1)
    dn = stock['Low'].shift(1) - stock['High']
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dip = np.zeros(len(stock['Adj Close']) - (n - 1))
    dip[0] = pos.dropna()[0:n].sum()
    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip)-1):
        dip[i] = dip[i-1] - (dip[i-1]/float(n)) + pos[n+i]

    din = np.zeros(len(stock['Adj Close']) - (n - 1))
    din[0] = neg.dropna()[0:n].sum()
    neg = neg.reset_index(drop=True)
    for i in range(1, len(din)-1):
        din[i] = din[i-1] - (din[i-1]/float(n)) + neg[n+i]

    # dip = np.zeros(len(trs))
    # for i in range(len(trs)):
    #     dip[i] = 100 * (dip[i]/trs[i])
    # din = np.zeros(len(trs))
    # for i in range(len(trs)):
    #     din[i] = 100 * (din[i]/trs[i])

    dx = 100 * np.abs((dip - din) / (dip + din))
    
    adx = np.zeros(len(trs)-1)
    adx[n] = dx[0:n].mean()
    for i in range(n+1, len(adx)):
        adx[i] = ((adx[i-1] * (n - 1)) + dx[i-1]) / float(n)

    adx = np.concatenate((trs_initial, adx), axis=0)
    stock['adx'] = adx
    return stock

def calculateAroonIndicator(stock):
    """Aroon Indicator
    Identify when trends are likely to change direction.
    Aroon Up = ((N - Days Since N-day High) / N) x 100
    Aroon Down = ((N - Days Since N-day Low) / N) x 100
    Aroon Indicator = Aroon Up - Aroon Down
    https://www.investopedia.com/terms/a/aroon.asp
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Aroon Indicator
    """
    
    high25 = stock['High'].rolling(window=25, min_periods=25, center=False).max()
    low25 = stock['Low'].rolling(window=25, min_periods=25, center=False).min()
    
    ind = range(0,len(stock))
    indexlist = list(ind)
    stock.index = indexlist
    
    recent_high = high25.iloc[-1]
    ind_of_high = np.where(high25==recent_high)[0][0] 
    
    recent_low = low25.iloc[-1]
    ind_of_low = np.where(low25==recent_low)[0][0]
    
    days_since_high = (len(stock) - 1) - ind_of_high
    days_since_low = (len(stock) - 1) - ind_of_low
    
    stock['aroon_up'] = float(((25 - days_since_high)/25) * 100)
    stock['aroon_down'] = float(((25 - days_since_low)/25) * 100)    
    return stock


""" Indicators caluclating the Volume Trend of the Stock """
def calculateMoneyFlowVolume(stock):
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Chaikin Money Flow Indicator
    """
    stock["MF Multiplier"] = ((stock["Close"] - stock["Low"]) - (stock["High"] - stock["Close"]))/(stock["High"] - stock["Low"])
    stock["MF Volume"] = stock["MF Multiplier"] * stock["Volume"] 
    stock['Money_Flow_Volume'] = stock["MF Volume"].sum()/stock["Volume"].sum()
    return stock

def calculateForceIndex(stock):
    """Force Index (FI)
    It illustrates how strong the actual buying or selling pressure is. High
    positive values mean there is a strong rising trend, and low values signify
    a strong downward trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Force Index
    """
    stock['ForceIndex'] = stock['Adj Close'].diff(5) * stock['Volume']
    return stock

def calculateEaseOfMovement(stock):
    """Ease of movement (EoM, EMV)
    It relate an asset's price change to its volume and is particularly useful
    for assessing the strength of a trend.
    https://en.wikipedia.org/wiki/Ease_of_movement
    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Ease of Movement and Ease of Moving Average Movement Index
    """
    stock['ease_movement'] = ((stock['High'].diff(1) + stock['Low'].diff(1)) * (stock['High'] - stock['Low']) / (2 * stock['Volume']))*100000000
    stock['sma_ease_movement'] = stock['ease_movement'].rolling(window=14, min_periods = 14).mean()
    return stock

def calculateVolumePriceTrendIndicator(stock):
    """Volume Price Trend Indicator
    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Volume Price Trend Indicator
    """
    
    vpt = (stock['Volume'] * ((stock['Adj Close'] - stock['Adj Close'].shift(1, fill_value=stock['Adj Close'].mean()))
                               / stock['Adj Close'].shift(1, fill_value=stock['Adj Close'].mean())))
    stock['volume_price_trend'] = vpt
    return stock

""" Indicators calculating the volatility of the stock"""
def calculateAverageTrueRange(stock):
    """Average True Range (ATR)
    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Average True Range Indicator
    """
    for index, row in stock.iterrows():
        true_range = max(row["High"] - row["Low"],row["High"] - stock.iloc[index-1]["Close"],row["Low"] - stock.iloc[index-1]["Close"])
        stock.at[index, 'true_range'] =  true_range
    stock['avg_true_range'] = stock['true_range'].rolling(min_periods=14, window=14, center=False).mean()
    return stock

def calculateBollingerBands(stock):
    """ Bollinger Bands
    Calculate upper band at K times an N-period standard deviation above the moving average (MA + Kdeviation)
    and the lower band at K times an N-period standard deviation below the moving average (MA − Kdeviation).
    If the stock hits the lower band it means its oversold and a buy possibility, if the stock hits the upper band
    it means the stock is overbought and it is a sell possibility
    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Bollinger Upper, Lower and Distance to Bands
    """
    stock['lower_bound'] = stock['Adj Close'].rolling(window=20).mean() - 2*stock['Adj Close'].rolling(window=20).std()
    stock['upper_bound'] = stock['Adj Close'].rolling(window=20).mean() + 2*stock['Adj Close'].rolling(window=20).std()
    stock['distance_2upBound'] = stock['upper_bound'] - stock['Adj Close']
    stock['distance_2lowBound'] = stock['Adj Close'] - stock['lower_bound']
    return stock
   
    
def calculateKeltnerChannel(stock):
    """KeltnerChannel
    Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and
    channel direction. Channels can also be used to identify overbought and oversold levels when the trend
    is flat.
    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels
    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Keltner Bands
    """
    
    stock['keltner_channelM'] = stock['Adj Close'].ewm(span=20).mean()
    for index, row in stock.iterrows(): 
        stock['keltner_channelU'] = stock['keltner_channelM'] + 2*max(row["High"] - row["Low"],row["High"] - stock.iloc[index-1]["Close"],row["Low"] - stock.iloc[index-1]["Close"])
        stock['keltner_channelL'] = stock['keltner_channelM'] - 2*max(row["High"] - row["Low"],row["High"] - stock.iloc[index-1]["Close"],row["Low"] - stock.iloc[index-1]["Close"])
    return stock


""" Indicators calculating the Momentum of the Stock"""
def calculateRelativeStrengthIndex(stock):
    """Relative Strength Index
    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Relative Strength Index
    """
    stock["Change"] = stock["Close"].diff()
    # Make the positive gains (up) and negative gains (down) Series
    up, down = stock["Change"].copy(), stock["Change"].copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Calculate the EWMA
    roll_up_evm = up.ewm(span=14).mean()
    roll_down_evm = down.abs().ewm(span=14).mean()
    # Calculate the SMA
    roll_up_sma = up.rolling(window=14).mean()
    roll_down_sma = down.abs().rolling(window=14).mean()
    # Calculate the SMA
    roll_up = up.rolling(min_periods=14, window=14, center=False).mean()
    roll_down = down.abs().rolling(min_periods=14, window=14, center=False).mean()
    
    # Calculate the RSI based on EWMA
    rs_evm = roll_up_evm / roll_down_evm
    rsi_evm = 100.0 - (100.0 / (1.0 + rs_evm))    
    # Calculate the RSI based on EWMA
    rs_sma = roll_up_sma / roll_down_sma
    rsi_sma = 100.0 - (100.0 / (1.0 + rs_sma))
    # Calculate the RSI based on EWMA
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
        
    stock['RSI_EVM'] = rsi_evm
    stock['RSI_SMA'] = rsi_sma
    stock['RSI'] = rsi
    return stock     

def calculateStochasticK(stock):
    """Stockastic K/D Index
    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Stockastic K and D Index
    """
    #Create the "L14" column in the DataFrame
    low14d = stock['Low'].rolling(window=14).min()
    #Create the "H14" column in the DataFrame
    high14d = stock['High'].rolling(window=14).max()  
    #Create the "%K" column in the DataFrame
    stock['%K'] = 100*((stock['Close'] - low14d) / (high14d - low14d) )    
    #Create the "%D" column in the DataFrame
    stock['%D'] = stock['%K'].rolling(window=3).mean()
    return stock

def calculateWilliamsR(stock):
    """WilliamsR Index
    When the indicator is between -20 and zero the price is overbought, or near the high of its recent price range. 
    When the indicator is between -80 and -100 the price is oversold, or far from the high of its recent range.
    https://www.investopedia.com/terms/w/williamsr.asp
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with WilliamsR Index
    """
    hh = stock['High'].rolling(14, min_periods=0).max()  # highest high over lookback period lbp
    ll = stock['High'].rolling(14, min_periods=0).min()  # lowest low over lookback period lbp
    stock['williamsR'] = -100 * (hh - stock['Close']) / (hh - ll)    
    return stock    

def calculateRateofChange(stock):
    """Rate of Change Index
    Calculates the rate of change index with a period of 12 days
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Rate of Change Index
    """
    stock['rate_of_change'] = ((stock['Adj Close'] - stock['Adj Close'].shift(12))/stock['Adj Close'].shift(12))*100
    return stock

""" Other Indicators """
def calculateDailyReturn(stock):
    """Daily Return (DR)
    Returns the daily return and the label if you would have earned money with a day trade
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Daily Return and Label
    """
    stock['daily_return'] = stock['Adj Close'].pct_change(1) + 1
    label = stock['daily_return'].copy()
    label[stock['daily_return'] < 1] = 0
    label[stock['daily_return'] > 1] = 1
    stock['daily_label'] = label
    return stock 

def calculateIndicators(stock):
    calculateVolumePriceTrendIndicator(stock)
    calculateEaseOfMovement(stock)
    calculateIchimokuIndex(stock)
    calculateMoneyFlowVolume(stock)
    calculateRateofChange(stock)
    calculateKST(stock)
    calculateRelativeStrengthIndex(stock)
    calculateCommodityChannelIndex(stock)
    calculateForceIndex(stock)
    calculateWilliamsR(stock)
    calculateTrix(stock)
    calculateKeltnerChannel(stock)
    calculateAroonIndicator(stock)
    calculateDetrendedPriceOscillator(stock)
    calculateBollingerBands(stock)
    calculateMovingAverages(stock)
    calculateStochasticK(stock)
    calculateDailyReturn(stock)
    return stock
# -*- coding: utf-8 -*-
"""

modul: financial indicators
modul author: Christoph Doerr

"""

import pandas as pd
import numpy as np

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
    rocma1 = ((stock['Close'] - stock['Close'].shift(10, fill_value=stock['Close'].mean()))
              / stock['Close'].shift(10, fill_value=stock['Close'].mean())).rolling(window=10, min_periods=0).mean()
    rocma2 = ((stock['Close'] - stock['Close'].shift(15, fill_value=stock['Close'].mean()))
              / stock['Close'].shift(15, fill_value=stock['Close'].mean())).rolling(window=10, min_periods=0).mean()
    rocma3 = ((stock['Close'] - stock['Close'].shift(20, fill_value=stock['Close'].mean()))
              / stock['Close'].shift(20, fill_value=stock['Close'].mean())).rolling(window=10, min_periods=0).mean()
    rocma4 = ((stock['Close'] - stock['Close'].shift(30, fill_value=stock['Close'].mean()))
              / stock['Close'].shift(30, fill_value=stock['Close'].mean())).rolling(window=15, min_periods=0).mean()
    stock['KST'] = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    stock['signal_KST']  = stock['KST'] .rolling(window=9, min_periods=0).mean()

    return stock

def calculateDetrendedPriceOscillator(stock):     
    """Detrended Price Oscillator (DPO)
    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.
    https://www.investopedia.com/terms/d/detrended-price-oscillator-dpo.asp
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
     When the CCI is above zero it indicates the price is above the historic average. 
     When CCI is below zero, the price is below the hsitoric average. High readings of 100 or above, for example,
     indicate the price is well above the historic average and the trend has been strong to the upside.
     Low readings below -100, for example, indicate the price is well below the historic average and the
     trend has been strong to the downside. Going from negative or near-zero readings to +100 can be used 
     as a signal to watch for an emerging uptrend. Going from positive or near-zero readings to -100 may indicate 
     an emerging downtrend. 

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    https://www.investopedia.com/terms/c/commoditychannelindex.asp
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
    
    https://www.investopedia.com/terms/a/adx.asp
    ADX readings above 25 indicate that a security is trending, while readings below 25 indicate sideways price action. 

    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with ADX Indicator
    """
    
    #Calculate True Range (Step 5)
    cs = stock['Close'].shift(1)
    pdm = pd.Series(np.amax([stock['High'], cs], axis=0))
    pdn = pd.Series(np.amin([stock['Low'], cs], axis=0))
    tr = pdm - pdn
    
    trs_initial = np.zeros(n-1)
    trs = np.zeros(len(stock['Adj Close']) - (n-1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    #Smooth TR (Step6-8)
    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1]/float(n)) + tr[n+i]

    #Calculate +dm, -dm and smooth it (1-4)
    up = stock['High'] - stock['High'].shift(1)
    dn = stock['Low'].shift(1) - stock['Low']
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dmp = np.zeros(len(stock['Adj Close']) - (n - 1))
    dmp[0] = pos.dropna()[0:n].sum()
    pos = pos.reset_index(drop=True)
    for i in range(1, len(dmp)-1):
        dmp[i] = dmp[i-1] - (dmp[i-1]/float(n)) + pos[n+i]

    dmn = np.zeros(len(stock['Adj Close']) - (n - 1))
    dmn[0] = neg.dropna()[0:n].sum()
    neg = neg.reset_index(drop=True)
    for i in range(1, len(dmn)-1):
        dmn[i] = dmn[i-1] - (dmn[i-1]/float(n)) + neg[n+i]
 
    #9-10
    dip = np.zeros(len(trs))
    for i in range(len(trs)):
        dip[i] = 100*(np.divide(dmp[i], trs[i], out=np.zeros_like(dmp[i]), where=trs[i]!=0))
        # dip[i] = 100 * (dmp[i]/trs[i]) ############ Division by zero
    din = np.zeros(len(trs))
    for i in range(len(trs)):
        din[i] = 100*(np.divide(dmn[i], trs[i], out=np.zeros_like(dmn[i]), where=trs[i]!=0))
        # din[i] = 100 * (dmn[i]/trs[i]) ############ Division by zero
    
    #Directional Movement Index(11)
    # dx = 100 * np.abs((dip - din) / (dip + din))  ############ Division by zero
    dx = 100*np.abs(np.divide(dip - din, dip + din, out=np.zeros_like(dip), where=dip + din!=0))
    #Calculate ADX (12-14)
    adx = np.zeros(len(trs))
    adx[n] = np.mean(dx[0:n])
    for i in range(n+1, len(adx)):
        adx[i] = ((adx[i-1] * (n - 1)) + dx[i-1]) / float(n)
    adx = np.concatenate((trs_initial, adx), axis=0)
    dmn = np.concatenate((trs_initial, din), axis=0)
    dmp = np.concatenate((trs_initial, dip), axis=0)

    stock['adx'] = adx
    stock['+dm'] = dmp
    stock['-dm'] = dmn
    return stock

def calculateAroonIndicator(stock):
    """Aroon Indicator
    Identify when trends are likely to change direction.
    When the Aroon Up is above the Aroon Down, it indicates bullish price behavior.
    When the Aroon Down is above the Aroon Up, it signals bearish price behavior.
    Crossovers of the two lines can signal trend changes. For example, when Aroon Up crosses above
    Aroon Down it may mean a new uptrend is starting. The indicator moves between zero and 100. 
    A reading above 50 means that a high/low (whichever line is above 50) was seen within the last 12 periods.
    A reading below 50 means that the high/low was seen within the 13 periods.
    https://www.investopedia.com/terms/a/aroon.asp
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Aroon Indicator
    """
    
    rolling_close = stock['Adj Close'].rolling(window=25, min_periods=0)
    stock['aroon_up'] = rolling_close.apply(lambda x: float(np.argmax(x) + 1) / 25 * 100, raw=True)
    stock['aroon_down'] = rolling_close.apply(lambda x: float(np.argmin(x) + 1) / 25 * 100, raw=True)
    return stock


""" Indicators caluclating the Volume Trend of the Stock """
def calculateMoneyFlowVolume(stock):
    """ Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period. Chaikin Money Flow measures buying
    and selling pressure for a given period of time. A move into positive territory indicates buying pressure,
    while a move into negative territory indicates selling pressure. 
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Chaikin Money Flow Indicator
    """
    mfv = ((stock["Close"] - stock["Low"]) - (stock["High"] - stock["Close"]))/(stock["High"] - stock["Low"])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= stock['Volume']
    stock['Money_Flow_Volume']  = mfv.rolling(window=20, min_periods=0).sum() / stock['Volume'].rolling(window=20, min_periods=0).sum()
    return stock

def calculateForceIndex(stock):
    """Force Index (FI)
    A rising force index, above zero, helps confirm rising prices.
    A falling force index, below zero, helps confirm falling prices.
    A breakout, or a spike, in the force index, helps confirm a breakout in price.
    If the force index is making lower swing highs while the price is making higher swing highs, 
    this is bearish divergence and warns the price may soon decline. If the force index is making higher
    swing lows while the price is making lower swing lows, this is bullish divergence and warns the price may
    soon head higher.
    https://www.investopedia.com/terms/f/force-index.asp    
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Force Index
    """
    fi = (stock['Close'] - stock['Close'].shift(1)) * stock['Volume']
    stock['ForceIndex'] = fi.ewm(span=13, min_periods = 1).mean()
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

def calculateVolumePriceTrend(stock):
    """Volume Price Trend Indicator
    A signal line, which is just a moving average of the indicator, can be applied and used to generate trading signals.
    For example, a trader may buy a stock when the VPT line crosses above its signal line and sell when the VPT line passes
    below its signal line.
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Volume Price Trend Indicator
    """
    
    vpt = (stock['Volume'] * ((stock['Adj Close'] - stock['Adj Close'].shift(1, fill_value=stock['Adj Close'].mean()))
                               / stock['Adj Close'].shift(1, fill_value=stock['Adj Close'].mean())))
    stock['volume_price_trend'] = vpt.shift(1, fill_value=vpt.mean()) + vpt
    stock['vpt_signal'] = stock['volume_price_trend'].rolling(window=20,).mean()
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
        true_range = max(row["High"] - row["Low"],row["High"] - stock.iloc[index-1]["Close"],abs(row["Low"] - stock.iloc[index-1]["Close"]))
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
    is flat. If the stock hits the lower band it means its oversold and a buy possibility, if the stock hits the upper band
    it means the stock is overbought and it is a sell possibility
    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels
    https://www.investopedia.com/terms/k/keltnerchannel.asp
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Keltner Bands
    """
    
    stock['keltner_channelM'] = stock['Close'].ewm(span=20).mean()
    for index, row in stock.iterrows(): 
        stock['keltner_channelU'] = stock['keltner_channelM'] + 2*max(row["High"] - row["Low"],row["High"] - stock.iloc[index-1]["Close"],abs(row["Low"] - stock.iloc[index-1]["Close"]))
        stock['keltner_channelL'] = stock['keltner_channelM'] - 2*max(row["High"] - row["Low"],row["High"] - stock.iloc[index-1]["Close"],abs(row["Low"] - stock.iloc[index-1]["Close"]))
    return stock


""" Indicators calculating the Momentum of the Stock"""
def calculateRelativeStrengthIndex(stock):
    """Relative Strength Index
    Compares the magnitude of recent gains and losses over a specified time period to measure speed 
    and change of price movements of a security. It is primarily used to attempt to identify 
    overbought or oversold conditions in the trading of an asset.
    The RSI compares bullish and bearish price momentum plotted against the graph of an asset's price.
    Signals are considered overbought when the indicator is above 70% and oversold when the indicator is below 30%.
    https://www.investopedia.com/terms/r/rsi.asp
    Input: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with Relative Strength Index
    """
    diff = stock['Close'].diff(1)
    up = diff.where(diff > 0, 0.0)
    dn = -diff.where(diff < 0, 0.0)
    emaup = up.ewm(alpha=1/14, min_periods=0, adjust=False).mean()
    emadn = dn.ewm(alpha=1/14, min_periods=0, adjust=False).mean()
    rs = emaup / emadn
    stock['RSI'] = pd.Series(np.where(emadn == 0, 100, 100-(100/(1+rs))), index=stock['Close'].index)
    return stock     

def calculateStochasticK(stock):
    """Stockastic K/D Index
    When the indicator is between 20 and zero the price is oversold, or far from the high of its recent range. 
    When the indicator is between 80 and 100 the price is overbought, or close from the high of its recent range.
    https://www.investopedia.com/ask/answers/020615/how-do-i-read-and-interpret-stochastic-oscillator.asp
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
    for index, row in stock.iterrows():
        if(stock['%K'][index] < 0 or stock['%K'][index] > 100):
            print("Watch out, %K is out of bounds")
    return stock

def calculateWilliamsR(stock):
    """WilliamsR Index
    When the indicator is between -20 and zero the price is overbought, or near the high of its recent price range. 
    When the indicator is between -80 and -100 the price is oversold, or far from the high of its recent range.
    https://www.investopedia.com/terms/w/williamsr.asp
    Input stock: pandas series with High, Low, Adj Close and Close columns over time
    Return stock: pandas series with WilliamsR Index
    """
    hh = stock['High'].rolling(14, min_periods=0).max()  # highest high over lookback period lbp
    ll = stock['Low'].rolling(14, min_periods=0).min()  # lowest low over lookback period lbp
    stock['williamsR'] = -100 * (hh - stock['Close']) / (hh - ll)
    for index, row in stock.iterrows():
        if(stock['williamsR'][index] > 0 or stock['williamsR'][index] < -101):
            print("Watch out, WilliamsR is out of bounds")
    return stock    

def calculateRateofChange(stock):
    """Rate of Change Index
    Calculates the rate of change index with a period of 12 days
    Input stock: pandas series with High, Low, Adj Close and Close columns over time
    Return stock: pandas series with Rate of Change Index
    """
    stock['rate_of_change'] = ((stock['Adj Close'] - stock['Adj Close'].shift(12))/stock['Adj Close'].shift(12))*100
    return stock

""" Other Indicators """
def calculateDailyReturn(stock):
    """Daily Return (DR)
    Returns the daily return and the label if you would have earned money with a day trade
    Input stock: pandas series with High, Low, Adj Close and Close columns over time
    Return stock: pandas series with Daily Return and Label
    """
    stock['daily_return'] = stock['Adj Close'].pct_change(1) + 1
    label = stock['daily_return'].copy()
    label[stock['daily_return'] < 1] = 0
    label[stock['daily_return'] > 1] = 1
    stock['daily_label'] = label
    return stock 

def calculateIndicators(stock):
    """
    Function that calculates all indicators
    Input stock: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with all indicators
    """
    calculateVolumePriceTrend(stock)
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

def calculateIndicatorsforEvaluation(stock):
    """
    Function that calculates indicators for evaluation
    Input stock: pandas series with High, Low, Adj Close and Close columns over time
    Return: pandas series with all indicators
    """
    calculateMovingAverages(stock)
    calculateBollingerBands(stock)
    calculateIchimokuIndex(stock)
    calculateMoneyFlowVolume(stock)
    calculateRelativeStrengthIndex(stock)
    calculateWilliamsR(stock)
    calculateCommodityChannelIndex(stock)
    return stock
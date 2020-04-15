# -*- coding: utf-8 -*-
"""
modul: model utilities
modul author: Christoph Doerr

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import tensorflow as tf

def splitData(stock, start_idx, end_idx, forcast_idx, ratio= 0.75):
    data_points = end_idx - start_idx
    data_points_train = int(data_points * ratio)
    train_data = stock.iloc[start_idx:(start_idx + data_points_train)].reset_index()
    test_data = stock.iloc[(start_idx + data_points_train + 1):end_idx].reset_index()
    predict_data = stock.iloc[end_idx:forcast_idx].reset_index()
    return (train_data, test_data, predict_data)

def getXY(stock):
    #toDo: use regex filter
    X = stock.drop(['Date','daily_label'], axis = 1).to_numpy(dtype='float32')
    Y = stock['daily_label'].to_numpy(dtype='float32')
    return (X,Y)

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

def safeModel(model, safe_model_path, number_epochs, batch_size, model_name=None):
    if model_name == None:
        model_name = len(os.listdir(safe_model_path)) + 1
    model.save('{}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))
    print('safed model to {}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))

def loadModel(safe_model_path, model_name, number_epochs, batch_size):
    print('loading model to {}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))
    return tf.keras.models.load_model('{}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))
    
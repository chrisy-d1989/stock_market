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
from keras import utils as np_utils



def trainModel(model, X_train, Y_train, X_test, Y_test, batch_size=32, number_epochs=51, validation_split = 0.2, loss = 'sparse_categorical_crossentropy'):
    # model.compile(optimizer='adam', loss=loss, metrics=['accuracy'],)
    model.compile(loss=loss, optimizer='adam', metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train, epochs=number_epochs, batch_size=batch_size, validation_split = validation_split)
    # model.evaluate(X_test, Y_test)
    return model

def splitData(stock, start_idx, end_idx, forcast_idx, ratio= 0.75):
    data_points = end_idx - start_idx
    data_points_train = int(data_points * ratio)
    train_data = stock.iloc[start_idx:(start_idx + data_points_train)].reset_index()
    test_data = stock.iloc[(start_idx + data_points_train + 1):end_idx].reset_index()
    predict_data = stock.iloc[end_idx:forcast_idx].reset_index()
    return (train_data, test_data, predict_data)

def getXY(train_data, test_data, predict_data):
    #toDo: use regex filter
    X_train = train_data.drop(['index', 'Date','daily_label', 'future_close'], axis = 1).to_numpy(dtype='float32')
    Y_train = train_data['daily_label'].to_numpy(dtype='float32')
    X_test = test_data.drop(['index', 'Date','daily_label', 'future_close'], axis = 1).to_numpy(dtype='float32')
    Y_test = test_data['daily_label'].to_numpy(dtype='float32')
    X_predict = predict_data.drop(['index', 'Date','daily_label', 'future_close'], axis = 1).to_numpy(dtype='float32')
    Y_predict = predict_data['daily_label'].to_numpy(dtype='float32')
    return (X_train, Y_train, X_test, Y_test, X_predict, Y_predict)

def prepareDataforLTSM(data, sample_length = 300, Y_data = False):
    number_samples = int(len(data)/sample_length)
    if Y_data:
        features = 1
    else:
        features = len(data[0]) 
    samples = np.zeros((number_samples, sample_length, features))  
    for j in range(0, number_samples):
        if Y_data:
            sample = data[j*sample_length : j*sample_length + sample_length]
            samples[j, :, 0] = sample
        else:    
            sample = data[j*sample_length : j*sample_length + sample_length, :]
            samples[j, :, :] = sample
    return samples

def standardizeIndicators(stock):
    stock_std = stock.copy()
    for key, value in stock_std.iteritems():
        if key == 'Date':
            continue
        elif (key == 'daily_label'):
                mean = 0
                std = 1
        else:
            mean = value.mean()
            std = value.std()
        stock_std.loc[:,key] = (stock.loc[:, key] - mean) / std
    return (stock_std)

def normalizeIndicators(stock):
    stock_std = stock.copy()
    for key, value in stock_std.iteritems():
        if key == 'Date':
            continue
        else:
            x = stock_std[[key]].values.astype(float)
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            stock_std[key] = pd.DataFrame(x_scaled)
    return (stock_std)

def safeModel(model, safe_model_path, number_epochs, batch_size, model_name=None):
    if model_name == None:
        model_name = len(os.listdir(safe_model_path)) + 1
    model.save('{}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))
    print('safed model to {}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))

def loadModel(safe_model_path, model_name):
    print('loading model to {}{}.h5'.format(safe_model_path, model_name))
    return tf.keras.models.load_model('{}{}.h5'.format(safe_model_path, model_name))

def defineCNN(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='elu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model
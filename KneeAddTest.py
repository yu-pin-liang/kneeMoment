import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import layers, Input
from keras.layers import LSTM
from keras.models import save_model, load_model
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)

def kneeRNN ():
    ##reshape
    x = np.stack((latup,latz,latx),axis=2)
    xTest = np.stack((latupTest,latzTest,latxTest),axis=2)
    m = m.to_numpy()
    mTest = mTest.to_numpy()

    # normalized usimg traditional way
    train_mean = x.mean()
    train_std = x.std()
    x = ((x - train_mean) / train_std).round(5)

    target_mean = m.mean()
    target_std = m.std()
    m = ((m - target_mean) / target_std).round(5)

    test_mean = xTest.mean()
    test_std = xTest.std()
    xTest = ((xTest - test_mean) / test_std).round(5)

    testTarget_mean = mTest.mean()
    testTarget_std = mTest.std()
    mTest = ((mTest - testTarget_mean) / testTarget_std).round(5)

    # RNN model
    time_steps = 55
    n_chanel = 3
    model = Sequential()
    lstm1 = LSTM(time_steps, input_shape=(time_steps, n_chanel), return_sequences=True, dropout=0.2)
    lstm2 = LSTM(time_steps, return_sequences=True, dropout=0.2)

    model.add(lstm1)
    model.add(lstm2)
    model.add(lstm2)
    model.add(Dense(100))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["RootMeanSquaredError"])

    model.fit(x, m, epochs=50, batch_size=5, verbose=1)

    score = model.evaluate(xTest, mTest, verbose=0)
    # trainPredict = model.predict(x)
    testPredict = model.predict(xTest)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    #
    # predict = scaler.inverse_transform([testPredict])
    # truth = scaler.inverse_transform([mTest])
    print(mTest.shape)
    print(mTest[0])
    p = plt.plot(mTest[3])
    p = plt.plot(testPredict[3, :])
    plt.show()
    print(testPredict.shape)
import os
import numpy as np
import pandas as pd

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                            ExtraTreesRegressor, GradientBoostingRegressor,
                            RandomForestRegressor)

import tensorflow as tf
from tensorflow.contrib import rnn

import matplotlib.pyplot as plt

file = 'history/AUD_JPY_D.csv'
folder = 'history/'

def walk_folder(folder):

    all_files = list()
    for files in os.walk(folder):
        for f in files[2]:
            f = os.path.join(folder, f)
            all_files.append(f)

    return all_files

# TODO: test this some more, it works with Keras and sklearn; but had trouble with TF.
def split_data(X, y, n, split, o):

    m = int(n * split)
    X_train, X_test = X[0:m], X[m:n-o]
    y_train, y_test = y[o:m+o], y[m+o:n]

    return X_train, X_test, y_train, y_test

def tf_nn_research(file, epochs=10, split=0.6, o=1):
    # import data and clean it for processing
    df = pd.read_csv(file)
    df = df[['Close', 'Open','High', 'Low']]
    n = df.shape[0]
    p = df.shape[1]
    data = df.values
    # Training and test data; split_data doesn't work too well with TF atm
    train_start = 0
    train_end = int(0.6*n)
    test_start = train_end + 1
    test_end = n
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]
    # Scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    # Build X and y
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    # length of training data
    m = X_train.shape[1]
    # neurons in each layer
    neurons_1 = 1024
    neurons_2 = 512
    neurons_3 = 256
    neurons_4 = 128
    # session
    sess = tf.InteractiveSession()
    # Placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, m])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    # initializers
    sigma = 1
    W = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    b = tf.zeros_initializer()
    # hidden weights
    W_hidden_1 = tf.Variable(W([m, neurons_1]))
    b_hidden_1 = tf.Variable(b([neurons_1]))
    W_hidden_2 = tf.Variable(W([neurons_1, neurons_2]))
    b_hidden_2 = tf.Variable(b([neurons_2]))
    W_hidden_3 = tf.Variable(W([neurons_2, neurons_3]))
    b_hidden_3 = tf.Variable(b([neurons_3]))
    W_hidden_4 = tf.Variable(W([neurons_3, neurons_4]))
    b_hidden_4 = tf.Variable(b([neurons_4]))
    # output weights
    W_out = tf.Variable(W([neurons_4, 1]))
    b_out = tf.Variable(b([1]))
    # hidden layers
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), b_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), b_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), b_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), b_hidden_4))
    # transpose the output layer
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), b_out))
    # cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))
    # optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    # init
    sess.run(tf.global_variables_initializer())
    # start plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)
    line2, = ax1.plot(y_test * 0.5)
    plt.show()
    # fit neural net
    batch_size = 1
    mse_train = []
    mse_test = []
    # run the network
    for e in range(epochs):
        # shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # run optimizer with batch
            sess.run(opt, feed_dict={X: batch_x, Y: batch_y})
            # show progress
            if np.mod(i, 50) == 0:
                # MSE train and test
                mse_train.append(sess.run(mse, feed_dict={X: X_train, Y: y_train}))
                mse_test.append(sess.run(mse, feed_dict={X: X_test, Y: y_test}))
                print('MSE Train: ', mse_train[-1])
                print('MSE Test: ', mse_test[-1])
                # prediction
                pred = sess.run(out, feed_dict={X: X_test})
                line2.set_ydata(pred)
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                plt.pause(0.10)
                plt.savefig('graphs/%s_%s.png' %(str(e), str(i)))

# Xcols=20 returns better results than Xcols=1, with o=5
# Xcols=20 and o=1 returns 90-95% accuracy for every model
def ensemble_research(file, split=0.6, o=5, Xcols=1):
    """Xcols == 1 or 20; 1 for 'Open' or 20 for all input values"""
    # import, split, and prepare data
    df = pd.read_csv(file)
    n = len(df)
    if Xcols == 1:
        y = df['Close'].values.reshape(n, 1)
        X = df['Open'].values.reshape(n, 1)
        X_train, X_test, y_train, y_test = split_data(X, y, n, split, o)
    elif Xcols == 20:
        y = df['Close'].values
        a = df['Close'].mean()
        df.fillna(a, inplace=True)
        for col in df.columns:
            if 'Unnamed' in col:
                df.drop(col, axis=1, inplace=True)
        df.drop('Close', axis=1, inplace=True)
        X = df[df.columns.tolist()].values
        X_train, X_test, y_train, y_test = split_data(X, y, n, split, o)

    # instantiate the models to be explored
    models = [AdaBoostRegressor(),
              BaggingRegressor(),
              DecisionTreeRegressor(),
              ExtraTreesRegressor(),
              GradientBoostingRegressor(),
              RandomForestRegressor()]

    # set and run models
    scores = list()
    for model in models:
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        score = model.score(X_test, y_test)
        scores.append(score)
        plt.plot(yhat, label='predictions')
        plt.plot(y_test, label='real data')
        plt.title(str(score))
        plt.legend()
        plt.show()

# multivariate LSTM in Keras (OHL, plus technical indicators; 20 columns in total)
def LSTMulti_research(file, o=5, split=0.6, show_loss=False):
    # import and clean the dataframe as needed, filling in any missing data
    df = pd.read_csv(file)
    n = len(df)
    y = df['Close'].values
    a = df['Close'].mean()
    df.fillna(a, inplace=True)
    for col in df.columns:
        if "Unnamed" in col:
            df.drop(col, axis=1, inplace=True)
    df.drop('Close', axis=1, inplace=True)
    X = df[df.columns.tolist()].values
    # split data and reshape input values
    X_train, X_test, y_train, y_test = split_data(X,y,n,split,o)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(20, input_shape=(1,20,), activation='linear'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(X_train, y_train, epochs=50, batch_size=1,
                        validation_data=(X_test, y_test), verbose=2, shuffle=False)
    score = model.evaluate(X_test, y_test)#history['']
    if show_loss == True:
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title('LSTMulti Score' + str(score))
        plt.legend()
        plt.show()
    else:
        plt.plot(y_test, label='prices')
        plt.plot(model.predict(X_test), label='predictions')
        plt.title('LSTMulti Score ' + str(score))
        plt.legend()
        plt.show()

def plot_Ridge(file, o=5, split=0.6):
    df = pd.read_csv(file)
    n = len(df)
    X = df['Open'].values.reshape(n,1)
    y = df['Close'].values.reshape(n,1)
    X_train, X_test, y_train, y_test = split_data(X, y, n, split, o)

    model = Ridge()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    print("Model score: %.2f" % score)

    plt.plot([x for x in range(len(X_test))], predictions, label='predictions')
    plt.plot([x for x in range(len(X_test))], y_test, label='actual data')
    plt.legend()
    plt.show()

def plot_LinearRegression(file, o=5, split=0.6):
    # o == offset, how far away is the prediction we want to make?
    # changing o from 5 to 1 improves prediction accuract from 81% to 94%
    df = pd.read_csv(file)
    n = len(df)
    # convert pd.Series to np.ndarrays now for manipulation
    X = df['Open'].values.reshape(n,1)
    y = df['Close'].values.reshape(n,1)

    # slight offset, X[t] is used to 'predict' y[t+o]
    X_train, X_test, y_train, y_test = split_data(X,y,n,split,o)

    # check to make sure the two sets of arrays match in length
    print("Training sizes: X: %.2f, y: %.2f" % (X_train.shape[0], y_train.shape[0]))
    print("Test sizes: X: %.2f, y: %.2f" % (X_test.shape[0], y_test.shape[0]))

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    print("Model score: ", score)
    plt.plot([x for x in range(len(X_test))], predictions, label='predictions')
    plt.plot([y for y in range(len(y_test))], y_test, label='real data')
    plt.legend()
    plt.show()

#arima_research(file, 'Close', 5, 1, 1) # best fit
def arima_research(file, series, p, d, q):
    # prepare data
    df = pd.read_csv(file)
    X = df[series].values
    size = int(len(X) * 0.6)
    train, test = X[:size], X[size:]
    history = [x for x in train]
    predictions = list()

    # fit the model and make predictions
    for t in range(len(test)):
        model = ARIMA(history, order=(p,d,q))
        model_fit = model.fit(disp=0)
        #print(model_fit.summary())
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        err = abs(yhat - obs)
        print("Predicted: %.3f, Expected: %.3f, Error: %.3f" % (yhat, obs, err))

    # print summary and graph results
    error = mean_squared_error(test, predictions)
    print("Test MSE: %.3f" % (error))
    plt.plot(test, label='actual prices')
    plt.plot(predictions, label='predicted prices')
    plt.legend()
    plt.show()

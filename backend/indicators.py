# short list of trading indicators
import numpy as np
import pandas as pd
from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split

def sma(series, window):
    mavg = series.rolling(window=window, min_periods=window).mean()
    return mavg

# exponential weighted moving average
def ewma(series, span):
    ema = pd.ewma(series, span=span)
    return ema

def pivotPoints(df):
    n = len(df) - 1 # pivotPoints would be based on the last candle
    pp = (df['High'] + df['Low'] + df['Close']) / 3
    r1 = (2 * pp) - df['Low']
    s1 = (2 * pp) - df['High']
    r2 = (pp - s1) + r1
    s2 = pp - (r1 - s1)
    r3 = (pp - s2) + r2
    s3 = pp - (r2 - s2)

    pivots = {'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3,
                       'S1': s1, 'S2': s2, 'S3': s3}

    return pivots

# rate of change, applied directly to the dataframe
def roc(df):
    closes = df['Close'].apply(float)
    df['ROC'] = closes.diff() * 100
    df['ROC'].fillna(0)
    return df

# stochastic oscillator, applied directly to the dataframe
def stoch(df, period_K, period_D, graph=False):
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Lstoch'] = df['Low'].rolling(window=period_K).min()
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Hstoch'] = df['High'].rolling(window=period_K).max()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['%K'] = 100*((df['Close'] - df['Lstoch']) / (df['Hstoch'] - df['Lstoch']))
    df['%D'] = df['%K'].rolling(window=period_D).mean()

    if graph == True:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20,10))
        df['Close'].plot(ax=axes[0])#, title='Close'
        df[['K', 'D']].plot(ax=axes[1])#, title='Oscillator'
        plt.show()

    return df

# bollinger bands, applied directly to the dataframe
def bollBands(df, window, n_std, graph=False):
    close = df['Close']
    rolling_mean = close.rolling(window).mean()
    rolling_std  = close.rolling(window).std()

    df['rolling_mean']   = rolling_mean
    df['boll_high'] = rolling_mean + (rolling_std * n_std)
    df['boll_low']  = rolling_mean - (rolling_std * n_std)

    if graph == True:
        plt.plot(close)
        plt.plot(df['Rolling Mean'])
        plt.plot(df['Bollinger High'])
        plt.plot(df['Bollinger Low'])
        plt.show()

    return df

# AdaBoostRegressor with Decision Tree base, uses most of the standard df
def AdaBoost(df):
    # clean the data
    n = len(df)
    X = np.asarray(df[['Open','High','Low','Volume']][:n-1]) # add or subtract input data columns here
    X = X.reshape(n-1, 4)                                    # adjust '4' to match the number of input columns used above
    y = np.asarray(df[['Close']][1:])
    # build and score the model
    split = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)
    dtree = tree.DecisionTreeRegressor(max_depth=1000)
    model = ensemble.AdaBoostRegressor(n_estimators=5000, learning_rate=2.0, base_estimator=dtree)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    #print("Model1 accuracy: ", score)
    # make a prediction
    prediction = model.predict(df[['Open','High','Low','Volume']][n-1:])
    #print("AdaBoost predicted close for next candle: ", prediction)
    return prediction

# RandomForestRegressor with Decision Tree base, uses most of the standard df
def RandomForest(df):
    # clean the data
    n = len(df)
    X = np.asarray(df[['Open','High','Low','Volume']][:n-1]) # add or subtract input data columns here
    X = X.reshape(n-1, 4)                                    # adjust '4' to match the number of input columns used above
    y = np.asarray(df[['Close']][1:])
    # build and score the model
    split = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)
    model = ensemble.RandomForestRegressor(n_estimators=10000, max_depth=1000)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    #print("Model2 accuracy: ", score)
    # make a prediction
    prediction = model.predict(df[['Open','High','Low','Volume']][n-1:])
    #print("RandomForest predicted close for next candle: ", prediction)
    return prediction

"""
script that runs continuously and collects pricing and technical indicators data,
entering the data into instrument/window data tables.
"""
import sqlite3
import sqlalchemy
import pandas as pd
from datetime import datetime
from backend.pricing import history
from backend.config import currencies
from backend.indicators import sma, stoch, pivotPoints, bollBands


windows = ['M5', 'M15', 'H1', 'H4', 'D']
for instr in currencies:
    for w in windows:
        LiteCurrencyDB(instr, w)

def LiteCurrencyDB(instr, w):
    # connect to the DB
    db = sqlite3.connect('liteDB/currencies.db')
    c = db.cursor()
    table_name = "%s_%s" % (instr, w)
    # get last 500 data points
    df = history(instr, w)
    prices = df['Close']
    # set the dataframe with our technical indicators
    df['fast_sma'] = sma(prices, 6)
    df['slow_sma'] = sma(prices, 10)
    df = stoch(df, 6, 3)
    df = bollBands(df, 5, 0.5)
    pp = pd.DataFrame(data=pivotPoints(df))
    df = pd.concat([df,pp], axis=1, join_axes=[df.index])
    inputs = list(tuple(row for idx, row in df.iterrows()))
    # insert them into the dedicated table for this currency and timeframe
    c.execute('''CREATE TABLE IF NOT EXISTS {}(
                dt TEXT, Open REAL, High REAL, Low REAL, Close REAL,
                Volume INTEGER, fast_sma REAL, slow_sma REAL, Lstoch REAL,
                Hstoch REAL, K REAL, D REAL, rolling_mean REAL, boll_high REAL,
                boll_low REAL, PP REAL, R1 REAL, R2 REAL, R3 REAL, S1 REAL,
                S2 REAL, S3 REAL
                )'''.format(table_name))
    c.executemany('''INSERT INTO {}(
            dt, Open, High, Low, Close, Volume, fast_sma, slow_sma, Lstoch,
            Hstoch, K, D, rolling_mean, boll_high, boll_low, PP, R1, R2, R3,
            S1, S2, S3
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
            .format(table_name), inputs)
    db.commit()
    print('table created successfully')
    return db, df

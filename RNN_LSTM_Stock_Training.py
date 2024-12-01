import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta

data = yf.download(tickers = '^RUI', start = '2012-03-11', end = '2022-07-10')
print(data.head(10))

# Adding indicators
data['RSI'] = ta.rsi(data.Close, length = 15)
data['EMAF'] = ta.ema(data.Close, length = 20)
data['EMAM'] = ta.ema(data.Close, length = 100)
data['EMAS'] = ta.ema(data.Close, length = 150)

data['Target'] = data['Adj Close']-data.Open
data['Target'] = data['Target'].shift(-1)

data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]

data['TargetNextClose'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace=True)

data.drop(['Volume', 'Close', 'Date'], axis = 1, inplace = True)

data_set = data.iloc[:, 0:11]
pd.set_option('display.max_columns', None)

data_set.head(5)
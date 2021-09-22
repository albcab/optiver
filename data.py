"""Data processing and preprocessing"""

import pandas as pd
import numpy as np

from .utils import log_return, realized_volatility

def build_train(book_file, trade_file):
    """Build train dataset of a single stock (parallel)"""
    stock_id = book_file.split("=")[1]
    assert stock_id == trade_file.split("=")[1]
    book = pd.read_parquet(book_file)
    trade = pd.read_parquet(trade_file)
    
    #first the book
    book['wap1'] = (book['bid_price1'] * book['ask_size1'] + book['ask_price1'] * book['bid_size1']
                   ) / (book['bid_size1'] + book['ask_size1'])
    book['wap2'] = (book['bid_price2'] * book['ask_size2'] + book['ask_price2'] * book['bid_size2']
                   ) / (book['bid_size2'] + book['ask_size2'])
    book['log_return1'] = book.groupby(['time_id'])['wap1'].apply(log_return)
    book['log_return2'] = book.groupby(['time_id'])['wap2'].apply(log_return)
    book = book[~book['log_return1'].isnull()]
    train = pd.DataFrame(book.groupby(['time_id'])[['log_return1', 'log_return2']].agg(realized_volatility)).reset_index()
    
    # then the trade
    trade['log_return'] = trade.groupby(['time_id'])['price'].apply(log_return)
    trade = trade[~trade['log_return'].isnull()]
    train = train.merge(pd.DataFrame(trade.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index(),
                       how='left', on='time_id')
    
    train = train.rename(columns={'log_return1': 'vol1',
                                  'log_return2': 'vol2',
                                  'log_return': 'volt'})
    train['stock_id'] = int(stock_id)
    return train

def preprocess(df_train, logX=True, shiftX=True, scaleX=True, logy=True, shifty=True, scaley=True):
    #some times at certain stocks don't have observations in trade, remove
    df_train_ = df_train[~np.isnan(df_train['volt'])]

    if logX:
        df_train_.loc[df_train_['volt'].values == 0, 'volt'] = 1e-3
        X = np.log(df_train_[['vol1', 'vol2', 'volt']].values)
        # X = np.log(df_train_[['vol1', 'vol2']].values)
    else:
        X = df_train_[['vol1', 'vol2', 'volt']].values

    if shiftX:
        for c in range(X.shape[1]):
            X[:, c] -= X[:, c].mean()
    if scaleX:
        for c in range(X.shape[1]):
            X[:, c] /= X[:, c].std()

    if logy:
        y = np.log(df_train_['target'].values)
    else:
        y = df_train_['target'].values
    
    mean, sd = 0., 1.
    if shifty:
        mean = y.mean()
        y -= mean
    if scaley:
        sd = y.std()
        y /= sd

    if not shifty and logy:
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    return y, X, mean, sd
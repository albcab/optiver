"""Data processing and preprocessing"""

import pandas as pd
import numpy as np

from itertools import combinations

from .utils import log_return, realized_volatility

def build_log_return(book_file, unique_times, tot_sec=600):
    """Build train for Gaussian processes on log returns"""
    stock_id = book_file.split("=")[1]
    book = pd.read_parquet(book_file)
    book['wap'] = (book['bid_price1'] * book['ask_size1'] + book['ask_price1'] * book['bid_size1']
                   ) / (book['bid_size1'] + book['ask_size1'])
    book['log_return'] = book.groupby(['time_id'])['wap'].apply(log_return)
    book = book[~book['log_return'].isnull()]

    book = book.sort_values(by=['time_id', 'seconds_in_bucket'])
        
    # if fill:
    book_fill = []
    new_index = pd.Index(np.arange(1, tot_sec), name='seconds_in_bucket')
    for t in book['time_id'].unique():
        bookt = book.loc[book['time_id'] == t, ['seconds_in_bucket', 'log_return']].set_index('seconds_in_bucket').reindex(new_index, fill_value=0).reset_index()
        bookt['time_id'] = t
        book_fill.append(bookt)
    book = pd.concat(book_fill, ignore_index=True)

    if not set(book.time_id.unique()) <= set(unique_times):
        new_time_index = pd.Index(unique_times.repeat(tot_sec-1), name='time_id')
        book = book.set_index('time_id').reindex(new_time_index).reset_index()
        
    sd = book.groupby(['time_id'])['log_return'].std().reset_index()
    sd = sd.rename(columns={'log_return': 'sd'})

    book = book.merge(sd, how='left', on='time_id')
    book['log_return_norm'] = book['log_return'] / book['sd']
    book['stock_id'] = int(stock_id)
    
    time_array = book['log_return_norm'].values.reshape((len(unique_times), tot_sec-1))

    return book, time_array

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

    book['price_spread'] = (book['ask_price1'] - book['bid_price1']) / ((book['ask_price1'] + book['bid_price1']) / 2)
    book['price_spread2'] = (book['ask_price2'] - book['bid_price2']) / ((book['ask_price2'] + book['bid_price2']) / 2)
    book["bid_ask_spread"] = abs((book['bid_price1'] - book['bid_price2']) - (book['ask_price1'] - book['ask_price2']))
    book['total_volume'] = (book['ask_size1'] + book['ask_size2']) + (book['bid_size1'] + book['bid_size2'])
    book['volume_imbalance'] = abs((book['ask_size1'] + book['ask_size2']) - (book['bid_size1'] + book['bid_size2']))

    aggregator = {'log_return1': [realized_volatility],
                  'log_return2': [realized_volatility],
                  'price_spread': [np.mean],
                  'price_spread2': [np.mean],
                  'bid_ask_spread': [np.mean],
                  'total_volume': [np.mean],
                  'volume_imbalance': [np.mean]}

    train = pd.DataFrame(book.groupby(['time_id']).agg(aggregator)).reset_index()
    
    # then the trade
    trade['log_return'] = trade.groupby(['time_id'])['price'].apply(log_return)
    trade = trade[~trade['log_return'].isnull()]

    aggregator = {
        'log_return': [realized_volatility],
        'seconds_in_bucket': [lambda x: len(x.unique())],
        'size': [np.mean],
        'order_count': [np.mean],
    }

    train = train.merge(pd.DataFrame(trade.groupby(['time_id']).agg(aggregator)).reset_index(),
                       how='left', on='time_id')
    
    train['stock_id'] = int(stock_id)
    return train

def preprocess(df_train, logX=True, shiftX=True, scaleX=True, logy=True, shifty=True, scaley=True):
    #some times at certain stocks don't have observations in trade, remove
    col = df_train.columns
    df_train_ = df_train[~np.isnan(df_train[col[10]])]

    if logX:
        # df_train_.loc[df_train_[col[10]] == 0, col[10]] = 1e-3
        X = df_train_[col[3:]].values
        X[X == 0] = 1e-3
        X = np.concatenate([X , X**2], axis=1)
        all_pairs = list(combinations(np.arange(len(col)-3), 2))
        for pair in all_pairs:
            X = np.concatenate([X, X[:, pair].prod(axis=1)[:, None]], axis=1)
        X = np.log(X)
    else:
        X = df_train_[['vol1', 'vol2', 'volt']].values

    if shiftX:
        for c in range(X.shape[1]):
            X[:, c] -= X[:, c].mean()
    if scaleX:
        for c in range(X.shape[1]):
            X[:, c] /= X[:, c].std()

    y = df_train_['target'].values.copy()
    if logy:
        y = np.log(df_train_['target'].values)
    
    mean, sd = 0., 1.
    if shifty:
        mean = y.mean()
        y -= mean
    if scaley:
        sd = y.std()
        y /= sd

    if not shifty and logy:
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    return y, X, mean, sd, df_train_
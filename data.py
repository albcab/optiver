"""Data processing and preprocessing"""

import pandas as pd
import numpy as np

from jax import vmap

from .utils import log_return, realized_volatility

unique_times = np.load('unique_times.npy')

def build_log_return(book_file, tot_sec=600):
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

# def build_time_array(df, unique_times, stock_id, sd):
#     """Build array with dimension (time_ids, seconds_in_bucket)"""
#     def _build_array(time_id):
#         return (df.loc[(df.time_id == time_id) * (df.stock_id == stock_id), 'seconds in bucket']
#                 ) / sd.loc[(sd.time_id == time_id) * (sd.stock_id == stock_id), 'sd'] 
#     return vmap(_build_array)(unique_times)

# def preprocess_log_return(df):
#     """Transform log returns so that each stock on each time has sd of 1, output original sd for prediction"""
#     std_dev = df.groupby(['stock_id', 'time_id'])['log_return'].std().reset_index()
#     std_dev = std_dev.rename(columns={'log_return': 'sd'})
#     sd = lambda r: r['log_return'] / std_dev.loc[std_dev.stock_id == r['stock_id'] and std_dev.time_id == r['time_id'], 'sd']
#     df['log_return'] = df.apply(sd, axis=1)
#     return df, std_dev

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
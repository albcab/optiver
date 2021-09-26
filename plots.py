"""Helper plots"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# from utils import realized_volatility

def plot_log_returns(df, times):
    """Helper function, plots log_returns for given times"""
    plt.figure(figsize=(20, 4))
    for t in times:
        df_t = df.loc[df['time_id'] == t, ['seconds_in_bucket', 'log_return']]
        plt.plot(df_t['seconds_in_bucket'], df_t['log_return'])
        # vol = realized_volatility(df_t['log_return']) / np.sqrt(600)
        # plt.hlines([vol, -vol], xmin=np.min(df_t['seconds_in_bucket']), xmax=np.max(df_t['seconds_in_bucket']))
    plt.show()

def plot_pred(T, pred_samples, y=None, Ty=None):
    """Helper function, plot predicted samples with real values"""
    plt.figure(figsize=(20, 4))
    plt.plot(T, np.mean(pred_samples, axis=0), color='blue')
    percentiles = np.percentile(pred_samples, [5., 95.], axis=0)
    plt.fill_between(T, percentiles[0, :], percentiles[1, :], color='lightblue')

    if y is not None:
        if Ty is not None:
            plt.plot(Ty, y, color='red')
        else:
            plt.plot(T, y, color='red')
    plt.show()
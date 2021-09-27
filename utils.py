"""Utilities"""

from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import jax.numpy as jnp

le = LabelEncoder()

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

### data.build_train won't work with jnp but will with np!!!
def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))
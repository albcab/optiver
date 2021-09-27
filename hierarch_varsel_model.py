"""Hierarchical variable selection model with hierarchies per stock and per time"""

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam

import numpy as np

def model(stock_id, time_id, X, y=None):
    n_obs, n_reg = X.shape
    n_stock = len(np.unique(stock_id))
    n_time = len(np.unique(time_id))

    plate_obs = numpyro.plate('i', n_obs, dim=-1)
    plate_stock = numpyro.plate('s', n_stock, dim=-2)
    plate_reg = numpyro.plate('j', n_reg, dim=-1)
    plate_time = numpyro.plate('t', n_time, dim=None)

    scale_sd = numpyro.sample('ssd', dist.HalfCauchy(1.))
    with plate_time:
        sd = numpyro.sample('sd', dist.HalfCauchy(scale_sd))

    scale_tau = numpyro.sample('stau', dist.HalfCauchy(1.))
    with plate_stock:
        tau = numpyro.sample('tau', dist.HalfCauchy(scale_tau))

    with plate_reg:
        scale_lamda = numpyro.sample('slamda', dist.HalfCauchy(1.))
        with plate_stock:
            lamda = numpyro.sample('lamda', dist.HalfCauchy(scale_lamda))
            with numpyro.handlers.reparam(config={'beta': TransformReparam()}):
                beta = numpyro.sample(
                    'beta',
                    dist.TransformedDistribution(
                        dist.Normal(0., 1.),
                        dist.transforms.AffineTransform(0., lamda**2 * tau**2)
                    )
                )

    with plate_obs:
        numpyro.sample(
            'obs',
            dist.Normal(np.sum(X * beta[stock_id], axis=1), sd[time_id]**2)
        )
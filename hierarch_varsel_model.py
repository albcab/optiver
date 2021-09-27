"""Hierarchical variable selection model with hierarchies per stock and per time"""

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam

import numpy as np

import jax.numpy as jnp

def hcauchy_icdf(u, scale):
    return scale * jnp.tan(jnp.pi * (u - 1/2.))

def model(stock_id, X, y=None):
    n_obs, n_reg = X.shape
    n_stock = len(np.unique(stock_id))

    plate_obs = numpyro.plate('i', n_obs, dim=-1)
    plate_stock = numpyro.plate('s', n_stock, dim=-2)
    plate_reg = numpyro.plate('j', n_reg, dim=-1)

    u_sd = numpyro.sample('usd', dist.Uniform())
    sd = numpyro.deterministic('sd', hcauchy_icdf(u_sd, 1.))

    # scale_tau = numpyro.sample('stau', dist.HalfCauchy(1.))
    u_scale_tau = numpyro.sample('ustau', dist.Uniform())
    scale_tau = numpyro.deterministic('stau', hcauchy_icdf(u_scale_tau, 1.))
    with plate_stock:
        # tau = numpyro.sample('tau', dist.HalfCauchy(scale_tau))
        u_tau = numpyro.sample('utau', dist.Uniform())
        tau = numpyro.deterministic('tau', hcauchy_icdf(u_tau, scale_tau))

    with plate_reg:
        # scale_lamda = numpyro.sample('slamda', dist.HalfCauchy(1.))
        u_scale_lamda = numpyro.sample('uslamda', dist.Uniform())
        scale_lamda = numpyro.deterministic('slamda', hcauchy_icdf(u_scale_lamda, 1.))
        with plate_stock:
            # lamda = numpyro.sample('lamda', dist.HalfCauchy(scale_lamda))
            u_lamda = numpyro.sample('ulamda', dist.Uniform())
            lamda = numpyro.deterministic('lamda', hcauchy_icdf(u_lamda, scale_lamda))
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
            dist.Normal(np.sum(X * beta[stock_id, :], axis=1), sd**2),
            obs=y
        )
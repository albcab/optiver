"""Standard Bayesian Linear regression"""

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam

import numpy as np

def model(X, y=None):
    num_obs, num_reg = X.shape
    
    plate_obs = numpyro.plate('i', num_obs, dim=-1)
    plate_reg = numpyro.plate('j', num_reg, dim=-1)
    
    sigma = numpyro.sample('sigma', dist.HalfCauchy(1.))
    
    with plate_reg:
        beta = numpyro.sample('beta', dist.Normal(0., 1.))

    with plate_obs:
        numpyro.sample('obs', dist.LogNormal(np.sum(X * beta, axis=1), sigma), obs=y)

def model_log(X, y=None):
    num_obs, num_reg = X.shape
    
    plate_obs = numpyro.plate('i', num_obs, dim=-1)
    plate_reg = numpyro.plate('j', num_reg, dim=-1)
    
    sigma = numpyro.sample('sigma', dist.HalfCauchy(1.))
    
    with plate_reg:
        beta = numpyro.sample('beta', dist.Normal(0., 1.))

    with plate_obs:
        numpyro.sample('obs', dist.Normal(np.sum(X * beta, axis=1), sigma), obs=y)

def guide(X, y=None):
    _, num_reg = X.shape
    
    plate_reg = numpyro.plate('j', num_reg, dim=-1)
    
    mu_sigma = numpyro.param('mus', 0.)
    sigma_sigma = numpyro.param('sigmas', 1., constraint=dist.constraints.positive)
    numpyro.sample(
        'sigma',
        dist.TransformedDistribution(
            dist.Normal(mu_sigma, sigma_sigma),
            dist.transforms.ExpTransform()
        )
    )
    
    with plate_reg:
        mu_beta = numpyro.param('mub', 0.)
        sigma_beta = numpyro.param('sigmab', 1., constraint=dist.constraints.positive)
        with numpyro.handlers.reparam(config={'beta': TransformReparam()}):
            numpyro.sample(
                'beta',
                dist.TransformedDistribution(
                    dist.Normal(0., 1.),
                    dist.transforms.AffineTransform(mu_beta, sigma_beta)
                )
            )
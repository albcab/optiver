"""Hirearchical Linear model"""

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam

import numpy as np

def model(stock_id, X, y=None):
    num_obs, num_reg = X.shape
    num_stock = len(np.unique(stock_id))

    plate_obs = numpyro.plate('i', num_obs, dim=-1)
    plate_reg = numpyro.plate('j', num_reg, dim=-1)
    plate_stock = numpyro.plate('s', num_stock, dim=-2)

    sigma = numpyro.sample('sigma', dist.HalfCauchy(1.))

    with plate_reg:
        beta_mu = numpyro.sample('mub', dist.Normal(0., 1.)) #jnp.ones(num_reg)
        beta_sigma = numpyro.sample('sigmab', dist.HalfCauchy(1.))

        with plate_stock:
            with numpyro.handlers.reparam(config={'beta': TransformReparam()}):
                beta = numpyro.sample(
                    'beta', 
                    dist.TransformedDistribution(
                        dist.Normal(0., 1.),
                        dist.transforms.AffineTransform(beta_mu, beta_sigma)
                    )
                )
    
    with plate_obs:
        numpyro.sample('obs', dist.LogNormal(np.sum(X * beta[stock_id, :], axis=1), sigma), obs=y)

def model_log(stock_id, X, y=None):
    num_obs, num_reg = X.shape
    num_stock = len(np.unique(stock_id))

    plate_obs = numpyro.plate('i', num_obs, dim=-1)
    plate_reg = numpyro.plate('j', num_reg, dim=-1)
    plate_stock = numpyro.plate('s', num_stock, dim=-2)

    sigma = numpyro.sample('sigma', dist.HalfCauchy(1.))

    with plate_reg:
        beta_mu = numpyro.sample('mub', dist.Normal(0., 1.)) #jnp.ones(num_reg)
        beta_sigma = numpyro.sample('sigmab', dist.HalfCauchy(1.))

        with plate_stock:
            with numpyro.handlers.reparam(config={'beta': TransformReparam()}):
                beta = numpyro.sample(
                    'beta', 
                    dist.TransformedDistribution(
                        dist.Normal(0., 1.),
                        dist.transforms.AffineTransform(beta_mu, beta_sigma)
                    )
                )
    
    with plate_obs:
        numpyro.sample('obs', dist.Normal(np.sum(X * beta[stock_id, :], axis=1), sigma), obs=y)

def guide(stock_id, X, y=None):
    _, num_reg = X.shape
    num_stock = len(np.unique(stock_id))
    
    plate_reg = numpyro.plate('j', num_reg, dim=-1)
    plate_stock = numpyro.plate('s', num_stock, dim=-2)
    
    mu_sigma = numpyro.param('mus', 0.)
    sigma_sigma = numpyro.param('sigmas', 1., constraint=dist.constraints.positive)
#     with numpyro.handler.reparam(config={'sigma': TransformReparam()}):
    numpyro.sample(
        'sigma',
        dist.TransformedDistribution(
            dist.Normal(mu_sigma, sigma_sigma),
            dist.transforms.ExpTransform()
        )
    )
    
    with plate_reg:
        mu_bmu = numpyro.param('mubm', 0.)
        sigma_bmu = numpyro.param('sigmabm', 1., constraint=dist.constraints.positive)
        beta_mu = numpyro.sample('mub', dist.Normal(mu_bmu, sigma_bmu))
        
        mu_bsigma = numpyro.param('mubs', 0.)
        sigma_bsigma = numpyro.param('sigmabs', 1., constraint=dist.constraints.positive)
        beta_sigma = numpyro.sample(
            'sigmab',
            dist.TransformedDistribution(
                dist.Normal(mu_bsigma, sigma_bsigma),
                dist.transforms.ExpTransform()
            )
        )
        
        with plate_stock:
            mu_beta = numpyro.param('mub', 0.)
            sigma_beta = numpyro.param('sigmab', 1., constraint=dist.constraints.positive)
            with numpyro.handlers.reparam(config={'beta': TransformReparam()}):
                numpyro.sample(
                    'beta',
                    dist.TransformedDistribution(
                        dist.Normal(mu_beta, sigma_beta),
                        dist.transforms.AffineTransform(beta_mu, beta_sigma)
                    )
                )
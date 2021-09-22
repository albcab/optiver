"""Gaussian process regression model"""

import numpyro
import numpyro.distributions as dist

from scipy.linalg import cho_solve, cho_factor
import numpy as np

from jax import vmap
import jax.numpy as jnp

def kernel(X, Z, var, length, noise, jitter=1e-10, include_noise=True):
    norm = ((X[:, None] - Z)**2).sum(axis=2) #needs at least one d >= 2
    k = var * jnp.exp(-.5 * norm / length)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

def model(X, y=None):
    log_var = numpyro.sample('lvar', dist.Normal(0., 10.))
    log_length = numpyro.sample('llength', dist.Normal(0., 10.))
    log_noise = numpyro.sample('lnoise', dist.Normal(0., 10.))
    
    C = kernel(X, X, jnp.exp(log_var), jnp.exp(log_length), jnp.exp(log_noise))
    
    numpyro.sample(
        'obs',
        dist.TransformedDistribution(
            dist.MultivariateNormal(np.zeros(X.shape[0]), covariance_matrix=C),
            dist.transforms.ExpTransform()
        )
    )

def model_log(X, y=None):
    log_var = numpyro.sample('lvar', dist.Normal(0., 10.))
    log_length = numpyro.sample('llength', dist.Normal(0., 10.))
    log_noise = numpyro.sample('lnoise', dist.Normal(0., 10.))
    
    C = kernel(X, X, jnp.exp(log_var), jnp.exp(log_length), jnp.exp(log_noise))
    
    numpyro.sample(
        'obs',
        dist.MultivariateNormal(np.zeros(X.shape[0]), covariance_matrix=C),
        obs=y
    )

def predict(X_test, X, y, var, length, noise, jitter=1e-10):
    C_XX = kernel(X, X, var, length, noise, include_noise=True)
    C_inv = cho_solve(cho_factor(C_XX), np.eye(X.shape[0]))
    
    def predict_obs(x_test):
        k = kernel(X, x_test, var, length, noise, include_noise=False)
        return k.T @ C_inv @ y, -k.T @ C_inv @ k + (var + noise + jitter) #include_noise=True
    return vmap(predict_obs)(X_test)

def guide(X, y=None):
    numpyro.sample(
        'lvar',
        dist.Normal(numpyro.param('mulvar', 0.),
                    numpyro.param('sigmalvar', 1., constraint=dist.constraints.positive))
    )
    numpyro.sample(
        'llength',
        dist.Normal(numpyro.param('mullength', 0.),
                    numpyro.param('sigmallength', 1., constraint=dist.constraints.positive))
    )
    numpyro.sample(
        'lnoise',
        dist.Normal(numpyro.param('mulnoise', 0.),
                    numpyro.param('sigmalnoise', 1., constraint=dist.constraints.positive))
    )
    
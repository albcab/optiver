"""Gaussian process model on the log returns per time and stock"""

import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.linalg import cho_solve, cho_factor

import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam

from .utils import realized_volatility

def exp_kernel_sec(t, m, var, length, noise, jitter=1e-6, include_noise=True):
    k = var * jnp.exp(-.5 * (jnp.expand_dims(t, axis=1) - m)**2 / length)
    if include_noise:
        k += (noise + jitter) * jnp.eye(t.shape[0])
    return k

def model(T, y=None):
    log_var = numpyro.sample('lvar', dist.Normal(0., 10.))
    log_length = numpyro.sample('llength', dist.Normal(0., 10.))
    log_noise = numpyro.sample('lnoise', dist.Normal(0., 10.))

    C = exp_kernel_sec(T, T, jnp.exp(log_var), jnp.exp(log_length), jnp.exp(log_noise))

    numpyro.sample(
        'obs',
        dist.MultivariateNormal(jnp.zeros(T.shape[0]), covariance_matrix=C),
        obs=y
    )

def hierarch_model(T, y): #TODO: add stock_id
    n_times, n_sec = y.shape

    plate_times = numpyro.plate('times', n_times, dim=-1) #dim=-2

    mu_var = numpyro.sample('mvar', dist.Normal(0., 1.))
    sigma_var = numpyro.sample('svar', dist.HalfCauchy(1.))

    mu_length = numpyro.sample('mlength', dist.Normal(0., 1.))
    sigma_length = numpyro.sample('slength', dist.HalfCauchy(1.))

    mu_noise = numpyro.sample('mnoise', dist.Normal(0., 1.))
    sigma_noise = numpyro.sample('snoise', dist.HalfCauchy(1.))

    with plate_times:
        with numpyro.handlers.reparam(config={
            'lvar': TransformReparam(),
            'llength': TransformReparam(),
            'lnoise': TransformReparam()
        }):
            log_var = numpyro.sample(
                'lvar',
                dist.TransformedDistribution(
                    dist.Normal(0., 1.),
                    dist.transforms.AffineTransform(mu_var, sigma_var)
                )
            )
            log_length = numpyro.sample(
                'llength',
                dist.TransformedDistribution(
                    dist.Normal(0., 1.),
                    dist.transforms.AffineTransform(mu_length, sigma_length)
                )
            )
            log_noise = numpyro.sample(
                'lnoise',
                dist.TransformedDistribution(
                    dist.Normal(0., 1.),
                    dist.transforms.AffineTransform(mu_noise, sigma_noise)
                )
            )

    with plate_times:
        C = exp_kernel_sec(T, T, jnp.exp(log_var), jnp.exp(log_length), jnp.exp(log_noise))
        numpyro.sample(
            'obs',
            dist.MultivariateNormal(jnp.zeros(n_sec), covariance_matrix=C)
        )

def predict(T_test, T, y, var, length, noise, jitter=1e-6):
    C_yy = exp_kernel_sec(T, T, var, length, noise, jitter, include_noise=True)
    C_inv = cho_solve(cho_factor(C_yy), jnp.eye(T.shape[0]))
    C_ty = exp_kernel_sec(T_test, T, var, length, noise, jitter, include_noise=False) #not the same obs
    C_tt = exp_kernel_sec(T_test, T_test, var, length, noise, jitter, include_noise=True)
    
    return C_ty @ C_inv @ y, C_tt - C_ty @ C_inv @ C_ty.T

def samples_vol(rng, n_iter, mean, cov):
    samples = random.multivariate_normal(rng, mean, cov, shape=(n_iter, ))
    return samples, vmap(realized_volatility)(samples)

def samples_per_iter(rng, T_pred, T, y, samples, iter_per_sample):
    n_samples, _ = samples['lvar'].shape
    n_times, = T_pred.shape
    rng, keys = random.split(rng, n_samples+1)
    pred_samples, pred_vol = vmap(
        lambda key, lvar, llength, lnoise: 
            samples_vol(key, 
                        iter_per_sample, 
                        *predict(T_pred, T, y, jnp.exp(lvar), jnp.exp(llength), jnp.exp(lnoise)))
    )(keys, **samples)
    pred_samples = pred_samples.reshape((iter_per_sample * n_samples, n_times))
    pred_vol = pred_vol.reshape((iter_per_sample * n_samples, ))
    return pred_samples, pred_vol

def guide(T, y=None):
    numpyro.sample(
        'lvar',
        dist.Normal(numpyro.param('muv', 0.),
                    numpyro.param('sigmav', 1., constraint=dist.constraints.positive))
    )
    numpyro.sample(
        'llength',
        dist.Normal(numpyro.param('mul', 0.),
                    numpyro.param('sigmal', 1., constraint=dist.constraints.positive))
    )
    numpyro.sample(
        'lnoise',
        dist.Normal(numpyro.param('mun', 0.),
                    numpyro.param('sigman', 1., constraint=dist.constraints.positive))
    )

def hierarch_guide(T, y):
    n_times, _ = y.shape

    plate_times = numpyro.plate('times', n_times, dim=-1) #dim=-2

    mu_var = numpyro.sample(
        'mvar', 
        dist.Normal(numpyro.param('mmvar', 0.),
                    numpyro.param('smvar', 1., constraint=dist.constraints.positive))
    )
    sigma_var = numpyro.sample(
        'svar', 
        dist.TransformedDistribution(
            dist.Normal(numpyro.param('msvar', 0.),
                        numpyro.param('ssvar', 1., constraint=dist.constraints.positive))
        )
    )

    mu_length = numpyro.sample(
        'mlength', 
        dist.Normal(numpyro.param('mmlength', 0.),
                    numpyro.param('smlength', 1., constraint=dist.constraints.positive))
    )
    sigma_length = numpyro.sample(
        'slength', 
        dist.TransformedDistribution(
            dist.Normal(numpyro.param('mslength', 0.),
                        numpyro.param('sslength', 1., constraint=dist.constraints.positive))
        )
    )

    mu_noise = numpyro.sample(
        'mnoise', 
        dist.Normal(numpyro.param('mmnoise', 0.),
                    numpyro.param('smnoise', 1., constraint=dist.constraints.positive))
    )
    sigma_noise = numpyro.sample(
        'snoise', 
        dist.TransformedDistribution(
            dist.Normal(numpyro.param('msnoise', 0.),
                        numpyro.param('ssnoise', 1., constraint=dist.constraints.positive))
        )
    )

    with plate_times:
        with numpyro.handlers.reparam(config={
            'lvar': TransformReparam(),
            'llength': TransformReparam(),
            'lnoise': TransformReparam()
        }):
            numpyro.sample(
                'lvar',
                dist.TransformedDistribution(
                    dist.Normal(numpyro.param('mlvar', 0.),
                                numpyro.param('slvar', 1., constraint=dist.constraints.positive)),
                    dist.transforms.AffineTransform(mu_var, sigma_var)
                )
            )
            numpyro.sample(
                'llength',
                dist.TransformedDistribution(
                    dist.Normal(numpyro.param('mllength', 0.),
                                numpyro.param('sllength', 1., constraint=dist.constraints.positive)),
                    dist.transforms.AffineTransform(mu_length, sigma_length)
                )
            )
            numpyro.sample(
                'lnoise',
                dist.TransformedDistribution(
                    dist.Normal(numpyro.param('mlnoise', 0.),
                                numpyro.param('slnoise', 1., constraint=dist.constraints.positive)),
                    dist.transforms.AffineTransform(mu_noise, sigma_noise)
                )
            )
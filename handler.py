"""Multipurpose model handler"""

import numpyro
from numpyro.infer import Trace_ELBO, MCMC, NUTS, HMC, SVI, Predictive

from jax import random, devices, device_put

import numpy as np

from .utils import rmspe, r2_score

class ModelHandler(object):
    def __init__(
        self, 
        model, 
        guide=None, 
        rng_key=123, 
        Loss=Trace_ELBO, 
        Optim=numpyro.optim.Adam
    ) -> None:
        self.model = model
        self.guide = guide
        self.rng = random.PRNGKey(rng_key)
        self.loss = Loss
        self.optim = Optim
        self.svi = None
        self.result = None
        self.mcmc = None
        self.samples = None
        self.pred = None

    def run_svi(self, *data, n_iter=1000, n_particles=1, step_size=.01):
        assert self.guide is not None
        assert self.svi is None, "Reset first"
        self.svi = SVI(self.model, self.guide, self.optim(step_size), self.loss(n_particles))
        self.result = self.svi(self.rng, n_iter, *data)

    def sample_svi(self, *data, n_samples=1000, return_sites=None):
        assert self.svi is not None
        assert self.samples is None, "Reset first"
        if return_sites is not None:
            assert 'obs' in return_sites, "Dafuk you doin?"
        predictive = Predictive(self.model, params=self.result.params, num_samples=n_samples, return_sites=return_sites)
        self.samples = predictive(self.rng, *data)

    def run_mcmc(self, *data, n_warm=500, n_iter=1000, n_chains=1, nuts=True):
        assert self.mcmc is None, "Reset first"
        if nuts:
            method = NUTS(self.model)
        else:
            method = HMC(self.model)
        self.mcmc = MCMC(method, num_warmup=n_warm, num_samples=n_iter, num_chains=n_chains)
        self.mcmc.run(self.rng, *data, extra_fields=('potential_energy', ))
        pe = self.mcmc.get_extra_fields()['potential_energy']
        print('Expected log joint density: {:.2f}'.format(np.mean(-pe)))
        self.mcmc.print_summary()

    def sample_mcmc(self, *data, return_sites=None):
        assert self.mcmc is not None
        assert self.samples is None, "Reset first"
        if return_sites is not None:
            assert 'obs' in return_sites, "Dafuk you doin?"
        predictive = Predictive(self.model, self.mcmc.get_samples(), return_sites=return_sites)
        self.samples = predictive(self.rng, *data)

    def diagnose(self, y, mean=0., sd=1., log_scale=False, cpu=False):
        assert self.samples is not None
        assert self.pred is None, "Reset first"
        if cpu:
            obs = device_put(self.samples['obs'], device=devices('cpu')[0])
        else:
            obs = self.sample['obs']
        self.pred = np.mean(obs, axis=0) * sd + mean
        if log_scale:
            self.pred = np.exp(self.pred)
        R2 = round(r2_score(y_true=y, y_pred=self.pred), 3)
        RMSPE = round(rmspe(y_true=y, y_pred=self.pred), 3)
        print(f'Performance of the naive prediction: R2 score: {R2}, RMSPE: {RMSPE}')

    @property
    def get_samples(self):
        assert self.samples is not None
        return self.samples

    @property
    def get_pred(self):
        assert self.pred is not None
        return self.pred

    def reset_svi(self):
        self.svi = None
        self.result = None

    def reset_mcmc(self):
        self.mcmc = None

    def reset_samples_pred(self):
        self.samples = None
        self.pred = None

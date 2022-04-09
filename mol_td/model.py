from flax import linen as nn
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from jax.tree_util import tree_map

from .model_encdec import *
from .model_transfer import *
from .config import Config

leaky_relu = partial(nn.leaky_relu, negative_slope=.2)  # TF default


def describe_distributions(distributions):
    print('\n'.join([str(d) for d in distributions]))


def mean_r(x, y):
    return jnp.mean(jnp.sum((x-y)**2, axis=-1)**0.5)


class SimpleTDVAE(nn.Module):

    cfg: Config

    def setup(self):
        
        self.encoder = MLPEncoder(self.cfg)
        self.transfer = MLPTransfer(self.cfg)
        self.decoder = MLPDecoder(self.cfg)

        self.prior_test = MLPPrior(self.cfg)

    def __call__(self, data, test: bool=False):
        n_data, nt = data.shape[:2]

        posteriors = self.encoder(data)

        scan = nn.scan(lambda f, carry, inputs: f(carry, inputs),
                        variable_broadcast='params',
                        split_rngs=dict(params=False, sample=True),
                        in_axes=1, out_axes=1,
                        length=nt)

        initial = self.transfer.zero_state((n_data,)) if test is False else {k: v[:, 0, :] for k, v in posteriors.items()}
        
        carry = self.prior_test(initial['mean'])
        final, (priors,) = scan(self.transfer, initial, (posteriors,))

        decode = posteriors['mean'] if test is False else priors['mean']
        # print(decode.shape)

        ys = self.decoder(decode)

        # print('Dimensions: ', {'mean': mean.shape, 'std': std.shape, 'y': y.shape})

        posteriors = tfd.Independent(tfd.Normal(posteriors['mean'], posteriors['std']))
        priors = tfd.Independent(tfd.Normal(priors['mean'], priors['std']))
        output_dist = tfd.Independent(tfd.Normal(ys, self.cfg.y_std)) 
        # describe_distributions([latent_dist, output_dist, prior])
        
        data_target = data[..., :-self.cfg.n_atoms]
        nll = -jnp.mean(output_dist.log_prob(data_target))
        kl_div =  jnp.mean(posteriors.kl_divergence(priors))  # mean over the batch dimension
        loss = nll + kl_div
        ys_mean_r = mean_r(data_target, ys)

        signal = dict(loss=loss,
                      kl_div=kl_div, 
                      nll=nll,
                      priors=priors,
                      posteriors=posteriors,
                      ys=ys,
                      ys_mse=ys_mean_r,
                      final=final)

        return loss, signal




class SimpleVAE(nn.Module):

    cfg: Config

    def setup(self):

        self.encoder = MLPEncoder(self.cfg)
        self.decoder = MLPDecoder(self.cfg)

    def __call__(self, data, warm_up=jnp.array(1.0), eval=False):
        n_data, n_features_in = data.shape

        data_target = data[..., :-self.cfg.n_atoms]

        posterior = self.encoder(data, eval=eval)
        mean, std = posterior['mean'], posterior['std']
        
        posterior = tfd.Independent(tfd.Normal(mean, std))
        prior = tfd.Independent(tfd.Normal(jnp.zeros(mean.shape), jnp.ones(mean.shape)))

        # eps = prior.sample(seed=self.make_rng('sample'))
        # mean = mean + eps * std
        z = posterior.sample(seed=self.make_rng('sample'))

        y = self.decoder(z, eval=eval)   

        output_dist = tfd.Independent(tfd.Normal(y, jnp.ones(y.shape)*self.cfg.prediction_std))  # 1 is the std, when is 1 goes to mse
        # describe_distributions([latent_dist, output_dist, prior])

        nll = - 2500. * jnp.mean(output_dist.log_prob(data_target))
        kl_div = jnp.mean(posterior.kl_divergence(prior))
        
        loss = nll + warm_up * kl_div
        
        data_r = data_target.reshape((n_data, self.cfg.n_atoms, -1))[..., :3]
        data_force = data_target.reshape((n_data, self.cfg.n_atoms, -1))[..., 3:6]
        y_r = y.reshape((n_data, self.cfg.n_atoms, 6))[..., :3]
        y_force = y.reshape((n_data, self.cfg.n_atoms, 6))[..., 3:6]

        y_mean_r = mean_r(data_r, y_r)

        signal = dict(posterior_mean=mean,
                      posterior_std=std,
                      loss=loss,
                      kl_div=kl_div, 
                      nll=nll,
                      prior=prior,
                      posterior=posterior,
                      y=y,
                      y_mean_r=y_mean_r,
                      data_r=data_r,
                      data_force=data_force,
                      y_r=y_r,
                      y_force=y_force)
        
        return loss, signal



from flax import linen as nn
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from jax.tree_util import tree_map

from .model_base import *
from .model_latent import *
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

    def __call__(self, data, z_t0: jnp.ndarray=None, training: bool=False):
        n_data, nt = data.shape[:2]
        data_target = data[..., :-self.cfg.n_atoms]

        embedding = self.encoder(data)

        scan = nn.scan(lambda f, state, inputs: f(state, inputs, training=training),
                        variable_broadcast='params',
                        split_rngs=dict(params=False, sample=True, dropout=True),
                        in_axes=1, out_axes=1)

        zero_state = self.transfer.zero_state((n_data,))
        
        if training:
            z_t0 = zero_state
        else:
            if z_t0 is None:
                z_t0 = {'z': embedding[:, 0, :]}
                embedding = embedding[:, 1:, :]
                data_target = data_target[:, 1:, :]
            z_t0 = zero_state | z_t0  # lhs overwritten by lhs
            
        final_state, (prior, posterior) = scan(self.transfer, z_t0, (embedding, ))

        prior['dist'] = tfd.Normal(prior['mean'], prior['std'])
        posterior['dist'] = tfd.Normal(posterior['mean'], posterior['std'])

        y = self.decoder(posterior['z'])  # when not training, posterior = prior
        
        likelihood = tfd.Normal(y, self.cfg.y_std)
        
        nll = - 4000. * jnp.mean(likelihood.log_prob(data_target), axis=0).sum()
        kl_div =  jnp.mean(posterior['dist'].kl_divergence(prior['dist']), axis=0).sum()  # mean over the batch dimension
        loss = nll + kl_div
        
        data_r = data_target.reshape((n_data, self.cfg.n_timesteps, self.cfg.n_atoms, -1))[..., :3]
        data_force = data_target.reshape((n_data, self.cfg.n_timesteps, self.cfg.n_atoms, -1))[..., 3:6]
        y_r = y.reshape((n_data, self.cfg.n_timesteps, self.cfg.n_atoms, 6))[..., :3]
        y_force = y.reshape((n_data, self.cfg.n_timesteps, self.cfg.n_atoms, 6))[..., 3:6]

        y_mean_r = mean_r(data_r, y_r)
        
        signal = dict(posterior_std=posterior['std'].mean(),
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


def print_dict(d):
    [print(k, v.shape) for k, v in d.items()]


class HierarchicalTDVAE(nn.Module):

    cfg: Config

    def setup(self):
        
        self.encoder = MLPEncoder(self.cfg)
        self.transfer = MLPTransfer(self.cfg)
        self.decoder = MLPDecoder(self.cfg)
        self.ladder = [nn.Dense(self.cfg.n_embed) for i in range(self.cfg.n_latent)]

    def __call__(self, data, z_t0: jnp.ndarray=None, training: bool=False):

        n_data, nt = data.shape[:2]

        data_target = data[..., :-self.cfg.n_atoms]

        embedding = self.encoder(data)

        scan = nn.scan(lambda f, state, inputs: f(state, inputs, training=training),
                            variable_broadcast='params',
                            split_rngs=dict(params=False, sample=True, dropout=True),
                            in_axes=1, out_axes=1)

        priors = []
        posteriors = []                
        for latent_idx, ladder in enumerate(self.ladder):

            zero_state = self.transfer.zero_state((n_data,))
            
            if latent_idx == 0:
                zero_state_over_time = self.transfer.zero_state((n_data, nt - int(not training)))
                context = zero_state_over_time | {'z': jnp.zeros_like(zero_state_over_time['z'])}
            else:
                embedding = activations[self.cfg.latent_activation](ladder(embedding))
            
            if training:
                z_t0 = zero_state
            else:
                if z_t0 is None:
                    z_t0 = {'z': embedding[:, 0, :]}
                    embedding = embedding[:, 1:, :]
                    data_target = data_target[:, 1:, :]
                z_t0 = zero_state | z_t0  # lhs overwritten by lhs
            
            # print_dict(z_t0)
            # print_dict(context)
            # print(embedding.shape)
            final_state, (prior, posterior) = scan(self.transfer, z_t0, (embedding, context))

            # print_dict(prior)
            # print_dict(posterior)

            context = prior

            priors.append(prior)
            posteriors.append(posterior)

        for prior_tmp, posterior_tmp in zip(priors, posteriors):
            prior_tmp['dist'] = tfd.Normal(prior_tmp['mean'], prior_tmp['std'])
            posterior_tmp['dist'] = tfd.Normal(posterior_tmp['mean'], posterior_tmp['std'])
        
        y = self.decoder(posterior['z'])  # when not training, posterior = prior
        
        likelihood = tfd.Normal(y, self.cfg.y_std)
        
        nll = -10000. * jnp.mean(likelihood.log_prob(data_target), axis=0).sum()
        kl_div =  jnp.sum(jnp.array([jnp.mean(posterior['dist'].kl_divergence(prior['dist']), axis=0).sum() 
                            for prior, posterior in zip(priors, posteriors)]))  # mean over the batch dimension
        loss = nll + kl_div
        
        data_r = data_target.reshape((n_data, self.cfg.n_timesteps, self.cfg.n_atoms, -1))[..., :3]
        data_force = data_target.reshape((n_data, self.cfg.n_timesteps, self.cfg.n_atoms, -1))[..., 3:6]
        y_r = y.reshape((n_data, self.cfg.n_timesteps, self.cfg.n_atoms, 6))[..., :3]
        y_force = y.reshape((n_data, self.cfg.n_timesteps, self.cfg.n_atoms, 6))[..., 3:6]

        y_mean_r = mean_r(data_r, y_r)
        
        signal = dict(posterior_std=posterior['std'].mean(),
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



class SimpleVAE(nn.Module):

    cfg: Config

    def setup(self):

        self.encoder = MLPEncoder(self.cfg)
        self.mlp_posterior = MLPPosterior(self.cfg)
        self.decoder = MLPDecoder(self.cfg)

    def __call__(self, data, eval=False):
        n_data, n_features_in = data.shape

        data_target = data[..., :-self.cfg.n_atoms]

        embedding = self.encoder(data, eval=eval)
        posterior = self.mlp_posterior(embedding)
        mean, std = posterior['mean'], posterior['std']
        
        posterior = tfd.Independent(tfd.Normal(mean, std))
        prior = tfd.Independent(tfd.Normal(jnp.zeros(mean.shape), jnp.ones(mean.shape)))

        z = posterior.sample(seed=self.make_rng('sample'))

        y = self.decoder(z, eval=eval)   

        output_dist = tfd.Independent(tfd.Normal(y, jnp.ones(y.shape)*self.cfg.prediction_std))  # 1 is the std, when is 1 goes to mse
        # describe_distributions([latent_dist, output_dist, prior])

        nll = - 2500. * jnp.mean(output_dist.log_prob(data_target))
        kl_div = jnp.mean(posterior.kl_divergence(prior))
        loss = nll + kl_div
        
        data_r = data_target.reshape((n_data, self.cfg.n_atoms, -1))[..., :3]
        data_force = data_target.reshape((n_data, self.cfg.n_atoms, -1))[..., 3:6]
        y_r = y.reshape((n_data, self.cfg.n_atoms, 6))[..., :3]
        y_force = y.reshape((n_data, self.cfg.n_atoms, 6))[..., 3:6]

        y_mean_r = mean_r(data_r, y_r)

        # only jnp arrays come out here
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



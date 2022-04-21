from flax import linen as nn
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from typing import Tuple

from .model_base import *
from .model_latent import *
from .config import Config, enc_dec

encoders =  {'MLP': MLPEncoder,
            'GNN': GNNEncoder}
decoders = {'MLP': MLPDecoder,
            'GNN': GNNDecoder}

leaky_relu = partial(nn.leaky_relu, negative_slope=.2)  # TF default


def describe_distributions(distributions):
    print('\n'.join([str(d) for d in distributions]))


def mean_r(x, y, axis=0):
    return jnp.mean(jnp.sum((x-y)**2, axis=-1)**0.5, axis=axis)



class HierarchicalTDVAE(nn.Module):

    cfg: Config

    def setup(self):
        
        self.encoder = encoders[self.cfg.encoder](self.cfg)
        self.decoder = decoders[self.cfg.decoder](self.cfg)
        self.transfer = MLPTransfer(self.cfg)
        self.ladder = [nn.Dense(self.cfg.n_embed) for i in range(self.cfg.n_latent)]
        self.dropout = nn.Dropout(rate=self.cfg.dropout)

    def __call__(self, data: Tuple, latent_states: list=None, training: bool=False, sketch: bool=False):
        data, data_target = data
        if sketch: print('Data input shape: ', data.shape, 'Target shape: ', data_target.shape)
        n_data, nt = data.shape[:2]

        embedding = self.encoder(data, training=training)

        embeddings = []
        for latent_idx, ladder in enumerate(self.ladder):
            embedding_tmp = activations[self.cfg.latent_activation](ladder(embedding))
            # embedding = embedding_tmp if not self.cfg.skip_connections else embedding_tmp + embedding
            embedding = embedding_tmp + embedding
            embeddings.insert(0, embedding)


        jumps = [2**i for i in range(self.cfg.n_latent-1, -1, -1)]
        print(jumps)

        priors = []
        posteriors = [] 
        next_latent_states = {}              
        for latent_idx, (embedding, jump) in enumerate(zip(embeddings, jumps)):  # deeper latent space longer embeddeding function
            print('jump', jump)
            if not jump == 1:
                embedding = jnp.split(embedding, embedding.shape[1]//jump, axis=1)
                # embedding = jnp.concatenate([e for i, e in enumerate(embedding) if i % jump == 0], axis=1)
                embedding = jnp.concatenate([jnp.mean(e, axis=1, keepdims=True) for e in embedding], axis=1)
                embedding = self.dropout(embedding, deterministic=not training)
            
            nt = embedding.shape[1]
            
            print('nt', nt)
            scan = nn.scan(lambda f, state, inputs: f(state, inputs, training=training),
                       variable_broadcast='params',
                       split_rngs=dict(params=False, sample=True, dropout=True),
                       in_axes=1, out_axes=1,
                       length=nt)

            if latent_states is None:
                latent_state = self.transfer.zero_state((n_data,))  # needs z and carry
            else:
                latent_state = latent_states[f'l{latent_idx}']

            if latent_idx == 0:
                zero_state_over_time = self.transfer.zero_state((n_data, nt))
                # context = {k: jnp.zeros_like(v) for k, v in zero_state_over_time.items() if not isinstance(v, tuple)}
                context = jnp.concatenate([zero_state_over_time['z'], zero_state_over_time['z']], axis=-1)

            print(embedding.shape, context.shape, latent_state['z'].shape, latent_state['carry'].shape)
            latent_state, (prior, posterior) = scan(self.transfer, latent_state, (embedding, context))

            context = posterior['context']
            context = jnp.concatenate([jnp.repeat(x, 2, axis=1) for x in jnp.split(context, nt, axis=1)], axis=1)

            priors.append(prior)
            posteriors.append(posterior)
            next_latent_states[f'l{latent_idx}'] = latent_state

        for prior_tmp, posterior_tmp in zip(priors, posteriors):
            prior_tmp['dist'] = tfd.Normal(prior_tmp['mean'], prior_tmp['std'])
            posterior_tmp['dist'] = tfd.Normal(posterior_tmp['mean'], posterior_tmp['std'])
        

        z = posterior['z'] #if training else posterior['mean']
        y = self.decoder(z, training=training)  # when not training, posterior = prior
        if sketch: print('y shape: ', y.shape)
        
        likelihood = tfd.Normal(y, self.cfg.y_std)
        
        nll = -self.cfg.beta * jnp.mean(likelihood.log_prob(data_target), axis=0).sum()
        
        if self.cfg.likelihood_prior:
            likelihood_prior = tfd.Normal(self.decoder(prior['z'], training=training), self.cfg.y_std)
            nll = nll - self.cfg.beta * jnp.mean(likelihood_prior.log_prob(data_target), axis=0).sum()

        kl_div =  jnp.sum(jnp.array([jnp.mean(posterior['dist'].kl_divergence(prior['dist']), axis=0).sum() 
                            for prior, posterior in zip(priors, posteriors)]))  # mean over the batch dimension
        
        std_losses = [(pri['std'] - post['std'])**2 for pri, post in zip(priors, posteriors)]
        std_losses = jnp.sum(jnp.array([jnp.array([1000. * jnp.where(std_loss > 0.01, std_loss, 0.01)]).mean(0).sum() for std_loss in std_losses]))
        
        loss = nll + kl_div # + std_losses

        y_mean_r_over_time = mean_r(data_target, y, axis=(0, 2))
        y_mean_r = jnp.mean(y_mean_r_over_time, axis=0)

        signal = dict(posterior_std=posterior['std'].mean(),
                      loss=loss,
                      kl_div=kl_div, 
                      nll=nll,
                      prior=prior,
                      posterior=posterior,
                      y=y,
                      y_mean_r_over_time=y_mean_r_over_time,
                      y_mean_r=y_mean_r,
                      data_target=data_target,
                      y_r=y,
                      latent_states=next_latent_states,
                      std_loss=std_losses)

        return loss, signal



class SimpleTDVAE(nn.Module):

    cfg: Config

    def setup(self):
        
        self.encoder = MLPEncoder(self.cfg)
        self.transfer = MLPTransfer(self.cfg)
        self.decoder = MLPDecoder(self.cfg)

    def __call__(self, data, z_t0: jnp.ndarray=None, training: bool=False):
        print('Data input shape: ', data.shape)
        
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
        
        nll = - self.cfg.beta * jnp.mean(likelihood.log_prob(data_target), axis=0).sum()
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
    [print(k, v.shape) if isinstance(v, jnp.ndarray) else v for k, v in d.items()]


def quad_scale_time(x):
    x = jnp.split(x, x.shape[1], axis=1)
    x = [jnp.repeat(t, 2, axis=1) for t in x]
    return jnp.concatenate(x, axis=1)






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


models = {'SimpleVAE': SimpleVAE,
          'SimpleTDVAE': SimpleTDVAE,
          'HierarchicalTDVAE': HierarchicalTDVAE}
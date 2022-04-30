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


def compute_da(data, y):
    data0, data1 = data[:-1], data[1:]
    datar = data1 - data0
    datal = jnp.linalg.norm(datar, axis=-1, keepdims=True)
    datar = datar / datal
    y0, y1 = y[:-1], y[1:]
    yr = y1 - y0
    yl = jnp.linalg.norm(yr, axis=-1, keepdims=True)
    yr = yr / yl
    return jnp.mean(jnp.sum(yr*datar, axis=-1)), jnp.mean(datal / yl)



class HierarchicalTDVAE(nn.Module):

    cfg: Config

    def setup(self):
        
        self.encoder = encoders[self.cfg.encoder](self.cfg)
        self.decoder = decoders[self.cfg.decoder](self.cfg)
        self.transfer = MLPTransfer(self.cfg)
        self.ladder = [nn.Dense(self.cfg.n_embed) for i in range(self.cfg.n_latent)]
        self.dropout = [nn.Dropout(rate=self.cfg.dropout) for i in range(self.cfg.n_latent)]

    def __call__(self, 
                 data: Tuple, 
                 latent_states: list = None, 
                 training: bool = False, 
                 sketch: bool = False, 
                 mean_trajectory: bool = False,
                 use_obs: bool = True):
    
        data, data_target = data
        n_data, nt = data.shape[:2]
        if sketch: 
            print('Data input shape: ', data.shape, 
                  'Target shape: ', data_target.shape)

        embedding = self.encoder(data, training=training)

        embeddings = []
        for ladder, dropout in zip(self.ladder, self.dropout):
            embedding = activations[self.cfg.latent_activation](ladder(embedding)) + embedding
            # if training: embedding = dropout(embedding, deterministic=not training)
            # including this really meaningfully affects the validation error, questions to be answered! 
            embeddings.insert(0, embedding)

        if self.cfg.clockwork:
            jumps = [2**i for i in range(self.cfg.n_latent-1, -1, -1)]
        else:
            jumps = [1 for _ in range(self.cfg.n_latent)]

        priors = []
        posteriors = [] 
        next_latent_states = {}              
        for latent_idx, (embedding, jump) in enumerate(zip(embeddings, jumps)):  # deeper latent space longer embeddeding function

            if not jump == 1:
                embedding = jnp.split(embedding, embedding.shape[1]//jump, axis=1)
                embedding = jnp.concatenate([jnp.mean(e, axis=1, keepdims=True) for e in embedding], axis=1)
            
            nt = embedding.shape[1]
            
            scan = nn.scan(lambda f, state, inputs: f(state, inputs, use_obs=use_obs, mean_trajectory=mean_trajectory),
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
                context = jnp.concatenate([zero_state_over_time['z'], zero_state_over_time['z']], axis=-1)

            latent_state, (prior, posterior) = scan(self.transfer, latent_state, (embedding, context))

            context = posterior['context']
            if self.cfg.clockwork:  
                context = jnp.concatenate([jnp.repeat(x, 2, axis=1) for x in jnp.split(context, nt, axis=1)], axis=1)

            priors.append(prior)
            posteriors.append(posterior)
            next_latent_states[f'l{latent_idx}'] = latent_state

            if sketch:
                print('Embedding: ', embedding.shape,
                      'Context: ',   context.shape,
                      'Hierachy structure: ', jumps
                )

        for prior_tmp, posterior_tmp in zip(priors, posteriors):
            prior_tmp['dist'] = tfd.Normal(prior_tmp['mean'], prior_tmp['std'])
            posterior_tmp['dist'] = tfd.Normal(posterior_tmp['mean'], posterior_tmp['std'])
        
        predict_sigma = True
        y = self.decoder(posterior['z'], training=training, predict_sigma=predict_sigma)  # when not training, posterior = prior and when mean_trajectory 'z' = 'mean'
        
        
        if predict_sigma:
            y, std = y
        else:
            std = self.cfg.y_std

        if sketch: print('y shape: ', y.shape)

        likelihood = tfd.Normal(y, std)
        y = likelihood.sample(seed=self.make_rng('sample'))
        
        # nll = - self.cfg.beta * jnp.mean(likelihood.log_prob(data_target), axis=0).sum()
        nll = ((y - data_target)**2).mean(0).sum()

        kl_div =  jnp.sum(jnp.array([jnp.mean(posterior['dist'].kl_divergence(prior['dist']), axis=0).sum() 
                            for prior, posterior in zip(priors, posteriors)]))  # mean over the batch dimension
        
        loss = nll + kl_div 

        y_mean_r_over_time = mean_r(data_target, y, axis=(0, 2))
        y_mean_r = jnp.mean(y_mean_r_over_time, axis=0)

        directional_accuracy, step_size_accuracy = compute_da(data_target, y)
        y_std = jnp.mean(jnp.std(y, axis=0))

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
                      y_std=y_std,
                      latent_states=next_latent_states,
                      directional_accuracy=directional_accuracy,
                      step_size_accuracy=step_size_accuracy)

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
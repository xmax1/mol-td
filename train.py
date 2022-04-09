
from mol_td.utils import log_wandb_videos_or_images, save_pk
from mol_td.data_fns import prep_dataloaders
from mol_td import models
from mol_td.config import Config

import jax
from jax import jit
from jax import random as rnd, numpy as jnp
from flax import linen as nn
import optax
import wandb
from tqdm import tqdm
from functools import partial
from dataclasses import asdict
from tensorflow_probability.substrates.jax import distributions as tfd

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("-c", "--config", default="./configs/default_config.yaml", type=str)
parser.add_argument('--wb', action='store_true')
parser.add_argument('-i', '--id', default='', type=str)
parser.add_argument('-m', '--model', default='SimpleTDVAE', type=str)
parser.add_argument('-p', '--project', default='TimeDynamics', type=str)
parser.add_argument('-g', '--group', default='no_group', type=str)
args = parser.parse_args()

cfg = Config(**vars(args))
data = cfg.load_data('/home/amawi/projects/mol-td/data/uracil_dft.npz')
cfg.initialise_model_hype()  # can be done internally, done here to show network structure depends on data

train_loader, val_loader, test_loader = prep_dataloaders(cfg, data)

model = models[cfg.model](cfg)

rng, params_rng, sample_rng, dropout_rng = rnd.split(rnd.PRNGKey(cfg.seed), 4)
ex_batch, ex_target = next(train_loader)
params = model.init(dict(params=params_rng, sample=sample_rng, dropout=dropout_rng), ex_batch)

tx = optax.adam(learning_rate=cfg.lr)
opt_state = tx.init(params)

run = wandb.init(project=cfg.project, 
                 id=cfg.id, 
                 entity=cfg.user, 
                 config=asdict(cfg),
                 mode=cfg.wandb_status,
                 group=cfg.group)


@jit
def train_step(params, batch, opt_state, rng, warm_up):
    rng, sample_rng, dropout_rng = rnd.split(rng, 3)
    model_fwd = partial(model.apply, warm_up=warm_up, rngs=dict(sample=sample_rng, dropout=dropout_rng))
    loss_grad_fn = jax.value_and_grad(model_fwd, has_aux=True)
    (loss, signal), grads = loss_grad_fn(params, batch)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, signal, params, opt_state, rng


@jit
def validation_step(params, val_batch, rng):
    rng, sample_rng, dropout_rng = rnd.split(rng, 3)
    val_fwd = partial(model.apply, eval=True, rngs=dict(sample=sample_rng, dropout=dropout_rng))
    val_loss_batch, val_signal = val_fwd(params, val_batch)
    return val_loss_batch, val_signal, rng


def collect_signals(storage, signal):
    signal = {k:v for k,v in signal.items() if type(v) not in (tfd.Independent, list)}
    
    for k, v in signal.items():
        if type(v) == tfd.Independent:
            n_dim = 2
        else:
            n_dim = len(v.shape)
        if k in storage.keys():
            if n_dim == 0:
                storage[k] += v
            else:
                storage[k] += [v]
        else:
            if n_dim == 0:
                storage[k] = v
            else:
                storage[k] = [v]
    return storage


with run:
    for epoch in range(cfg.n_epochs):
        for batch_idx, (batch, target) in enumerate(tqdm(train_loader)):
            # warm_up = jnp.array(float(not epoch==0))
            warm_up = True
            loss, tr_signal, params, opt_state, rng = train_step(params, batch, opt_state, rng, warm_up)
            wandb.log({'loss': loss, 
                       'kl_div_tmp': tr_signal['kl_div'], # I can't even, the two plot bug only exists for name kl_div, something in cache? 
                       'nll': tr_signal['nll'],
                       'y_mean_r': tr_signal['y_mean_r']})

        train_loader.shuffle()

        if val_loader is not None:

            signals = {}
            for val_batch, target in val_loader:
                val_loss_batch, val_signal, rng = validation_step(params, val_batch, rng)
                signals = collect_signals(signals, val_signal)

            val_loader.shuffle()

            signals = {k: (jnp.concatenate(v, axis=0) 
                       if type(v) is list else v) 
                       for k, v in signals.items()}

            print(f'y_r shape {jnp.std(signals["y_r"], axis=0)[0, 0].shape}')

            save_pk(signals, 'data.pk')

            wandb.log({'val_loss_tmp': signals['loss'] / len(val_loader), 
                       'val_y_mean_r': signals['y_mean_r'] / len(val_loader),
                       'val_posterior_std': jnp.mean(signals['posterior_std']),
                       'val_y_posx_std': float(jnp.std(signals['y_r'], axis=0)[0, 0]),
                       'val_y_posy_std': float(jnp.std(signals['y_r'], axis=0)[0, 1]),
                       'val_y_posz_std': float(jnp.std(signals['y_r'], axis=0)[0, 2])
                       }, commit=False)


            media_logs = {'val_posterior_y_eval': val_signal['y'],
                          'val_ground_truth': val_batch[..., :-cfg.n_atoms],
                          'val_posterior_y': model.apply(params, val_batch, 
                                        eval=False, rngs=dict(sample=sample_rng, dropout=dropout_rng))[1]['y'],
                          'tr_ground_truth': batch[..., :-cfg.n_atoms],
                          'tr_posterior_y': tr_signal['y']}
        
            log_wandb_videos_or_images(media_logs, cfg, n_batch=3)

        

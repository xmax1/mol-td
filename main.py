
from types import NoneType
from mol_td.signal_utils import compute_rdfs, compute_rdfs_all_unique_bonds
from mol_td.utils import input_tuple, log_wandb_videos_or_images, input_bool, accumulate_signals, filter_scalars
from mol_td.utils import dict_to_yaml, yaml_to_dict, load_pk
from mol_td.data_fns import create_dataloaders, load_andor_transform_data, load_data
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
from typing import Callable

from dataclasses import dataclass


def train(cfg, 
          model,
          params, 
          rng,
          train_loader,
          val_loader=None):

    tx = optax.adam(learning_rate=cfg.lr)
    opt_state = tx.init(params)

    @jit
    def train_step(params, batch, opt_state, rng):
        rng, sample_rng, dropout_rng = rnd.split(rng, 3)
        model_fwd = partial(model.apply, 
                            training=True, 
                            rngs=dict(sample=sample_rng, dropout=dropout_rng))
        loss_grad_fn = jax.value_and_grad(model_fwd, has_aux=True)
        (loss, signal), grads = loss_grad_fn(params, batch)
        grad_norm = jnp.linalg.norm(jax.tree_leaves(jax.tree_map(jnp.linalg.norm, grads)))
        if cfg.clip_grad_norm_by:  # Clipping gradients by global norm
            scale = jnp.minimum(cfg.clip_grad_norm_by / grad_norm, 1)
            grads = jax.tree_map(lambda x: scale * x, grads)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, signal, params, opt_state, rng


    @jit
    def validation_step(params, val_batch, rng):
        warm_up_batch, eval_batch = val_batch
        rng, sample_rng, dropout_rng = rnd.split(rng, 3)
        
        val_fwd = partial(model.apply, 
                          training=False, 
                          mean_trajectory=cfg.mean_trajectory, 
                          rngs=dict(sample=sample_rng, dropout=dropout_rng))
        val_loss_batch, val_signal = val_fwd(params, warm_up_batch)

        rng, sample_rng, dropout_rng = rnd.split(rng, 3)
        val_fwd = partial(model.apply, 
                          training=False, 
                          use_obs=False, 
                          mean_trajectory=cfg.mean_trajectory, 
                          rngs=dict(sample=sample_rng, dropout=dropout_rng))
        val_loss_batch, val_signal = val_fwd(params, eval_batch, latent_states=val_signal['latent_states'])
        return val_loss_batch, val_signal, rng

    for epoch in range(cfg.n_epochs):
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            loss, tr_signal, params, opt_state, rng = train_step(params, batch, opt_state, rng)    
            wandb.log(filter_scalars(tr_signal, tag='tr_'))
        train_loader.shuffle()

        if val_loader is not None:

            signals = {}
            for val_batch in val_loader:
                val_loss_batch, val_signal, rng = validation_step(params, val_batch, rng)
                signals = accumulate_signals(signals, val_signal)
            val_loader.shuffle()
            signal = filter_scalars(signals, n_batch=len(val_loader), tag='val_')

            wandb.log(signal)

            y_mean_r_over_time = jnp.mean(jnp.stack(signals['y_mean_r_over_time'], axis=0), axis=0)
            data = [[x, y] for (x, y) in zip(range(1, y_mean_r_over_time.shape[0]+1), y_mean_r_over_time) ]
            table = wandb.Table(data=data, columns = ["x", "y"])
            wandb.log({"y_mean_r_over_time" : wandb.plot.line(table, "x", "y", title="y_mean_r_over_time")})

            val_posterior_y = model.apply(params, val_batch[0], training=True, 
                                rngs=dict(sample=sample_rng, dropout=dropout_rng))[1]['y_r']

            media_logs = {'val_posterior_y_eval': val_signal['y_r'],
                          'val_ground_truth': val_signal['data_target'],
                          'val_posterior_y': val_posterior_y,
                          'tr_ground_truth': tr_signal['data_target'],
                          'tr_posterior_y': tr_signal['y_r']}
            
            configurations = jnp.concatenate(signals['y'], axis=0).reshape(-1, cfg.n_atoms, 3)

            if cfg.media_logger is not None:
                cfg.media_logger(media_logs, cfg, n_batch=1)

            if cfg.compute_rdfs:
                val_rbfs = compute_rdfs(cfg.atoms, configurations, mode='all_unique_bonds')
                for k, v in val_rbfs.items():
                    table = wandb.Table(data=v, columns = ["x", "y"])
                    name = f'val_rbf_{k}'
                    wandb.log({name : wandb.plot.line(table, "x", "y", title=name)})

                for k, v in val_rbfs.items():
                    difference = float(jnp.mean(jnp.abs(rbfs[k][:, 1] - v[:, 1])))
                    wandb.log({f'rbf_{k}_l1norm': difference})

            if cfg.compute_energy is not None:
                energies = cfg.compute_energy(configurations)


@dataclass
class state:
    position: None
    velocity: None
    force: None
    mass: None
    nodes: None


def evaluate(cfg, warm_up_batch, n_unroll, model, params, rng):

    val_fwd = partial(model.apply, 
                      training=False, 
                      mean_trajectory=cfg.mean_trajectory, 
                      rngs=dict(sample=sample_rng, dropout=dropout_rng))
    val_loss_batch, val_signal = val_fwd(params, warm_up_batch)

    latent_states = val_signal['latent_states']
    
    @jit
    def unroll_step(params, batch, latent_states, rng):
        rng, sample_rng, dropout_rng = rnd.split(rng, 3)
        val_fwd = partial(model.apply, 
                          training=False, 
                          use_obs=False, 
                          mean_trajectory=cfg.mean_trajectory, 
                          rngs=dict(sample=sample_rng, dropout=dropout_rng))
        val_loss_batch, val_signal = val_fwd(params, batch, latent_states=latent_states)
        return val_loss_batch, val_signal, rng

    initial_info = cfg.initial_info
    data = {'R': [], 'F': [], 'V': [], 'KE': [], 'PE': [], 'TE': []}
    for i in range(n_unroll):
        val_loss_batch, val_signal, rng = unroll_step(params, warm_up_batch, latent_states, rng)
        latent_states = val_signal['latent_states']
        step_data, initial_info = cfg.evaluate_position(val_signal['y'], initial_info, **cfg.data_vars)
        for k, v in step_data.items():
            data[k] += [v]
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', default='md17/uracil', type=str)
    parser.add_argument('-l', '--load_model', default=None, type=str)
    parser.add_argument('-nf', '--node_features', default=('R', 'F', 'z'))

    parser.add_argument('--wb', action='store_true')
    parser.add_argument('-i', '--id', default='', type=str)
    parser.add_argument('-p', '--project', default='TimeDynamics', type=str)
    parser.add_argument('-g', '--group', default='junk', type=str)
    parser.add_argument('-tag', '--tag', default='no_tag', type=str)
    parser.add_argument('--xlog_media', action='store_true')

    parser.add_argument('-m', '--model', default='HierarchicalTDVAE', type=str)
    parser.add_argument('-t', '--transfer_fn', default='GRU', type=str)
    parser.add_argument('-enc', '--encoder', default='GNN', type=str)  # GCN for graph, MLP for line
    parser.add_argument('-dec', '--decoder', default='MLP', type=str)  # GCN for graph, MLP for line
    parser.add_argument('-nt', '--n_timesteps', default=4, type=int)
    parser.add_argument('-net', '--n_eval_timesteps', default=4, type=int)
    parser.add_argument('-new', '--n_eval_warmup', default=None, type=int)

    parser.add_argument('-nenc', '--n_enc_layers', default=1, type=int)
    parser.add_argument('-ndec', '--n_dec_layers', default=2, type=int)
    parser.add_argument('-tl', '--n_transfer_layers', default=2, type=int)
    parser.add_argument('-ne', '--n_embed', default=40, type=int)
    parser.add_argument('-nl', '--n_latent', default=2, type=int)
    parser.add_argument('-ystd', '--y_std', default=0.01, type=float)
    parser.add_argument('-b', '--beta', default=1., type=float)
    parser.add_argument('-lp', '--likelihood_prior', default=False, type=input_bool)
    parser.add_argument('-cw', '--clockwork', default=False, type=input_bool)
    parser.add_argument('-mj', '--mean_trajectory', default=False, type=input_bool)
    parser.add_argument('-nue', '--n_unroll_eval', default=0, type=int)

    parser.add_argument('-e', '--n_epochs', default=50, type=int)
    parser.add_argument('-bs', '--batch_size', default=128, type=int)
    parser.add_argument('-s', '--split', default=(0.8, 0.2, 0.0), type=input_tuple)
    parser.add_argument('-lr', '--lr', default=0.001, type=float)

    args = parser.parse_args()

    args = vars(args)

    eval = args['eval']
    
    if args.load_model is not None:  # | operator overwrite lhs
        loaded_cfg = yaml_to_dict(args.load_model + '/cfg.yaml') | {'dataset': args['dataset']}
        args = args | loaded_cfg

    cfg = Config(**args)

    run = wandb.init(project=cfg.project, 
                     id=cfg.id, 
                     entity=cfg.user, 
                     config=asdict(cfg),
                     mode=cfg.wandb_status,
                     group=cfg.group,
                     tags=[cfg.tag,])

    print(run.id)
    cfg.id = run.id

    data = load_andor_transform_data(cfg)
    
    train_loader, val_loader, test_loader = create_dataloaders(cfg, data, split=cfg.split, shuffle=not args.unroll_eval)

    cfg.initialise_model_hype()
    model = models[cfg.model](cfg)
    rng, params_rng, sample_rng, dropout_rng = rnd.split(rnd.PRNGKey(cfg.seed), 4)
    ex_batch = next(train_loader)
    params = model.init(dict(params=params_rng, sample=sample_rng, dropout=dropout_rng), ex_batch, training=True, sketch=True)

    if cfg.load_model is not None:
        params = load_pk(args.load_model + '/best_params.pk')
        
    with run:
        if cfg.compute_rdfs:
            rbfs = compute_rdfs(cfg.atoms, data[..., :cfg.n_dim], mode='all_unique_bonds')
            for k, v in rbfs.items():
                table = wandb.Table(data=v, columns = ["x", "y"])
                name = f'tr_rbf_{k}'
                wandb.log({name : wandb.plot.line(table, "x", "y", title=name)})

        if not args.n_unroll_eval:
            train(cfg, model, params, rng, train_loader, val_loader=val_loader)
        
        else:
            # unroll the model 
            # trainloader is simpler because it doesn't have the eval warmup
            warm_up_batch = next(train_loader)
            states = evaluate(cfg, warm_up_batch, args.n_unroll_eval, model, params, rng)

            
            


            

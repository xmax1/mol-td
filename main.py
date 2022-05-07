
from mol_td.signal_utils import compute_rdfs, compute_rdfs_all_unique_bonds
from mol_td.utils import input_tuple, input_bool, accumulate_signals, filter_scalars, save_cfg
from mol_td.utils_nojax import *
from mol_td.utils import load_cfg, load_pk, save_params, makedir
from mol_td.data_fns import create_dataloaders, load_andor_transform_data
from mol_td import models
from mol_td.config import Config, evaluate_positions, media_loggers

import jax
from jax import jit
from jax import random as rnd, numpy as jnp
import optax
import wandb
from tqdm import tqdm
from functools import partial
from dataclasses import asdict
import os
import numpy as onp

import argparse
from typing import Callable
from datetime import date

from dataclasses import dataclass

import matplotlib.pyplot as plt


def train(cfg, 
          model,
          params, 
          rng,
          train_loader,
          val_loader=None,
          test_loader=None,
          rbfs=None):

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

    best_error = 9999999.
    best_params = params
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

            if cfg.n_latent > 1:
                dts = jnp.stack(signals['mean_dts']).mean(0)
                idxs = jnp.argsort(dts)
                latent_covs = jnp.stack(signals['latent_covs'], axis=0).mean(0)
                dtwms = [jnp.dot(dts, jnp.abs(lc[idxs]).mean(-1)) for lc in latent_covs]

                for i, dtwm in enumerate(dtwms):
                    wandb.log({f"latent_{i}_dt_wmean_cov" : dtwm})


            if media_loggers[cfg.experiment] is not None:
                media_loggers[cfg.experiment](media_logs, cfg, n_batch=1)

            # if cfg.compute_energy is not None:
            #     energies = cfg.compute_energy(configurations)

            val_error = signal['val_y_mean_r']
        else:
            val_error = None
        
        ref_error = val_error if val_error is not None else tr_signal['y_mean_r']

        if ref_error < best_error:
            best_error = ref_error
            best_params = params

    save_params(best_params, cfg.run_path)

    if test_loader is not None:
        signals = {}
        for test_batch in test_loader:
            test_loss_batch, test_signal, rng = validation_step(best_params, test_batch, rng)
            signals = accumulate_signals(signals, test_signal)
        signal = filter_scalars(signals, n_batch=len(val_loader), tag='test_')

        positions = jnp.concatenate(signals['y'], axis=0).reshape(-1, cfg.n_nodes, 3)

        for k, v in signal.items():
            wandb.summary[k] = v 

        if cfg.compute_rdfs:
            val_rbfs = compute_rdfs(cfg.nodes, positions, mode='all_unique_bonds')
            for k, v in val_rbfs.items():
                table = wandb.Table(data=v, columns = ["x", "y"])
                name = f'test_rbf_{k}'
                wandb.log({name : wandb.plot.line(table, "x", "y", title=name)})

            if rbfs is not None:
                for k, v in val_rbfs.items():
                    difference = float(jnp.mean(jnp.abs(rbfs[k][:, 1] - v[:, 1])))
                    wandb.log({f'rbf_{k}_l1norm': difference})

        dts = jnp.stack(signals['mean_dts']).mean(0)
        idxs = jnp.argsort(dts)
        # print(signals['latent_covs'])
        latent_covs = jnp.stack(signals['latent_covs'], axis=0).mean(0)
        latent_covs = [lc[idxs] for lc in latent_covs]

        data = [[x, y] for (x, y) in zip(range(dts.shape[-1]), dts[idxs]) ]
        table = wandb.Table(data=data, columns = ["x", "y"])
        wandb.log({"dts" : wandb.plot.line(table, "x", "y", title="dts")})

        for i, lc in enumerate(latent_covs):
            fig = plt.figure()
            plt.imshow(lc, interpolation = 'nearest', cmap="summer")
            plt.xlabel('embedding')
            plt.ylabel('latent')
            # fig = wandb.plot.HeatMap(onp.arange(1, lc.shape[0]+1), onp.arange(1, lc.shape[1]+1), lc, show_text=False)
            wandb.log({f'latent_covs_{i}': fig })

    return best_params
            
    

@dataclass
class state:
    position: None
    velocity: None
    force: None
    mass: None
    nodes: None


def evaluate(cfg, 
             warm_up_batch, 
             n_unroll, 
             model, 
             params, 
             rng, 
             rbfs=None):

    rng, sample_rng, dropout_rng = rnd.split(rng, 3)

    val_fwd = partial(model.apply, 
                      training=False, 
                      mean_trajectory=cfg.mean_trajectory, 
                      rngs=dict(sample=sample_rng, dropout=dropout_rng))
    val_loss_batch, val_signal = val_fwd(params, warm_up_batch)
    
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
    
    data = {}
    for i in range(n_unroll):
        val_loss_batch, val_signal, rng = unroll_step(params, warm_up_batch, val_signal['latent_states'], rng)
        step_data, initial_info = evaluate_positions[cfg.experiment](cfg, val_signal['y'], initial_info)
        data = robust_dictionary_append(data, step_data)

    data = {k:jnp.concatenate(v, axis=1) for k, v in data.items()}
    video = data['R'][:, :100, ...]
    data['R'] = data['R'].reshape(-1, *data['R'].shape[2:])

    if cfg.compute_rdfs:
        val_rbfs = compute_rdfs(cfg.nodes, data['R'], mode='all_unique_bonds')
        for k, v in val_rbfs.items():
            table = wandb.Table(data=v, columns = ["x", "y"])
            name = f'eval_rbf_{k}'
            wandb.log({name : wandb.plot.line(table, "x", "y", title=name)})

        if rbfs is not None:
            for k, v in val_rbfs.items():
                difference = float(jnp.mean(jnp.abs(rbfs[k][:, 1] - v[:, 1])))
                wandb.log({f'eval_rbf_{k}_l1norm': difference})
    
    if media_loggers[cfg.experiment] is not None:
        media_loggers[cfg.experiment]({'eval_generation': video}, cfg, n_batch=1, fps=4)

    data['z'] = cfg.nodes
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', default='md17/uracil_dft', type=str)
    parser.add_argument('-rp', '--run_path', default=None, type=str)
    parser.add_argument('-nf', '--node_features', default=None, type=input_tuple)

    parser.add_argument('--wb', action='store_true')
    parser.add_argument('-i', '--id', default=None, type=str)
    parser.add_argument('-p', '--project', default='TimeDynamics_v2', type=str)
    parser.add_argument('-g', '--group', default='junk', type=str)
    parser.add_argument('-tag', '--tag', default=['no_tag',], type=input_tuple)
    parser.add_argument('--xlog_media', action='store_true')

    parser.add_argument('-m', '--model', default='HierarchicalTDVAE', type=str)
    parser.add_argument('-t', '--transfer_fn', default='GRU', type=str)
    parser.add_argument('-enc', '--encoder', default='GNN', type=str)  # GCN for graph, MLP for line
    parser.add_argument('-dec', '--decoder', default='MLP', type=str)  # GCN for graph, MLP for line
    parser.add_argument('-nt', '--n_timesteps', default=8, type=int)
    parser.add_argument('-net', '--n_eval_timesteps', default=8, type=int)
    parser.add_argument('-new', '--n_eval_warmup', default=None, type=int)

    parser.add_argument('-nenc', '--n_enc_layers', default=1, type=int)
    parser.add_argument('-ndec', '--n_dec_layers', default=1, type=int)
    parser.add_argument('-tl', '--n_transfer_layers', default=1, type=int)
    parser.add_argument('-ne', '--n_embed', default=40, type=int)
    parser.add_argument('-rcut', '--r_cutoff', default=0.5, type=float)
    parser.add_argument('-nl', '--n_latent', default=1, type=int)
    parser.add_argument('-drop', '--dropout', default=0.5, type=float)
    parser.add_argument('-ystd', '--y_std', default=0.01, type=float)
    parser.add_argument('-b', '--beta', default=1., type=float)
    parser.add_argument('-lp', '--likelihood_prior', default=False, type=input_bool)
    parser.add_argument('-cw', '--clockwork', default=False, type=input_bool)
    parser.add_argument('-mj', '--mean_trajectory', default=False, type=input_bool)
    parser.add_argument('-nue', '--n_unroll_eval', default=100, type=int)
    parser.add_argument('-lag', '--lag', default=1, type=int)

    parser.add_argument('-e', '--n_epochs', default=100, type=int)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-s', '--split', default=(0.7, 0.15, 0.15), type=input_tuple)
    parser.add_argument('-lr', '--lr', default=0.001, type=float)

    args = parser.parse_args()

    args = vars(args)

    today = date.today().strftime("%m%d")

    if args['run_path'] is not None:  # | operator overwrite lhs
        loaded_cfg = load_cfg(args['run_path'] + '/cfg.yml') | {'dataset': args['dataset']}
        args = args | loaded_cfg

    cfg = Config(**args)

    run = wandb.init(project=cfg.project, 
                     name=cfg.id, 
                     entity=cfg.user, 
                     config=asdict(cfg),
                     mode=cfg.wandb_status,
                     group=cfg.group,
                     tags=cfg.tag)

    data, targets = load_andor_transform_data(cfg)
    
    train_loader, val_loader, test_loader = create_dataloaders(cfg, data, targets, split=cfg.split, shuffle=not bool(cfg.n_unroll_eval))  # False if 0

    cfg.initialise_model_hype()

    model = models[cfg.model](cfg)
    rng, params_rng, sample_rng, dropout_rng = rnd.split(rnd.PRNGKey(cfg.seed), 4)
    ex_batch = next(train_loader)
    params = model.init(dict(params=params_rng, sample=sample_rng, dropout=dropout_rng), ex_batch, training=True, sketch=True)

    if cfg.run_path is not None:
        params = load_pk(args['run_path'] + f'/best_params.pk')  # {len(os.listdir(args["run_path"]))}
    else:
        experiment_group = today if cfg.tag is None else '/'.join(cfg.tag)
        run_dir = os.path.join('./log', cfg.experiment, experiment_group)
        if cfg.id is not None: run_dir += f'/{cfg.id}'
        makedir(run_dir)
        cfg.run_path = os.path.join(run_dir, f'run{len(os.listdir(run_dir))}')
        print('run path: ', cfg.run_path)

    save_cfg(cfg, cfg.run_path)
        
    with run:
        if cfg.compute_rdfs:
            rbfs = compute_rdfs(cfg.nodes, targets, mode='all_unique_bonds')
            for k, v in rbfs.items():
                table = wandb.Table(data=v, columns = ["x", "y"])
                name = f'tr_rbf_{k}'
                wandb.log({name : wandb.plot.line(table, "x", "y", title=name)})
        else:
            rbfs = None

        if train_loader is not None:
            params = train(cfg, model, params, rng, train_loader, val_loader=val_loader, test_loader=test_loader, rbfs=rbfs)
        
        if cfg.n_unroll_eval:
            train_loader.shuffle()
            warm_up_batch = next(train_loader)  # trainloader is simpler because it doesn't have the eval warmup
            states = evaluate(cfg, warm_up_batch, cfg.n_unroll_eval, model, params, rng, rbfs=rbfs)
            print(states['R'].shape)
            [print(v.shape) for k, v in cfg.initial_info.items()]
            onp.savez(cfg.run_path + '/eval_positions.npz', **states)

            x = onp.load(cfg.run_path + '/eval_positions.npz')
            print(list(x.keys()))
            [print(v.shape) for k, v in x.items()]

            
            


            

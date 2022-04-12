
from types import NoneType
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
parser.add_argument('-p', '--project', default='TimeDynamics', type=str)
parser.add_argument('-g', '--group', default=None, type=str)
parser.add_argument('-tag', '--tag', default='no_tag', type=str)
parser.add_argument('--xlog_media', action='store_true')

parser.add_argument('-m', '--model', default='HierarchicalTDVAE', type=str)
parser.add_argument('-t', '--transfer_fn', default='GRU', type=str)
parser.add_argument('-nt', '--n_timesteps', default=10, type=int)

parser.add_argument('-el', '--n_enc_layers', default=2, type=int)
parser.add_argument('-dl', '--n_dec_layers', default=2, type=int)
parser.add_argument('-tl', '--n_transfer_layers', default=1, type=int)
parser.add_argument('-ne', '--n_embed', default=20, type=int)
parser.add_argument('-nl', '--n_latent', default=2, type=int)
parser.add_argument('-y_std', '--y_std', default=1., type=float)
parser.add_argument('-b', '--beta', default=1000., type=int)
parser.add_argument('--skip_connections', action='store_true')
parser.add_argument('--post_into_prior', action='store_true')
parser.add_argument('--likelihood_prior', action='store_true')


parser.add_argument('-e', '--n_epochs', default=10, type=int)
parser.add_argument('-bs', '--batch_size', default=128, type=int)
parser.add_argument('-lr', '--lr', default=0.001, type=float)


args = parser.parse_args()

cfg = Config(**vars(args))
data = cfg.load_data('./data/uracil_dft.npz')
cfg.initialise_model_hype()  # can be done internally, done here to show network structure depends on data

n_data = len(data)
n_steps = 1000

run = wandb.init(project='TimeDynamics', 
                 id=cfg.id, 
                 entity=cfg.user,
                 mode=cfg.wandb_status,
                 group='data_analysis',
                 tags=['uracil',])

with run:
    media_logs = {'early_vid_1000steps': data[:n_steps][..., :-cfg.n_atoms][None, ...],
                'mid_vid_1000steps': data[(n_data//2):(n_data//2)+n_steps][..., :-cfg.n_atoms][None, ...],
                'end_vid_1000steps':data[-n_steps:][..., :-cfg.n_atoms][None, ...]}


    log_wandb_videos_or_images(media_logs, cfg, n_batch=1, fps=20)



        

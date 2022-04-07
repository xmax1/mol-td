
from mol_td.utils import log_video
from mol_td.data_fns import prep_dataloaders
from mol_td import models
from mol_td.config import Config

import jax
from jax import jit
from jax import random as rnd
from flax import linen as nn
import optax
import wandb
from tqdm import tqdm
from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="./configs/default_config.yaml", type=str)
parser.add_argument('--wb', action='store_true')
parser.add_argument('-n', '--name', default='', type=str)
args = parser.parse_args()

cfg = Config(use_wandb=args.wb, id=args.name)
data = cfg.load_data('/home/amawi/projects/mol-td/data/uracil_dft.npz')
cfg.initialise_model_hype()  # can be done internally, done here to show network structure depends on data

train_loader, val_loader, test_loader = prep_dataloaders(cfg, data)

model = models[cfg.model](cfg)

rng, params_rng, sample_rng = rnd.split(rnd.PRNGKey(cfg.seed), 3)
ex_batch, ex_target = next(train_loader)
params = model.init(dict(params=params_rng, sample=sample_rng), ex_batch)

tx = optax.sgd(learning_rate=cfg.lr)
opt_state = tx.init(params)

model_fwd = partial(model.apply, rngs=dict(sample=sample_rng))
loss_grad_fn = jit(jax.value_and_grad(model_fwd, has_aux=True))
fwd = jit(model_fwd)
val_test = fwd(params, ex_batch)

run = wandb.init(project=cfg.project, 
                 id=cfg.id, 
                 entity=cfg.user, 
                 config=cfg,
                 mode=cfg.wandb_status)

with run:
    for epoch in range(cfg.n_epochs):
        for batch, target in tqdm(train_loader):
            
            (loss, signal), grads = loss_grad_fn(params, batch)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            wandb.log({'loss': loss, 
                       'kl_div_tmp': signal['kl_div'], # I can't even, the two plot bug only exists for name kl_div, something in cache? 
                       'nll': signal['nll'],
                       'predictions_mse': signal['predictions_mse']})

        train_loader.shuffle()

        if val_loader is not None:
            val_loss_overall = 0.
            val_mse_overall = 0.

            for batch, target in val_loader:
                val_loss, signal = fwd(params, batch, test=True)
                
                val_loss_overall += val_loss
                val_mse_overall += signal['predictions_mse']

            wandb.log({'val_loss_tmp': val_loss_overall / len(val_loader), 
                       'val_mse': val_mse_overall / len(val_loader),})
                    #    'epoch': epoch})

        log_video(batch[0, :, :-cfg.n_atoms], 'data', atoms=batch[0,0,-cfg.n_atoms:])
        log_video(fwd(params, batch)[1]['predictions'][0], 'posterior_prediction', atoms=batch[0,0,-cfg.n_atoms:])
        log_video(signal['predictions'][0], 'prior_prediction', atoms=batch[0,0,-cfg.n_atoms:]) 

        

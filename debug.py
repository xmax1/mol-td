from mol_td.utils import load_config, name_run, log_video
from mol_td.data_fns import load_data, prep_data, get_split, prep_dataloaders
from mol_td import models

import jax
from jax import jit
from typing import Any, Callable, Sequence, Optional
from jax import lax, random as rnd, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import optax
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm


# experiment configuration
cfg = load_config('/home/amawi/projects/mol-td/configs/default_config.yaml')

# load and prep the data
data, raw_data = load_data('/home/amawi/projects/mol-td/data/uracil_dft.npz')
train_loader, val_loader, test_loader = prep_dataloaders(cfg, data)

# initialise the model
model = models[cfg['MODEL']['model']](cfg['MODEL'])

rng, video_rng, params_rng, sample_rng = rnd.split(rnd.PRNGKey(cfg['seed']), 4)
ex_batch = next(train_loader)
params = model.init(dict(params=params_rng, sample=sample_rng), ex_batch)

tx = optax.sgd(learning_rate=cfg['TRAIN']['lr'])
opt_state = tx.init(params)

loss_grad_fn = jit(jax.value_and_grad(model.apply, has_aux=True))
fwd = jit(model.apply)

run = wandb.init(project=cfg['WANDB']['project'], 
                 id=name_run(cfg)['WANDB']['id'], 
                 entity=cfg['WANDB']['user'], 
                 config=cfg['TRAIN']) if cfg['WANDB']['track'] else None

with run:
    for epoch in range(cfg['TRAIN']['n_epochs']):
        for batch in tqdm(train_loader, desc='training'):
            
            (loss, signal), grads = loss_grad_fn(params, batch)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            wandb.log({'loss': loss, 
                    'kl_div': signal['kl_div'], 
                    'nll': signal['nll']})

        train_loader.shuffle()

        if val_loader is not None:
            for batch_idx, batch in enumerate(val_loader):
                val_loss, signal = fwd(params, batch)
                
            wandb.log({'val_loss': loss, 'epoch': epoch})

        if cfg['TRAIN']['predict_video']:
            log_video(batch, 'data_video')
            log_video(signal['prediction'], 'prediction_video')

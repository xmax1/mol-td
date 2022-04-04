from .utils import load_config
import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random as rnd, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import optax
import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset



def get_split(n_data, data_seed=1, split=(0.7, 0.15, 0.15)):
    
    key = rnd.PRNGKey(data_seed)
    split_idxs = jnp.arange(n_data)
    split_idxs = rnd.shuffle(key, split_idxs)

    n_train, n_val, n_test = (
        int(n_data * split[0]),
        int(n_data * split[1]),
        int(n_data) * split[2]
    )

    train_idxs = split_idxs[:n_train]
    val_idxs = split_idxs[n_train:n_train + n_val]
    test_idxs = split_idxs[-n_test:]

    return train_idxs, val_idxs, test_idxs


def prep_data(data, labels, batch_size=32, data_seed=1):
    
    train_idxs, val_idxs, test_idxs = get_split(len(data), seed=data_seed)

    train_dataset = TensorDataset(*[jnp.array(input) for input in [data[train_idxs], labels[train_idxs]]],)
    val_dataset = TensorDataset(*[jnp.array(input) for input in [data[val_idxs], labels[val_idxs]]], )
    test_dataset = TensorDataset(*[jnp.array(input) for input in [data[test_idxs], labels[test_idxs]]], )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model, params, train_loader, val_loader=None, cfg=None):

    if cfg is None: cfg = load_config()  # loads default

    run = wandb.init(project=cfg['PATHS']['results'], entity=cfg['WANDB']['user'], config=cfg['TRAIN'])

    loss_grad_fn = jax.value_and_grad(loss_fn)

    tx = optax.sgd(learning_rate=cfg['TRAIN']['lr'])
    opt_state = tx.init(params)

    for epoch in range(cfg['TRAIN']['n_epochs']):
        for batch_idx, (batch, target) in enumerate(train_loader):
            
            loss, grads = loss_grad_fn(params, batch, target)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            wandb.log({'loss': loss})

            # indicators TODO

        if val_loader is not None:
            for batch_idx, (batch, target) in enumerate(train_loader):

                val_loss = loss_fn(batch)

    run.finish()

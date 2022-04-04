from .utils import load_config
import jax
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


# class DataLoader():
#     def __init__():

#     def 




def load_data(path):
    raw_data = np.load(path)
    # print(tuple(raw_data.keys()))
    forces, coords = raw_data['F'], raw_data['R']
    n_data, n_atoms, _ = forces.shape
    atoms = raw_data['z'][None, :, None].repeat(n_data, axis=0)
    print(forces.shape, coords.shape, atoms.shape)
    data = np.concatenate([forces, coords, atoms], axis=-1)
    data = np.reshape(data, (n_data, n_atoms * 7))
    return data, raw_data


def get_split(n_data, data_seed=1, split=(0.7, 0.15, 0.15)):
    
    key = rnd.PRNGKey(data_seed)
    split_idxs = jnp.arange(n_data)
    split_idxs = rnd.shuffle(key, split_idxs)

    n_train, n_val, n_test = (
        int(n_data * split[0]),
        int(n_data * split[1]),
        int(n_data * split[2])
    )

    train_idxs = split_idxs[:n_train]
    val_idxs = split_idxs[n_train:n_train + n_val]
    test_idxs = split_idxs[-n_test:]

    return train_idxs, val_idxs, test_idxs


def cut_remainder(idxs, batch_size):
    n_data = len(idxs)
    n_batches = n_data // batch_size
    n_remainder = n_data - n_batches * batch_size 
    return idxs[:-n_remainder]


def create_loader(data, target, idxs, batch_size):
    n_batches, remainder = divmod(len(data), batch_size)
    return zip(data[idxs].reshape((n_batches, data.shape[1:])), target[idxs].reshape((n_batches, target.shape[1:])))
    


# def prep_data(data, target,  batch_size=32, data_seed=1):
#     bs = batch_size

#     train_idxs, val_idxs, test_idxs = get_split(len(data), data_seed=data_seed)
#     # train_idxs, val_idxs, test_idxs = map(cut_remainder, idxs, [batch_size for _ in len(idxs)])

#     n_batches = len(train_idxs)//batch_size
#     train_loader = [(data[train_idxs][i*bs:(i+1)*bs], target[train_idxs][i*bs:(i+1)*bs]) for i in range(n_batches)]

#     n_batches = len(val_idxs)//batch_size
#     val_loader = [(data[train_idxs][i*bs:(i+1)*bs], target[train_idxs][i*bs:(i+1)*bs]) for i in range(n_batches)]
    
#     n_batches = len(test_idxs)//batch_size
#     test_loader = [(data[train_idxs][i*bs:(i+1)*bs], target[train_idxs][i*bs:(i+1)*bs]) for i in range(n_batches)]

#     return train_loader, val_loader, test_loader


def prep_data(data, target,  batch_size=32, data_seed=1):

    train_idxs, val_idxs, test_idxs = get_split(len(data), data_seed=data_seed)

    train_loader = create_loader(data, target, train_idxs, batch_size)
    val_loader = create_loader(data, target, val_idxs, batch_size)
    test_loader = create_loader(data, target, test_idxs, batch_size)

    return train_loader, val_loader, test_loader





class DataLoader():
    def __init__(self, 
                 data, 
                 target, 
                 batch_size):
        
        self.batch_size = batch_size
        self.data = data
        self.target = target

    def __next__():
        i += 1
        return self.data[i*self.batch_size:(i+1)*self.batch_sizes]





def prep_data_torch(data, target, batch_size=32, data_seed=1):
    
    idxs = get_split(len(data), data_seed=data_seed)
    train_idxs, val_idxs, test_idxs = map(np.array, idxs)

    train_dataset = TensorDataset(data[train_idxs], target[train_idxs])
    # val_dataset = TensorDataset(*[jnp.array(input) for input in [data[val_idxs], target[val_idxs]]], )
    # test_dataset = TensorDataset(*[jnp.array(input) for input in [data[test_idxs], target[test_idxs]]], )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader
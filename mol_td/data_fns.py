import jax
from jax import random as rnd, numpy as jnp
from functools import reduce
import jax.numpy as jnp 
import jax 
from .config import Config


def normalise(data, new_min=-1, new_max=1):
    data = ((data - jnp.min(data)) / (jnp.max(data) - jnp.min(data))) * (new_max - new_min) + new_min
    return data


def get_split(n_data, seed=1, split=(0.7, 0.15, 0.15)):
    
    key = rnd.PRNGKey(seed)
    split_idxs = jnp.arange(n_data)
    split_idxs = rnd.permutation(key, split_idxs, independent=True)

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
    n_batches, remainder = divmod(len(idxs), batch_size)
    return zip(data[idxs[:(n_batches*batch_size)]].reshape((n_batches, batch_size, data.shape[-1])), 
                target[idxs[:(n_batches*batch_size)]].reshape((n_batches, batch_size, target.shape[-1])))


def prep_dataloaders(cfg, data, split=(0.7, 0.15, 0.15)):
    n_data = len(data)
    # train_idxs, val_idxs, test_idxs = get_split(len(data), seed=cfg.seed)
    
    # train_data, val_data, test_data = Dataset(data[train_idxs]), Dataset(data[val_idxs]), Dataset(data[test_idxs])

    n_train, n_val, n_test = (
        int(n_data * split[0]),
        int(n_data * split[1]),
        int(n_data * split[2])
    )

    train_data, val_data, test_data = data[:n_train], data[n_train:n_train+n_val], data[-n_test:]

    train_loader = DataLoader(cfg, train_data)
    val_loader = DataLoader(cfg, val_data)
    test_loader = DataLoader(cfg, test_data)

    return train_loader, val_loader, test_loader


class DataLoader:
    
    def __init__(self, 
        cfg: Config,
        data: jnp.array,
        path: str=None,
        shuffle: bool=True
    ):
        """Naive dataloader implementation with support for 
        - shuffeling with explicit key, 
        - dropping the last batch if it is incomplete,
        - parallelism using `jax.vmap`.
        Args:
            data (object): Class that implements `__len__` and `__call__(idx)`
            batch_size (Union[int, Sequence[int]]): Batch size 
            shuffle (bool, optional): If `False` calling `dataloader.shuffle(key)` is a null-operation. Defaults to True.
            drop_last_batch (bool, optional): If `True` last batch is dropped if its batch size would be incomplete. Defaults to True.
        Raises:
            Exception: Requested batch size exceeds data-length

        Thieved from https://github.com/SimiPixel/jax_dataloader
        """

        self.batch_size = cfg.batch_size
        self.seed = cfg.seed
        self.n_timesteps = cfg.n_timesteps
        self.n_timesteps_eval = cfg.n_timesteps_eval

        ### PREPARATION OF THE DATA AND TARGETS ###
        n_data = len(data)

        if self.n_timesteps:
            n_data, remainder = divmod(n_data, self.n_timesteps)
            data = data[:-remainder] if remainder > 0 else data
            data = data.reshape((n_data, self.n_timesteps, *data.shape[1:]))
            
        self.data = data
        self.target = data[..., :-cfg.n_atoms]
        
        # Shuffle stuff
        self._slices = [slice(i,i+self.batch_size) for i in range(0,n_data,self.batch_size)]
        self._shuffle = shuffle
        self._order = jnp.arange(0, n_data)
        self.key = rnd.PRNGKey(self.seed)
        self.key, subkey = jax.random.split(self.key)
        self._order = jax.random.permutation(subkey, self._order)

        self._exhausted_batches = 0 
    
    
    def shuffle(self, key=None, returns_new_key=False, reset=True):

        if reset:
            self._exhausted_batches = 0
            
        if key is None:
            self.key, subkey = jax.random.split(self.key)

        if self._shuffle:
            self._order = jax.random.permutation(subkey, self._order)

        if returns_new_key:
            return self.key

    def __len__(self):
        return len(self._slices)

    def __iter__(self):
        self._exhausted_batches = 0
        return self 

    def __next__(self):
        
        i = self._exhausted_batches

        if i < len(self):
            batch = self.data[(self._order[self._slices[i]],)]
            target = self.target[(self._order[self._slices[i]],)]
            self._exhausted_batches += 1
            return batch, target
        else:
            raise StopIteration
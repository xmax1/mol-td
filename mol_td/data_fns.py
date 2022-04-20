import jax
from jax import random as rnd, numpy as jnp
from functools import reduce
import jax.numpy as jnp 
import jax 
from .config import Config


def transform(data, new_min=-1, new_max=1):
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


def create_loader(data, target, idxs, batch_size):
    n_batches, remainder = divmod(len(idxs), batch_size)
    return zip(data[idxs[:(n_batches*batch_size)]].reshape((n_batches, batch_size, data.shape[-1])), 
                target[idxs[:(n_batches*batch_size)]].reshape((n_batches, batch_size, target.shape[-1])))


def cut_remainder(data, n_batch):
    n_batch_time, remainder = divmod(data.shape[0], n_batch)
    data = data[:-remainder] if remainder > 0 else data
    return data


def filter_data_with_mechanism_for_including_first_data_point(data, idxs, batch_size):
    tr_idxs, val_idxs, test_idxs = idxs

    val_idxs = jnp.delete(val_idxs, jnp.where(val_idxs==0)[0])  # remove if first one! 
    test_idxs = jnp.delete(test_idxs, jnp.where(test_idxs==0)[0])  # remove if first one! 
    initial_states_val = jnp.expand_dims(data[(val_idxs - 1), -1, ...], axis=1)
    initial_states_test = jnp.expand_dims(data[(test_idxs - 1), -1, ...], axis=1)

    tr_data = cut_remainder(data[tr_idxs], batch_size)
    val_data = cut_remainder(jnp.concatenate([initial_states_val, data[val_idxs]], axis=1), batch_size)
    test_data = cut_remainder(jnp.concatenate([initial_states_test, data[test_idxs]], axis=1), batch_size)
    return tr_data, val_data, test_data


def filter_data_with_mechanism_for_including_first_data_point(data, idxs, batch_size):
    tr_idxs, val_idxs, test_idxs = idxs

    val_idxs = jnp.delete(val_idxs, jnp.where(val_idxs==0)[0])  # remove if first one! 
    test_idxs = jnp.delete(test_idxs, jnp.where(test_idxs==0)[0])  # remove if first one! 
    initial_states_val = jnp.expand_dims(data[(val_idxs - 1), -1, ...], axis=1)
    initial_states_test = jnp.expand_dims(data[(test_idxs - 1), -1, ...], axis=1)

    tr_data = cut_remainder(data[tr_idxs], batch_size)
    val_data = cut_remainder(jnp.concatenate([initial_states_val, data[val_idxs]], axis=1), batch_size)
    test_data = cut_remainder(jnp.concatenate([initial_states_test, data[test_idxs]], axis=1), batch_size)
    return tr_data, val_data, test_data



def filter_data(data, idxs, batch_size):
    tr_idxs, val_idxs, test_idxs = idxs

    tr_data = cut_remainder(data[tr_idxs], batch_size)
    val_data = cut_remainder(data[val_idxs], batch_size)
    test_data = cut_remainder(data[test_idxs], batch_size)

    return tr_data, val_data, test_data


def split_into_timesteps(data, n_timesteps):
    data = cut_remainder(data, n_timesteps)
    n_trajectories = data.shape[0]//n_timesteps
    data = data.reshape(n_trajectories, n_timesteps, *data.shape[1:])
    return data


def prep_neval_eq_ntr(cfg, split, data, target):
    data = split_into_timesteps(data, cfg.n_timesteps)
    target = split_into_timesteps(target, cfg.n_timesteps)
    n_trajectories = len(data)

    n_train, n_val, n_test = (
        int(n_trajectories * split[0]),
        int(n_trajectories * split[1]),
        int(n_trajectories * split[2])
    )

    key = rnd.PRNGKey(cfg.seed)
    idxs = rnd.permutation(key, jnp.arange(0, n_trajectories))
    tr_idxs, val_idxs, test_idxs = idxs[:n_train], idxs[n_train:(n_val+n_train)], idxs[-n_test:]

    val_idxs = jnp.delete(val_idxs, jnp.where(val_idxs==0)[0])  # remove if first one! 
    test_idxs = jnp.delete(test_idxs, jnp.where(test_idxs==0)[0])  # remove if first one! 
    initial_states_val_data = data[(val_idxs - 1), -cfg.n_eval_warmup:, ...]
    initial_states_test_data = data[(test_idxs - 1), -cfg.n_eval_warmup:, ...]
    initial_states_val_target = target[(val_idxs - 1), -cfg.n_eval_warmup:, ...]
    initial_states_test_target = target[(test_idxs - 1), -cfg.n_eval_warmup:, ...]

    tr_data = cut_remainder(data[tr_idxs], cfg.batch_size)
    val_data = cut_remainder(jnp.concatenate([initial_states_val_data, data[val_idxs]], axis=1), cfg.batch_size)
    test_data = cut_remainder(jnp.concatenate([initial_states_test_data, data[test_idxs]], axis=1), cfg.batch_size)

    tr_target = cut_remainder(target[tr_idxs], cfg.batch_size)
    val_target = cut_remainder(jnp.concatenate([initial_states_val_target, target[val_idxs]], axis=1), cfg.batch_size)
    test_target = cut_remainder(jnp.concatenate([initial_states_test_target, target[test_idxs]], axis=1), cfg.batch_size)

    print(f'Datasets length: Train {len(tr_data)} Val {len(val_data)} Test {len(test_data)}')
    print(f'Some idxs:  \n Train {tr_idxs[:5]} \n Val {val_idxs[:5]}')

    return (tr_data, tr_target), (val_data, val_target), (test_data, test_target)


def prep_dataloaders(cfg, data, target, split=(0.7, 0.15, 0.15)):
    n_data = len(data)

    n_train, n_val, n_test = (
        int(n_data * split[0]),
        int(n_data * split[1]),
        int(n_data * split[2])
    )

    # split into timesteps
    if cfg.n_timesteps:
        # max_timestep = max(cfg.n_timesteps, cfg.n_eval_timesteps+cfg.n_eval_warmup)

        # key = rnd.PRNGKey(cfg.seed)
        # idxs = rnd.permutation(key, jnp.arange(0, n_data))
        # all_idxs = idxs[:n_train], idxs[n_train:(n_val+n_train)], idxs[-n_test:]
        # tr_data, val_data, test_data = filter_data(data, all_idxs, cfg.batch_size)
        # tr_target, val_target, test_target = filter_data(target, all_idxs, cfg.batch_size)

        # tr_data = split_into_timesteps(tr_data, cfg.n_timesteps)
        # tr_target = split_into_timesteps(tr_target, cfg.n_timesteps)

        # n_eval = cfg.n_eval_timesteps + cfg.n_eval_warmup
        # val_data = split_into_timesteps(val_data, n_eval)
        # val_data = split_into_timesteps(val_data, n_eval)

        # test_data = split_into_timesteps(test_data, n_eval)
        # test_target = split_into_timesteps(test_target, n_eval)

        tr, val, test = prep_neval_eq_ntr(cfg, split, data, target)

        
    

        
        # tr_slice, val_slice, test_slice = slice(0, n_train), slice(n_train, n_train+n_val), slice(n_train+n_val, n_train+n_val+n_test)
        # tr_idxs, val_idxs, test_idxs = idxs[tr_slice], idxs[val_slice], idxs[test_slice]
        # all_idxs = idxs[:n_train], idxs[n_train:(n_val+n_train)], idxs[-n_test:]
        
        

        
    else:

        key = rnd.PRNGKey(cfg.seed)
        data = rnd.permutation(key, data, axis=0)
        data = data[:-divmod(data.shape[0], cfg.batch_size)[1]] # remove not full batch
        tr_data, val_data, test_data = data[:n_train], data[n_train:(n_train+n_val)], data[-n_test:]

    train_loader = DataLoader(cfg, *tr)
    val_loader = DataLoader(cfg, *val, eval=True)
    test_loader = DataLoader(cfg, *test, eval=True)

    return train_loader, val_loader, test_loader


class DataLoader:
    
    def __init__(self, 
        cfg: Config,
        data: jnp.array,
        target: jnp.array,
        shuffle: bool=True, 
        eval: bool=False
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
        self.n_eval_timesteps = cfg.n_eval_timesteps
        self.eval = eval
        
        self.data = data
        self.target = target
        
        self._shuffle = shuffle
        self._order = jnp.arange(0, data.shape[0])
        self.key = rnd.PRNGKey(self.seed)
        self.key, subkey = rnd.split(self.key)
        self._order = rnd.permutation(subkey, self._order)

        self._exhausted_batches = 0 
    
    
    def shuffle(self, key=None, returns_new_key=False, reset=True):
        
        self.key, subkey = rnd.split(self.key)
        self._order = rnd.permutation(subkey, self._order)

        if reset:
            print('Resetting dataloader... ')
            self._exhausted_batches = 0

        if returns_new_key:
            return self.key

    def __len__(self):
        return len(self._order)//self.batch_size

    def __iter__(self):
        self._exhausted_batches = 0
        return self 

    def __next__(self):
        
        i = self._exhausted_batches
        start = i * self.batch_size
        stop  = (i+1) * self.batch_size

        if i < len(self):
            batch = self.data[(self._order[start:stop],)]
            target = self.target[(self._order[start:stop],)]
            if self.eval:
                batch_warmup, batch_eval = jnp.split(batch, [batch.shape[1]-self.n_eval_timesteps,], axis=1)
                target_warmup, target_eval = jnp.split(target, [target.shape[1]-self.n_eval_timesteps,], axis=1)
                self._exhausted_batches += 1
                return ((batch_warmup, target_warmup), (batch_eval, target_eval))
            self._exhausted_batches += 1
            return batch, target
        else:
            raise StopIteration
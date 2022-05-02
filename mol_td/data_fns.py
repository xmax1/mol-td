import jax
from jax import numpy as jnp, random as rnd, jit, vmap
import jax.numpy as jnp 
from .config import Config

from jax_md.partition import neighbor_list, NeighborListFormat
from jax_md import space

import numpy as onp
import jraph


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

def untransform(data, old_min, old_max, new_min=-1, new_max=1, mean=None):
    data = ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    if mean is not None:
        data = data + mean
    return data

def transform(data, old_min, old_max, new_min=-1, new_max=1, mean=None):
    if mean is not None:
        data = data - mean
    data = ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return data

def cut_remainder(data, n_batch):
    n_batch_time, remainder = divmod(data.shape[0], n_batch)
    data = data[:-remainder] if remainder > 0 else data
    return data

def split_into_timesteps(data, n_timesteps):
    data = cut_remainder(data, n_timesteps)
    n_trajectories = data.shape[0]//n_timesteps
    data = data.reshape(n_trajectories, n_timesteps, *data.shape[1:])
    return data

def get_stats(data, axis=0):
    if len(data.shape) == 3:
        data = data.reshape(-1, data.shape[-1])
    return jnp.min(data, axis=axis), jnp.max(data, axis=axis), jnp.mean(data, axis=axis)

def compute_edges(positions, receivers, senders):
        return jnp.linalg.norm(positions[receivers] - positions[senders], axis=-1, keepdims=True)

def batch_graphs(nodes, edges, senders, receivers):
        graphs = []
        for n, e, s, r in zip(nodes, edges, senders, receivers):
            graph = jraph.GraphsTuple(nodes=n,
                                edges=e,
                                # edges=None,
                                n_node=jnp.array([n.shape[0]]),
                                n_edge=jnp.array([e.shape[0]]),
                                # n_edge=jnp.array([0]),
                                senders=s,
                                receivers=r,
                                globals={})
            graphs.append(graph)
        return jraph.batch(graphs)

def prep_neval_eq_ntr(cfg, split, data, shuffle):
    data = split_into_timesteps(data, cfg.n_timesteps)
    n_trajectories = len(data)

    n_train, n_val, n_test = (
        int(n_trajectories * split[0]),
        int(n_trajectories * split[1]),
        int(n_trajectories * split[2])
    )

    key = rnd.PRNGKey(cfg.seed)
    idxs = jnp.arange(0, n_trajectories)
    if shuffle: idxs = rnd.permutation(key, idxs)
    tr_idxs, val_idxs, test_idxs = idxs[:n_train], idxs[n_train:(n_val+n_train)], idxs[-n_test:]

    val_idxs = jnp.delete(val_idxs, jnp.where(val_idxs==0)[0])  # remove if first one! 
    test_idxs = jnp.delete(test_idxs, jnp.where(test_idxs==0)[0])  # remove if first one! 
    initial_states_val_data = data[(val_idxs - 1), -cfg.n_eval_warmup:, ...]
    initial_states_test_data = data[(test_idxs - 1), -cfg.n_eval_warmup:, ...]

    tr_data = cut_remainder(data[tr_idxs], cfg.batch_size)
    val_data = cut_remainder(jnp.concatenate([initial_states_val_data, data[val_idxs]], axis=1), cfg.batch_size)
    test_data = cut_remainder(jnp.concatenate([initial_states_test_data, data[test_idxs]], axis=1), cfg.batch_size)

    print(f'Datasets length: Train {len(tr_data)} Val {len(val_data)} Test {len(test_data)}')
    print(f'Some idxs:  \n Train {tr_idxs[:5]} \n Val {val_idxs[:5]}')

    print(tr_data.shape)
    return tr_data, val_data, test_data

def compute_edges(positions, receivers, senders):
    return jnp.linalg.norm(positions[receivers] - positions[senders], axis=-1, keepdims=True)


class DataLoader():
    def __init__(self, 
                 cfg: Config,
                 nodes: jnp.array,
                 shuffle: bool=True, 
                 eval: bool=False):
        
        self.seed = cfg.seed
        self.receivers_idx = cfg.receivers_idx
        self.senders_idx = cfg.senders_idx
        self.r_cutoff = cfg.r_cutoff
        self.capacity_multiplier = 1.
        self.box_size = 8.
        self.dr_threshold = cfg.dr_threshold
        self.batch_size = cfg.batch_size
        self.n_eval_timesteps = cfg.n_eval_timesteps
        self.eval = eval
        self.cfg = cfg

        self.n_data, self.n_timesteps, self.n_nodes, self.n_node_features = nodes.shape

        print(self.n_data, self.n_timesteps, self.n_nodes, self.n_node_features)

        self.nodes = nodes
        
        if cfg.periodic:
            self.displacement_fn, self.shift_fn = space.periodic(cfg.side, wrapped=True)
        else:
            self.displacement_fn, self.shift_fn = space.free()
        
        self._shuffle = shuffle
        key = rnd.PRNGKey(self.seed)
        self.key, subkey = rnd.split(key)

        self._order = jnp.arange(0, self.n_data)
        self._order = rnd.permutation(subkey, self._order)

        self._n_batches = len(self._order) // cfg.batch_size
        
        self._fin = 0
        self._create_graphs = None
        self.n_times_allocated = 0

    def allocate_neighbor_list(self, positions):

        neighbor_fn = neighbor_list(self.displacement_fn, 
                                    capacity_multiplier=1.+self.n_times_allocated/2.,
                                    box_size=self.box_size, 
                                    r_cutoff=self.r_cutoff,
                                    dr_threshold=0.,  # when the neighbor list updates
                                    format=NeighborListFormat.Sparse)

        self.nbrs = neighbor_fn.allocate(positions) 
        
        # print(self.nbrs.max_occupancy, 'max_occ')
        # print(len(self.nbrs.idx[0]))

        def get_edge_info(position):
            nbr = self.nbrs.update(position)
            receivers = nbr.idx[self.receivers_idx]
            senders = nbr.idx[self.senders_idx]
            edges = compute_edges(position, receivers, senders)
            return edges, senders, receivers
        
        _get_edge_info = vmap(get_edge_info, in_axes=(0,), out_axes=(0, 0, 0))

        @jit
        def create_graphs(nodes):
            nodes = nodes.reshape(-1, *nodes.shape[2:])
            edge_info = _get_edge_info(nodes[..., :self.cfg.n_dim])
            graphs = batch_graphs(nodes, *edge_info)
            return graphs

        self.n_times_allocated += 1
    
        return create_graphs

    def __iter__(self):
        return self

    def __len__(self):
        return self._n_batches

    def shuffle(self, key=None, returns_new_key=False, reset=True):
        
        self.key, subkey = rnd.split(self.key)
        self._order = rnd.permutation(subkey, self._order)

        if reset:
            print('Resetting dataloader... ')
            self._fin = 0

        if returns_new_key:
            return self.key

    def __next__(self):
        start = self._fin * self.batch_size
        self._fin += 1
        stop  = self._fin * self.batch_size
        idxs = (self._order[start:stop],)

        if self._create_graphs is None:
            self._create_graphs = self.allocate_neighbor_list(self.nodes[0, 0, :, :self.cfg.n_dim])

        if self._fin < len(self):
            
            if not self.eval:
                nodes = self.nodes[idxs]
                target = nodes[..., :self.cfg.n_dim]
                graphs = self._create_graphs(nodes)
                if self.nbrs.did_buffer_overflow:
                    self._create_graphs = self.allocate_neighbor_list(nodes[0, 0, :, :self.cfg.n_dim]) 
                return graphs, target
            else:
                nodes = self.nodes[idxs]
                nodes_warmup, nodes_eval = jnp.split(nodes, [nodes.shape[1]-self.n_eval_timesteps,], axis=1)
                target_warmup, target_eval = nodes_warmup[..., :self.cfg.n_dim], nodes_eval[..., :self.cfg.n_dim]
                
                graphs_warmup = self._create_graphs(nodes_warmup)
                graphs_eval = self._create_graphs(nodes_eval)
                if self.nbrs.did_buffer_overflow:
                    self._create_graphs = self.allocate_neighbor_list(nodes[0, 0, :, :self.cfg.n_dim]) 
                return ((graphs_warmup, target_warmup), (graphs_eval, target_eval))
            
        else:
            raise StopIteration


def load_andor_transform_data(cfg, raw_data=None):
    print('Loading and processing data.')
    if raw_data is None:
        raw_data = onp.load(cfg.dataset)

    keys = raw_data.keys()

    print(f'Taking features {cfg.node_features}')
    for feature in cfg.node_features:
        assert feature in keys, f'{feature} not in dataset'
    
    setattr(cfg, 'n_dim', raw_data['R'].shape[-1])

    if 'data_vars' in keys:
        setattr(cfg, 'data_vars', raw_data['data_vars'])

    # Get the data statisitics
    if not cfg.n_unroll_eval:
        for name in cfg.node_features:
            tmp_min, tmp_max, tmp_mean = get_stats(raw_data[name])
            setattr(cfg, f'{name}_min', tmp_min)
            setattr(cfg, f'{name}_max', tmp_max)
            setattr(cfg, f'{name}_mean', tmp_mean)
            setattr(cfg, f'{name}', raw_data[name])
    
    setattr(cfg, 'R_lims', tuple((cfg.R_min[i], cfg.R_max[i]) for i in range(cfg.n_dim)))
    setattr(cfg, 'n_nodes', raw_data['R'].shape[1])
    setattr(cfg, 'nodes', onp.squeeze(raw_data['z']))

    print(f'R-Lims: {tuple((float(cfg.R_min[i]), float(cfg.R_max[i])) for i in range(cfg.n_dim))}')

    n_data, n_nodes, _ = raw_data['R'].shape
    initial_info = {} | cfg.data_vars

    # Transform the data
    box_size = raw_data.get('box_size', False)
    if box_size:
        positions = transform(raw_data['R'], 0, box_size, mean=box_size/2.)
    else:
        positions = transform(raw_data['R'], cfg.R_min, cfg.R_max, mean=cfg.R_mean)
    initial_info['R'] = positions

    features = []
    if 'z' in keys:
        node_id = raw_data.get('z', False)
        z = jax.nn.one_hot((node_id-1), int(max(node_id)), dtype=jnp.float32)
        z = z[None, :].repeat(n_data, axis=0)
        print(f'A-Lims: {int(cfg.z_min)} {int(cfg.z_max)}')
        features += [z]
        initial_info['z'] = z
    
    if 'F' in keys:
        F = transform(raw_data['F'], -1, 1, mean=cfg.F_mean) 
        print('F-lims: ' + ' | '.join([f'{cfg.F_min[i]:.3f}, {float(cfg.F_max[i]):.3f}' for i in range(cfg.n_dim)]))
        features += [F]
        initial_info['F'] = F[cfg.n_timesteps]

    if 'V' in keys:
        V = transform(raw_data['V'], -1, 1, mean=cfg.V_mean)
        print(f'V-Lims: {tuple((float(cfg.V_min[i]), float(cfg.V_max[i])) for i in range(cfg.n_dim))}')
        features += [V]
        initial_info['V'] = V[cfg.n_timesteps]

    # Set the node features
    nodes = jnp.concatenate([positions, *features], axis=-1)

    setattr(cfg, 'n_features', nodes.shape[-1] * n_nodes)
    setattr(cfg, 'n_target_features', cfg.n_dim * n_nodes)

    if cfg.n_unroll_eval:
        assert cfg.batch_size == 1
        setattr(cfg, 'initial_info', initial_info)

    return nodes
    

def create_dataloaders(cfg, nodes, shuffle=True, split=(0.7, 0.15, 0.15)):
    # Get the val/train/test split and put into timeslices
    # Tr and val take normal time

    tr, val, test = prep_neval_eq_ntr(cfg, split, nodes, shuffle)

    train_loader = DataLoader(cfg, tr) if not len(tr) == 0 else None
    val_loader = DataLoader(cfg, val, eval=True) if not len(val) == 0 else None
    test_loader = DataLoader(cfg, test, eval=True) if not len(test) == 0 else None

    return train_loader, val_loader, test_loader

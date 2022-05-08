import jax
from jax import numpy as jnp, random as rnd, jit, vmap
import jax.numpy as jnp 
from .config import Config
from dataclasses import asdict

from jax_md.partition import neighbor_list, NeighborListFormat
from jax_md import space

import numpy as onp
import jraph
from math import floor


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


def untransform(data, old_min, old_max, new_min=-1, new_max=1, mean=None):
    data = ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    # if mean is not None:
    #     data = data + mean
    return data


def transform(data, old_min, old_max, new_min=-1, new_max=1, mean=None):
    # if mean is not None:
    #     data = data - mean
    #     old_min -= mean
    #     old_max -= mean
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
        floor(n_trajectories * split[0]),
        floor(n_trajectories * split[1]),
        floor(n_trajectories * split[2])
    )

    key = rnd.PRNGKey(cfg.seed)
    idxs = jnp.arange(0, n_trajectories)
    if shuffle: idxs = rnd.permutation(key, idxs)
    tr_idxs, val_idxs, test_idxs = idxs[:n_train], idxs[n_train:(n_val+n_train)], idxs[-(n_test+1):]

    val_idxs = jnp.delete(val_idxs, jnp.where(val_idxs==0)[0])  # remove if first one! 
    test_idxs = jnp.delete(test_idxs, jnp.where(test_idxs==0)[0])  # remove if first one! 
    initial_states_val_data = data[tuple(val_idxs - 1), -cfg.n_eval_warmup:, ...]
    initial_states_test_data = data[tuple(test_idxs - 1), -cfg.n_eval_warmup:, ...]

    if len(tr_idxs) > 1:
        tr_data = cut_remainder(data[tr_idxs, ...], cfg.batch_size)
        print(f'tr_data shape: {tr_data.shape}')
    else:
        tr_data = []
    
    if len(val_idxs) > 1:
        val_data = cut_remainder(jnp.concatenate([initial_states_val_data, data[val_idxs, ...]], axis=1), cfg.batch_size)
        print(f'val_data shape: {val_data.shape}')
    else:
        val_data = []
    
    if len(test_idxs) > 1:
        test_data = cut_remainder(jnp.concatenate([initial_states_test_data, data[test_idxs, ...]], axis=1), cfg.batch_size)
        print(f'test_data shape {test_data.shape}')
    else:
        test_data = []

    print(f'Datasets length: Train {len(tr_data)} Val {len(val_data)} Test {len(test_data)}')
    print(f'Some idxs:  \n Train {tr_idxs[:5]} \n Val {val_idxs[:5]}')

    print('Split: ', split)
     
    return tr_data, val_data, test_data


class DataLoader():
    def __init__(self, 
                 cfg: Config,
                 nodes: jnp.array,
                 targets: jnp.array,
                 shuffle: bool=True, 
                 eval: bool=False):
        
        self.seed = cfg.seed
        self.receivers_idx = cfg.receivers_idx
        self.senders_idx = cfg.senders_idx
        self.r_cutoff = cfg.r_cutoff
        self.dr_threshold = cfg.dr_threshold
        self.batch_size = cfg.batch_size
        self.n_eval_timesteps = cfg.n_eval_timesteps
        self.eval = eval
        self.cfg = cfg

        self.n_data, self.n_timesteps, self.n_nodes, self.n_node_features = nodes.shape

        self.nodes = nodes
        self.targets = targets
        
        if cfg.periodic:
            self.displacement_fn, self.shift_fn = space.periodic(side=1., wrapped=True)
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

    def __iter__(self):
        return self

    def __len__(self):
        return self._n_batches

    def shuffle(self, key=None, returns_new_key=False, reset=True):
        
        self.key, subkey = rnd.split(self.key)
        self._order = rnd.permutation(subkey, self._order)

        if reset:
            print('Resetting dataloader...')
            self._fin = 0

        if returns_new_key:
            return self.key

    def __next__(self):
        start = self._fin * self.batch_size
        self._fin += 1
        stop  = self._fin * self.batch_size
        idxs = (self._order[start:stop],)

        if self._create_graphs is None:
            self._create_graphs = self.allocate_neighbor_list(self.targets[(0, 0)])

        if self._fin < len(self):
            
            if not self.eval:
                nodes = self.nodes[idxs]
                target = self.targets[idxs]
                if self.nbrs.did_buffer_overflow:
                    self._create_graphs = self.allocate_neighbor_list(target[(0, 0)]) 
                
                graphs = self._create_graphs(nodes, target)

                return graphs, target
            else:
                nodes = self.nodes[idxs]
                target = self.targets[idxs]
                if self.nbrs.did_buffer_overflow:
                    self._create_graphs = self.allocate_neighbor_list(target[(0, 0)]) 

                nodes_warmup, nodes_eval = jnp.split(nodes, [nodes.shape[1]-self.n_eval_timesteps,], axis=1)
                target_warmup, target_eval = jnp.split(target, [nodes.shape[1]-self.n_eval_timesteps,], axis=1)
                
                graphs_warmup = self._create_graphs(nodes_warmup, target_warmup)
                graphs_eval = self._create_graphs(nodes_eval, target_eval)

                return ((graphs_warmup, target_warmup), (graphs_eval, target_eval))
            
        else:
            raise StopIteration

    def allocate_neighbor_list(self, positions):
        # positions: (n_node, n_dim)

        neighbor_fn = neighbor_list(self.displacement_fn, 
                                    capacity_multiplier=1.+self.n_times_allocated/2.,
                                    box_size=1. if self.cfg.periodic else self.cfg.box_size, 
                                    r_cutoff=self.cfg.r_cutoff,  # must be > 1/3 to avoid cell list, or disable_cell_list, I believe this is a performance issue. 
                                    dr_threshold=0.,  # when the neighbor list updates
                                    format=NeighborListFormat.Sparse)

        self.nbrs = neighbor_fn.allocate(positions) 
        
        print(f'Allocating neighbor list... Max occupancy now {self.nbrs.max_occupancy}')

        def compute_edges(positions, receivers, senders):
            displacement = positions[receivers] - positions[senders]
            position_edges = jnp.linalg.norm(displacement, axis=-1, keepdims=True)
            position_edges = jnp.concatenate([displacement, position_edges], axis=-1)
            return position_edges
        
        def get_edge_info(position):
            nbr = self.nbrs.update(position)
            receivers = nbr.idx[self.receivers_idx]
            senders = nbr.idx[self.senders_idx]
            edges = compute_edges(position, receivers, senders)
            return edges, senders, receivers

        def add_sigma(edges, senders, receivers, species, sigma):
            senders_tmp = species[senders] - 1  # to account for zero indexing
            receivers_tmp = species[receivers] - 1
            sigma = sigma[senders_tmp, receivers_tmp][..., None]
            return  jnp.concatenate([edges, sigma], axis=-1), senders, receivers
        
        _get_edge_info = vmap(get_edge_info, in_axes=(0,), out_axes=(0, 0, 0))
        _add_sigma = vmap(add_sigma, in_axes=(0, 0, 0, None, None), out_axes=(0, 0, 0))

        @jit
        def create_graphs(nodes, positions):
            nodes = nodes.reshape(-1, *nodes.shape[2:])
            positions = positions.reshape(-1, *positions.shape[2:])
            edge_info = _get_edge_info(positions)
            if hasattr(self.cfg, 'sigma'):
                species = jnp.array(self.cfg.species).astype(int)
                sigma = jnp.array(self.cfg.sigma)
                edge_info = _add_sigma(*edge_info, species, sigma)
            graphs = batch_graphs(nodes, *edge_info)
            return graphs

        self.n_times_allocated += 1
    
        return create_graphs


def append_lagged_variables(data, lag=1, n_cut=8):  # n_cut is the maximum C value explored
    assert lag < n_cut
    lagged_data = []
    for i in range(lag):
        sliced_data = data[(n_cut-i):-(n_cut+i)]
        lagged_data += [sliced_data]
    return jnp.concatenate(lagged_data, axis=-1)



def load_andor_transform_data(cfg, raw_data=None):
    print('Loading and processing data.')
    if raw_data is None:
        raw_data = onp.load(cfg.dataset, allow_pickle=True)

    keys = raw_data.keys()

    print(f'Taking features {cfg.node_features}')
    for feature in cfg.node_features:
        assert feature in keys, f'{feature} not in dataset'
    
    setattr(cfg, 'n_dim', raw_data['R'].shape[-1])

    data_vars = {k[9:]:v for k, v in raw_data.items() if 'data_var' in k}
    if len(data_vars) > 0: 
        print('Assigning data_vars ', list(data_vars.keys()))
        for k, v in data_vars.items():
            if k == 'species':
                if min(v) == 0:
                    v = v + 1
            setattr(cfg, k, v)
        # [setattr(cfg, k, v) for k, v in data_vars.items()]

    # Get the data statisitics
    for name in cfg.node_features:
        tmp_min, tmp_max, tmp_mean = get_stats(raw_data[name])
        if not '{name}_min' in asdict(cfg):
            setattr(cfg, f'{name}_min', tmp_min)
            setattr(cfg, f'{name}_max', tmp_max)
            setattr(cfg, f'{name}_mean', tmp_mean)
            setattr(cfg, f'{name}', raw_data[name])
    
    setattr(cfg, 'R_lims', tuple((cfg.R_min[i], cfg.R_max[i]) for i in range(cfg.n_dim)))
    setattr(cfg, 'n_nodes', raw_data['R'].shape[1])
    setattr(cfg, 'nodes', onp.squeeze(raw_data['z']))

    print('R-lims: ' + ' | '.join([f'{cfg.R_min[i]:.3f}, {float(cfg.R_max[i]):.3f}' for i in range(cfg.n_dim)]))

    initial_info = {} if cfg.data_vars is None else cfg.data_vars  # for datasets with or without metadata
    # Transform the data
    
    if cfg.periodic:
        box_size = data_vars['box_size']
        print('box_size is ', box_size)
        positions = transform(raw_data['R'], 0., box_size, new_min=0., new_max=1., mean=box_size/2.)
    else:
        positions = transform(raw_data['R'], cfg.R_min, cfg.R_max, mean=cfg.R_mean)
    
    targets = positions
    print(f'Max target checks: {jnp.max(positions):.2f}, {jnp.min(positions):.2f}')
    
    positions = append_lagged_variables(positions, lag=cfg.lag)
    
    initial_info['R'] = positions

    n_data, n_nodes, _ = positions.shape
    
    features = [positions]
    if 'z' in cfg.node_features:
        node_id = raw_data.get('z', False)
        z = jax.nn.one_hot((node_id-1), int(max(node_id)), dtype=jnp.float32)
        z = z[None, :].repeat(n_data, axis=0)
        print(f'A-Lims: {int(cfg.z_min)} {int(cfg.z_max)}')
        features += [z]
        initial_info['z'] = z
    
    if 'F' in cfg.node_features:
        F = transform(raw_data['F'], min(cfg.F_min), max(cfg.F_max), -1, 1, mean=cfg.F_mean) 
        print('F-lims: ' + ' | '.join([f'{cfg.F_min[i]:.3f}, {float(cfg.F_max[i]):.3f}' for i in range(cfg.n_dim)]))
        F = append_lagged_variables(F, lag=cfg.lag)
        features += [F]
        initial_info['F'] = F[cfg.n_timesteps]

    if 'V' in cfg.node_features:
        V = transform(raw_data['V'], min(cfg.V_min), max(cfg.V_max), -1, 1, mean=cfg.V_mean)
        print('V-lims: ' + ' | '.join([f'{cfg.V_min[i]:.3f}, {float(cfg.V_max[i]):.3f}' for i in range(cfg.n_dim)]))
        V = append_lagged_variables(V, lag=cfg.lag)
        features += [V]
        initial_info['V'] = V[cfg.n_timesteps]

    # Set the node features
    nodes = jnp.concatenate(features, axis=-1)

    setattr(cfg, 'n_features', nodes.shape[-1] * n_nodes)
    setattr(cfg, 'n_target_features', cfg.n_dim * n_nodes)
    graph_latent_size = nodes.shape[-1]
    setattr(cfg, 'graph_latent_size', graph_latent_size)
    graph_mlp_features = list((graph_latent_size for _ in range(cfg.n_enc_layers)))
    setattr(cfg, 'graph_mlp_features', graph_mlp_features)

    if cfg.n_unroll_eval:
        setattr(cfg, 'initial_info', initial_info)


    # build an n x n matrix of sigmas depending on the connections 

    return nodes, targets
    

def create_dataloaders(cfg, nodes, targets, shuffle=True, split=(0.7, 0.15, 0.15)):
    # Get the val/train/test split and put into timeslices
    # Tr and val take normal time

    tr, val, test = prep_neval_eq_ntr(cfg, split, nodes, shuffle)
    tr_target, val_target, test_target = prep_neval_eq_ntr(cfg, split, targets, shuffle)

    train_loader = DataLoader(cfg, tr, tr_target) if not len(tr) == 0 else None
    val_loader = DataLoader(cfg, val, val_target, eval=True) if not len(val) == 0 else None
    test_loader = DataLoader(cfg, test, test_target, eval=True) if not len(test) == 0 else None

    return train_loader, val_loader, test_loader

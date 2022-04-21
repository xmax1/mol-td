from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax
from datetime import datetime
from math import ceil


def compile_data(p, f, a,  flatten=False):
    n_data, n_atom = p.shape[:2]
    target = jnp.concatenate([p, f], axis=-1)
    if flatten:
        target = target.reshape(n_data, n_atom * 6)
        data = jnp.concatenate([target, a], axis=-1)
    else:
        data = jnp.concatenate([p, f, a[..., None]], axis=-1)
    return data, target


def uncompile_data(data, y, n_atoms, unflatten=False):
    n_data, nt = data.shape[:2]
    if unflatten:
        data = data.reshape(n_data, nt, n_atoms, -1)  # [..., :-n_atoms]
        y = y.reshape(n_data, nt, n_atoms, -1)
    data_r, data_f = data[..., :3], data[..., 3:6]
    y_r, y_f = y[..., :3], y[..., 3:6]
    return (data_r, data_f), (y_r, y_f)


enc_dec = {'MLP': {'n_features': lambda n_atoms: n_atoms * (3 + 3 + 1),
                   'n_target_features': lambda n_atoms: n_atoms * (3 + 3),
                   'compile_data': lambda p, f, a: compile_data(p, f, a, flatten=True), 
                   'uncompile_data': lambda data, y, n_atoms: uncompile_data(data, y, n_atoms, unflatten=True)
                  },

           'GCN': {'n_features': lambda n_atoms: (3 + 3 + 1),
                   'n_target_features': lambda n_atoms: n_atoms * (3 + 3),
                   'compile_data': lambda p, f, a: compile_data(p, f, a, flatten=False),
                   'uncompile_data': lambda data, y, n_atoms: uncompile_data(data, y, n_atoms, unflatten=False)
                   }
}

@dataclass
class Config:
    seed: int = 1

    # WANDB
    wb:             bool = False
    wandb_status:   str  = 'offline'
    user:           str  = 'xmax1'
    project:        str  = 'test'
    tag:            str  = 'no_tag'
    id:             str  = None  # null for nada for none
    group:          str  = None
    WANDB_API_KEY:  int  = 1

    # MODEL
    model:                  str   = 'SimpleTDVAE'
    n_enc_layers:           int   = 2
    n_dec_layers:           int   = 2
    n_transfer_layers:      int   = 1
    n_embed:                list  = 40
    n_latent:               int   = 2
    y_std:                  float = 1.
    latent_dist_min_std:    float = 0.0001  # 0.0001 cwvae
    dropout:                float = 0.5
    transfer_fn:            str   = 'GRU'
    encoder:                str   = 'GNN'
    decoder:                str   = 'MLP'
    latent_activation:      str   = 'leaky_relu'
    map_activation:         str   = 'leaky_relu'
    beta:                   float = 1000.
    skip_connections:       bool  = False
    post_into_prior:        bool  = False
    likelihood_prior:       bool  = False

    # DATA
    n_target:           int = None
    n_input:            int = None

    # TRAINING
    n_epochs:           int = 10
    batch_size:         int = 128
    lr:                 float = 0.001
    n_timesteps:        int = 3
    n_eval_timesteps:   int = 3
    n_eval_warmup:      int = None
    xlog_media:         bool = False
    clip_grad_norm_by:  float = 10000

    # PATHS
    root:           str = '/home/amawi/projects/mol-td'
    data:           str = './data'
    results:        str = './results/test'
    default_config: str = './configs/default_config.yaml'
    uracil_xyz:     str = './data/uracil.xyz'

    def __post_init__(self):

        print(f'Model: {self.model} \
                \n n_enc_layers: {self.n_enc_layers} \
                \n n_dec_layers: {self.n_dec_layers} ')

        if self.id is None:
            self.id = datetime.now().strftime('%y%m%d%H%M%S')

        if self.wb:
            self.wandb_status = 'online'
        else:
            self.wandb_status = 'disabled'

        if self.model == 'SimpleVAE':
            self.n_timesteps = 0

        if self.n_eval_warmup is None:
            self.n_eval_warmup = max(8, int(0.25 * self.n_eval_timesteps))

    # @property
    # def n_data(self):
    #     return self.n_data

    # def save(self):
    #     self.exp_rootdir.mkdir(parents=True, exist_ok=True)
    #     with (self.exp_rootdir / "config.yml").open("w") as f:
    #         yaml.dump(asdict(self), f, default_flow_style=False)

    def get_stats(self, data, axis=0):
        if len(data.shape) == 3:
            data = data.reshape(-1, 3)
        return jnp.min(data, axis=axis), jnp.max(data, axis=axis), jnp.mean(data, axis=axis)


    def load_raw_data(self, path):
        raw_data = np.load(path)
        positions, forces, atoms = raw_data['R'], raw_data['F'], raw_data['z']
        n_data, n_atoms, _ = positions.shape
        
        self.data_r_min, self.data_r_max, self.data_r_mean = self.get_stats(positions)
        self.data_f_min, self.data_f_max, self.data_f_mean = self.get_stats(forces)
        self.data_atoms_min, self.data_atoms_max, self.data_atoms_mean = self.get_stats(atoms)

        self.data_lims = tuple((self.data_r_min[i], self.data_r_max[i]) for i in range(3))

        self.positions = positions
        self.forces = forces
        self.atoms = atoms
        self.n_data = n_data
        self.n_atoms = n_atoms

        print(f' Pos-Lims: {tuple((float(self.data_r_min[i]), float(self.data_r_max[i])) for i in range(3))} \
                \n F-Lims: {tuple((float(self.data_f_min[i]), float(self.data_f_max[i])) for i in range(3))} \
                \n A-Lims: {int(self.data_atoms_min)} {int(self.data_atoms_max)}    ')

        return positions, forces, atoms

    def load_data(self, path):
        
        positions, forces, atoms = self.load_raw_data(path)

        positions = self.transform(positions, self.data_r_min, self.data_r_max, mean=self.data_r_mean)
        forces = self.transform(forces, self.data_f_min, self.data_f_max, mean=self.data_f_mean)
        # atoms = self.transform(atoms, self.data_atoms_min, self.data_atoms_max, mean=self.data_atoms_mean)
        atoms = jax.nn.one_hot((atoms-1).astype(int), int(max(atoms)))
        atoms = atoms[None, :].repeat(self.n_data, axis=0)
        
        self.n_features = (atoms.shape[-1] + positions.shape[-1] + forces.shape[-1]) * self.n_atoms
        self.n_target_features = positions.shape[-1] * self.n_atoms

        # description = enc_dec[self.enc_dec]
        # self.n_features = description['n_features'](self.n_atoms)
        # self.n_target_features = description['n_target_features'](self.n_atoms)
        # data, target = description['compile_data'](positions, forces, atoms)

        data = jnp.concatenate([positions, forces, atoms], axis=-1)
        target = positions

        return data, target
        
    def initialise_model_hype(self):
        '''
        recognise the difference between the embedding dimensions. 
        self.n_embed is the dimension of the latent space, whereas n_embed_encoder is the dimension of the features of the encoder,
        which in a graph is small but in MLP is ~20 
        '''
        # if self.enc_dec == 'GCN':
        #     n_embed_encoder = self.n_features
        # else:
        #     n_embed_encoder = max(2, ceil(self.n_features*0.25))
        self.enc_hidden = tuple(jnp.linspace(self.n_features, self.n_embed, num=self.n_enc_layers+1).astype(int)[1:])
        self.dec_hidden = tuple(jnp.linspace(self.n_embed, self.n_target_features, num=self.n_dec_layers+1).astype(int)[1:])
        # print('Encoder:', self.enc_hidden, 'Decoder:', self.dec_hidden)
    
    @property
    def total_filters(self):
        return self.filters * self.channels_mult

    def untransform(self, data, old_min, old_max, new_min=-1, new_max=1, mean=None):
        data = ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        if mean is not None:
            data = data + mean
        return data

    def transform(self, data, old_min, old_max, new_min=-1, new_max=1, mean=None):
        if mean is not None:
            data = data - mean
        data = ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        return data

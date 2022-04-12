from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from datetime import datetime


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
    latent_activation:      str   = 'relu'
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
    n_timesteps_eval:   int = 3
    xlog_media:         bool = False

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

    # @property
    # def n_data(self):
    #     return self.n_data

    # def save(self):
    #     self.exp_rootdir.mkdir(parents=True, exist_ok=True)
    #     with (self.exp_rootdir / "config.yml").open("w") as f:
    #         yaml.dump(asdict(self), f, default_flow_style=False)

    def get_stats(self, data, axis=0):
        return jnp.min(data, axis=axis), jnp.max(data, axis=axis), jnp.mean(data, axis=axis)

    def load_data(self, path):
        raw_data = np.load(path)
        positions = raw_data['R']
        self.n_data, self.n_atoms, _ = positions.shape
        positions = positions.reshape(-1, 3)
        forces = raw_data['F'].reshape(-1, 3)
        self.atoms = raw_data['z'].reshape(-1)

        self.data_r_min, self.data_r_max, self.data_r_mean = self.get_stats(positions)
        self.data_f_min, self.data_f_max, self.data_f_mean = self.get_stats(forces)
        self.data_atoms_min, self.data_atoms_max, self.data_atoms_mean = self.get_stats(self.atoms)

        self.data_lims = tuple((self.data_r_min[i], self.data_r_max[i]) for i in range(3))

        positions = self.transform(positions, self.data_r_min, self.data_r_max, mean=self.data_r_mean)
        forces = self.transform(forces, self.data_f_min, self.data_f_max, mean=self.data_f_mean)
        atoms = self.transform(self.atoms, self.data_atoms_min, self.data_atoms_max, mean=self.data_atoms_mean)[None, :].repeat(self.n_data, axis=0)

        # pos_mean = np.mean(positions.reshape((-1, 3)), axis=0)
        # pos_std = np.std(positions.reshape((-1, 3)), axis=0)
        # a_min = pos_mean-2*pos_std
        # a_max = pos_mean+2*pos_std
        # self.data_lims = tuple((vmin, vmax) for vmin, vmax in zip(a_min, a_max))
        
        
        print(f' Pos-Lims: {tuple((float(self.data_r_min[i]), float(self.data_r_max[i])) for i in range(3))} \
                \n F-Lims: {tuple((float(self.data_f_min[i]), float(self.data_f_max[i])) for i in range(3))} \
                \n A-Lims: {int(self.data_atoms_min)} {int(self.data_atoms_max)}    ')


        self.n_features = self.n_atoms * (3 + 3 + 1)
        self.n_target_features = self.n_features - self.n_atoms
        
        data = jnp.concatenate([positions.reshape(-1, self.n_atoms, 3), forces.reshape(-1, self.n_atoms, 3)], axis=-1).reshape(self.n_data, -1)
        data = jnp.concatenate([data, atoms], axis=-1)

        return data
        
    def initialise_model_hype(self):
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

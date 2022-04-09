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
    id:             str  = None  # null for nada for none
    group:          str  = 'no_group'
    WANDB_API_KEY:  int  = 1

    # MODEL
    model:                  str   = 'SimpleTDVAE'
    n_enc_layers:           int   = 2
    n_dec_layers:           int   = 2
    n_transfer_layers:      int   = 2
    n_embed:                list  = 20
    prediction_std:         float = 1.
    latent_dist_min_std:    float = 0.001  # 0.0001 cwvae
    dropout:                float = 0.5

    # DATA
    n_target:           int = None
    n_input:            int = None

    # TRAINING
    n_epochs:           int = 50
    batch_size:         int = 128
    lr:                 float = 0.001
    n_timesteps:        int = 2
    n_timesteps_eval:   int = 2

    # PATHS
    root:           str = '/home/amawi/projects/mol-td'
    data:           str = './data'
    results:        str = './results/test'
    default_config: str = './configs/default_config.yaml'
    uracil_xyz:     str = './data/uracil.xyz'

    def __post_init__(self):

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

    def load_data(self, path):
        raw_data = np.load(path)
        positions = raw_data['R']
        forces = raw_data['F']
        self.atoms = raw_data['z']
        self.data_r_min = jnp.min(positions)
        self.data_r_max = jnp.max(positions)
        self.data_f_min = jnp.min(forces)
        self.data_f_max = jnp.max(forces)
        self.data_atoms_min = jnp.min(self.atoms)
        self.data_atoms_max = jnp.max(self.atoms)

        self.n_data, self.n_atoms, _ = positions.shape
        
        pos_mean = np.mean(positions.reshape((-1, 3)), axis=0)
        pos_std = np.std(positions.reshape((-1, 3)), axis=0)
        a_min = pos_mean-2*pos_std
        a_max = pos_mean+2*pos_std
        self.data_lims = tuple((vmin, vmax) for vmin, vmax in zip(a_min, a_max))
        print(f'Position mean: {pos_mean} \n Position std: {pos_std} \n Lims: {self.data_lims} \
                \n F-Lims: {float(self.data_f_min):.4f} {float(self.data_f_max):.4f} \
                \n A-Lims: {int(self.data_atoms_min)} {int(self.data_atoms_max)}    ')

        positions = self.transform(positions, self.data_r_min, self.data_r_max)
        forces = self.transform(raw_data['F'], self.data_f_min, self.data_f_max)
        atoms = self.transform(self.atoms, self.data_atoms_min, self.data_atoms_max)[None, :].repeat(self.n_data, axis=0)

        self.n_features = self.n_atoms * (3 + 3 + 1)
        self.n_target_features = self.n_features - self.n_atoms
        
        data = jnp.concatenate([positions, forces], axis=-1).reshape(self.n_data, -1)
        data = jnp.concatenate([data, atoms], axis=-1)

        return data
        
    def initialise_model_hype(self):
        self.enc_hidden = tuple(jnp.linspace(self.n_features, self.n_embed, num=self.n_enc_layers+1).astype(int)[1:])
        self.dec_hidden = tuple(jnp.linspace(self.n_embed, self.n_target_features, num=self.n_dec_layers+1).astype(int)[1:])
        # print('Encoder:', self.enc_hidden, 'Decoder:', self.dec_hidden)
    
    @property
    def total_filters(self):
        return self.filters * self.channels_mult

    def transform(self, data, old_min, old_max, new_min=-1, new_max=1):
        data = ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        return data

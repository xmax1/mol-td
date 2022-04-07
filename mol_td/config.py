from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from datetime import datetime


@dataclass
class Config:
    seed: int = 1

    # WANDB
    use_wandb:      bool = False
    wandb_status:   str = 'offline'
    user:           str = 'xmax1'
    project:        str ='test'
    id:             str = None  # null for nada for none
    WANDB_API_KEY:  int = 1

    # MODEL
    model:                  str = 'SimpleTDVAE'
    n_enc_layers:           int = 1
    n_dec_layers:           int = 1
    n_transfer_layers:      int = 1
    n_embed:                list = 20
    prediction_std:         float = 1.
    latent_dist_min_std:    float = 0.01  # 0.0001 cwvae

    # DATA
    n_target:           int = None
    n_input:            int = None

    # TRAINING
    n_epochs:           int = 2
    batch_size:         int = 16
    lr:                 float = 0.0001
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

        if self.use_wandb:
            self.wandb_status = 'online'
        else:
            self.wandb_status = 'disabled'

    # @property
    # def n_data(self):
    #     return self.n_data

    # def save(self):
    #     self.exp_rootdir.mkdir(parents=True, exist_ok=True)
    #     with (self.exp_rootdir / "config.yml").open("w") as f:
    #         yaml.dump(asdict(self), f, default_flow_style=False)

    def load_data(self, path):
        raw_data = np.load(path)
        positions = normalise(raw_data['R'])
        forces = normalise(raw_data['F'])
        self.n_data, self.n_atoms, _ = positions.shape
        atoms = normalise(raw_data['z'])[None, :].repeat(self.n_data, axis=0)

        self.n_features = self.n_atoms * (3 + 3 + 1)
        self.n_target_features = self.n_features - self.n_atoms
        
        data = jnp.concatenate([positions, forces], axis=-1).reshape(self.n_data, -1)
        data = jnp.concatenate([data, atoms], axis=-1)

        return data
        
    def initialise_model_hype(self):
        self.enc_hidden = tuple(jnp.linspace(self.n_features, self.n_embed, num=self.n_enc_layers+1).astype(int)[1:])
        self.dec_hidden = tuple(jnp.linspace(self.n_embed, self.n_target_features, num=self.n_dec_layers+1).astype(int))

        # return dict(positions=raw_data['R'],
        #             forces=raw_data['F'],
        #             atoms=raw_data['z'][None, :].repeat(self.n_data, axis=0))

    
    
    @property
    def total_filters(self):
        return self.filters * self.channels_mult

def normalise(data, new_min=-1, new_max=1):
        data_min = jnp.min(data)
        data_max = jnp.max(data)
        data = ((data - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min
        return data
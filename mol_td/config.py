from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax.numpy as jnp
import jax
from datetime import datetime
from math import ceil
import os
from .utils import create_animation_2d, md17_log_wandb_videos_or_images
from .evaluation import evaluate_position_nve, evaluate_position_md17

evaluate_positions = {'md17': evaluate_position_md17,
                      'nve': evaluate_position_nve
}

media_loggers = {'md17': md17_log_wandb_videos_or_images,
                 'nve': create_animation_2d
}

@dataclass
class Config:
    seed: int = 1

    # WANDB / LOGGING
    wb:             bool = False
    wandb_status:   str  = 'offline'
    user:           str  = 'xmax1'
    project:        str  = 'test'
    tag:            str  = 'no_tag'
    id:             str  = None  # null for nada for none
    group:          str  = None
    WANDB_API_KEY:  int  = 1
    compute_rdfs:   bool = True

    # MODEL
    model:                  str   = 'SimpleTDVAE'
    n_enc_layers:           int   = 2
    n_dec_layers:           int   = 2
    n_transfer_layers:      int   = 1
    n_embed:                int   = 40
    n_embed_latent:         int   = 20
    n_latent:               int   = 2
    y_std:                  float = 1.
    latent_dist_min_std:    float = 0.0001  # 0.0001 cwvae
    dropout:                float = 0.5
    transfer_fn:            str   = 'GRU'
    encoder:                str   = 'GNN'
    decoder:                str   = 'MLP'
    latent_activation:      str   = 'relu'
    map_activation:         str   = 'leaky_relu'
    beta:                   float = 1.
    likelihood_prior:       bool  = False
    clockwork:              bool  = False
    mean_trajectory:        bool  = False
    predict_sigma:          bool  = False
    node_features:          tuple = None  # R=position, F=Force, z=atom_type, V=velocity
    edge_features:          tuple = ('r',)
    
    graph_mlp_features:     int  = None
    graph_latent_size:      int  = None

    # DATA
    n_target:           int = None
    n_input:            int = None
    dataset:            str = './data/md17/uracil_dft.npz'
    load_model:         str = None
    eval:               bool = False
    data_vars:          dict = None
    run_path:           str = None
    n_unroll_eval:      int = 0
    split:              tuple = (0.7, 0.15, 0.15)
    lag:                int = 1

    # SYSTEM
    r_cutoff:           float = 2.
    box_size:           float = 8.
    dr_threshold:       float = 0.0
    periodic:           bool = False
    receivers_idx:      int = 0
    senders_idx:        int = 1

    # TRAINING
    n_epochs:           int = 10
    batch_size:         int = 16
    lr:                 float = 0.001
    n_timesteps:        int = 4
    n_eval_timesteps:   int = 4
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
        
        self.n_dec_layers = self.n_enc_layers

        print(f'Model: {self.model} \
                \n n_enc_layers: {self.n_enc_layers} \
                \n n_dec_layers: {self.n_dec_layers} ')
        
        self.experiment = self.dataset.split('/')[0]
        self.dataset = f'./data/{self.dataset}.npz'

        if self.load_model is not None:
            self.load_model = os.path.join(self.experiment, self.load_model)

        if self.wb:
            self.wandb_status = 'online'
        else:
            self.wandb_status = 'disabled'

        if self.model == 'SimpleVAE':
            self.n_timesteps = 0

        if self.n_eval_warmup is None:
            self.n_eval_warmup = max(8, int(0.25 * self.n_eval_timesteps))

        # TODO change to experiment configuration files
        if 'nve' == self.experiment:
            print('NVE experiment')
            self.compute_rdfs = False
            self.periodic = True
            if self.node_features is None:
                self.node_features = ('R', 'F', 'z', 'V')
            self.box_size = 1.
            self.r_cutoff = 0.35

        elif 'md17' == self.experiment:
            self.compute_rdfs = True
            self.periodic = False
            if self.node_features is None:
                self.node_features = ('R', 'F', 'z')
            self.box_size = 8.  # this needs to be 3x bigger than r_cutoff
            self.r_cutoff = 2.


    def get_stats(self, data, axis=0):
        if len(data.shape) == 3:
            data = data.reshape(-1, 3)
        return jnp.min(data, axis=axis), jnp.max(data, axis=axis), jnp.mean(data, axis=axis)


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
        self.dec_hidden = tuple(jnp.linspace(self.n_embed_latent, self.n_target_features, num=self.n_dec_layers+1).astype(int)[1:])
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


from mol_td.utils import create_animation_2d
from mol_td.config import Config
from mol_td.data_fns import load_andor_transform_data
import wandb
from dataclasses import asdict
cfg = Config(dataset='nve/test_data')
data, target = load_andor_transform_data(cfg)

n_species = len(cfg.species)
print(n_species)
data = data[..., :2].reshape(1, -1, n_species, 2)[:, ::10]

data = {'test': data}
# import numpy as np
# data = np.array(data)
# np.save('test_anim.npz', data)


run = wandb.init(project='TimeDynamics_v2', 
                     entity=cfg.user, 
                     group='new_group')

with run:
    create_animation_2d(data, cfg)


print(cfg.data_vars)






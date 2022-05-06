
from mol_td.utils import create_animation_2d
from mol_td.config import Config
from mol_td.data_fns import load_andor_transform_data

cfg = Config(dataset='nve/test_data')
data = load_andor_transform_data(cfg)

n_species = len(cfg.species)
print(n_species)
data = data[..., :2].reshape(-1, n_species, 2)[::10]

# import numpy as np
# data = np.array(data)
# np.save('test_anim.npz', data)

create_animation_2d(cfg, data, name='test.gif')


print(cfg.data_vars)






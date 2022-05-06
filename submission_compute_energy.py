import re
import os
import itertools
from mol_td.utils import get_directory_leafs


folder = './log/md17/energy_test'
leafs = get_directory_leafs(folder)
print('Running energy computation for: \n', *[f'{s} \n' for s in leafs])

for path in leafs:
    dataset = os.path.join(path, 'eval_positions.npz')
    if 'eval_positions.npz' in os.listdir(path):
        os.system(f'sbatch  ./run_compute_energy.sh {dataset}')


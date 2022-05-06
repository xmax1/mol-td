from ase import Atoms
from gpaw import GPAW, PW

import numpy as onp
import argparse
from .utils import get_base_folder, robust_dictionary_append
import os

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', default=None, type=str)  # pass path to eval_positions

args = parser.parse_args()

args = vars(args)

base_folder = get_base_folder(args['dataset'])

data = onp.load(args['dataset'])
positions  = data['R']
atoms = data['z']

energies = {}
for r in positions:
    atoms = Atoms(numbers=[int(z) for z in atoms], pbc=False, positions=r)
    atoms.center(vacuum=3.0)   # maybe 5 so no interactions
    atoms.calc = GPAW(symmetry={'point_group': False}, mode=PW(400), xc='PBE')
    energy = atoms.get_total_energy()
    ke = atoms.get_kinetic_energy()
    pe = atoms.get_potential_energy()

    energies = robust_dictionary_append(energies, {'TE': energy, 'PE': pe, 'KE': ke})

energies = {k: onp.concatenate(v, axis=0) for k, v in energies.items()}
onp.savez(os.path.join(base_folder, 'energies.npz', **energies))
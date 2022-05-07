from ase import Atoms
from gpaw import GPAW, PW

import numpy as onp
import argparse
from mol_td.utils_nojax import get_base_folder, robust_dictionary_append
import os


def compute_energy(path):

    data = onp.load(path)
    positions  = data['R']
    nodes = data['z']

    energies = {}
    for r in positions:
        atoms = Atoms(numbers=[int(z) for z in nodes], pbc=False, positions=r)
        atoms.center(vacuum=3.0)   # maybe 5 so no interactions
        atoms.calc = GPAW(symmetry={'point_group': False}, mode=PW(400), xc='PBE')
        energy = atoms.get_total_energy()
        ke = atoms.get_kinetic_energy()
        pe = atoms.get_potential_energy()

        energies = robust_dictionary_append(energies, {'TE': energy, 'PE': pe, 'KE': ke})

    energies = {k: onp.concatenate(v, axis=0) for k, v in energies.items()}

    return energies




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', default=None, type=str)  # pass path to eval_positions

    args = parser.parse_args()

    args = vars(args)

    base_folder = get_base_folder(args['dataset'])

    energies = compute_energy(args['dataset'])

    onp.savez(os.path.join(base_folder, 'energies.npz', **energies))
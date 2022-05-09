from ase import Atoms
# from gpaw import GPAW, PW

import numpy as onp
import argparse
from mol_td.utils_nojax import get_base_folder, robust_dictionary_append
import os
from xtb.ase.calculator import XTB
from time import time

def compute_energy(path, every=100):

    data = onp.load(path)
    positions  = data['R'][::every]
    nodes = data['z']

    energies = {}
    t0 = time()
    for i, r in enumerate(positions):
        atoms = Atoms(numbers=[int(z) for z in nodes], pbc=False, positions=r)
        atoms.center(vacuum=3.0)   # maybe 5 so no interactions
        # atoms.calc = GPAW(symmetry={'point_group': False}, mode=PW(400), xc='PBE')
        atoms.calc = XTB(method="GFN2-xTB")
        energy = atoms.get_total_energy()
        ke = atoms.get_kinetic_energy()
        pe = atoms.get_potential_energy()

        tmp = {'TE': onp.array([energy]), 'PE': onp.array([pe]), 'KE': onp.array([ke])}
        energies = robust_dictionary_append(energies, tmp)

        if i % 100 == 0:
            t1 = time()
            t = t1 - t0
            print(f'Iteration {i} time {t:.3f}')
            t0 = t1


    energies = {k: onp.concatenate(v, axis=0) for k, v in energies.items()}

    return energies




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', default=None, type=str)  # pass path to eval_positions
    parser.add_argument('-e', '--every', default=100, type=int)
    parser.add_argument('-n', '--name', default=None, type=str)
    args = parser.parse_args()

    args = vars(args)

    if args['name'] is None:
        exit('no name provided')

    base_folder = get_base_folder(args['dataset'])

    energies = compute_energy(args['dataset'], every=args['every'])

    onp.savez(os.path.join(base_folder, 'energies', args['name'] + '_energies.npz'), **energies)
import jax.numpy as jnp
from itertools import product, combinations
import numpy as np


atoms_list = {1: 'H',
         2: 'He',
         3: 'Li',
         4: 'Be',
         5: 'B',
         6: 'C',
         7: 'N',
         8: 'O',
         9: 'F',
         10: 'Ne',
         11: 'Na',
         12: 'Mg', 
         13: 'Al',
         14: 'Si'

}


def compute_rdfs_all_unique_bonds(atoms, positions, n_skip=10):
    unique_atoms = jnp.unique(atoms).astype(int)
    positions = positions[::n_skip]
    unique_atoms = [int(a) for a in jnp.unique(atoms).astype(int)]
    # bonds = product(*(tuple(unique_atoms), tuple(unique_atoms))) # gives all combinations
    bonds = list(combinations(unique_atoms, 2))  # only gives unique combinations
    bonds.extend((x, x) for x in unique_atoms)

    rbfs = {}
    for A, B in bonds:
        idxsA = np.where(atoms == A)[0]
        idxsB = np.where(atoms == B)[0]
        Ar = positions[:, idxsA, ...][:, :, None, :]
        Br = positions[:, idxsB, ...][:, None, :, :]
        distances = np.linalg.norm(Ar - Br, axis=-1).reshape(-1)
        rbf, x = np.histogram(distances, bins=100, range=(0, np.sqrt(2)), density=True)
        x = x[:-1] + (x[1] - x[0]) / 2.
        rbfs[f'{atoms_list[A]}-{atoms_list[B]}'] = np.stack([x, rbf], axis=-1)
    return rbfs


def compute_rdfs(atoms, positions, mode='all_unique_bonds'):
    if mode == 'all_unique_bonds':
        rdfs = compute_rdfs_all_unique_bonds(atoms, positions)
    return rdfs


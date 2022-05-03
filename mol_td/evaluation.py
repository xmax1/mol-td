from jax_md import space, energy, quantity
from jax import jit, random as rnd
from jax import lax
from jax import numpy as jnp

import time

from jax_md import space, smap, energy, minimize, quantity, simulate


def evaluate_position_nve(cfg, positions, initial_info, print_every=20):
    F = initial_info['F']
    V = initial_info['V']
    mass = initial_info['mass']

    positions = positions.reshape(-1, *positions.shape[2:])

    displacement, shift_fn = space.periodic(cfg.data_vars['box_size']) 
    energy_fn = energy.soft_sphere_pair(displacement, species=cfg.data_vars['species'], sigma=cfg.data_vars['sigma'])
    
    force_fn = quantity.canonicalize_force(energy_fn)
    dt = cfg.data_vars['dt']

    dt = dt
    dt_2 = dt / 2
    dt2_2 = dt ** 2 / 2
    Minv = 1 / mass

    @jit
    def compute_update(position, V, F):
    
        F_new = force_fn(position)
        V += (F + F_new) * dt_2 * Minv

        F = F_new

        PE = energy_fn(position)
        KE = quantity.kinetic_energy(V)
        return V, F, PE, KE

                
    data = {'R': [], 'F': [], 'V': [], 'PE': [], 'KE':[], 'TE': []}
    for i, position in enumerate(positions):

        V, F, PE, KE = compute_update(position, V, F)

        data['R'].append(position)
        data['F'].append(F)
        data['V'].append(V)
        data['KE'].append(jnp.array([KE]))
        data['PE'].append(jnp.array([PE]))
        data['TE'].append(jnp.array([KE + PE]))

        if i % print_every == 0:
            print(f'Iteration {i} Energy {data["TE"][-1]}')
    return data, {'F': F, 'V': V, 'mass': mass}
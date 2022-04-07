import os
import pickle as pk
import wandb

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.image as img
from matplotlib.cm import get_cmap


DEFAULT_CONFIG_PATH = './configs/default_config.yaml'


def get_sizes(data, new_min=2, new_max=200):
    mean = np.mean(data)
    std = np.std(data)
    data = np.clip(data, a_min=mean-2*std, a_max=mean+2*std)
    data_min = np.min(data)
    data_max = np.max(data)
    sizes = ((data - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min
    return sizes


def get_image(data_r, atoms, sizes):
    unique_atoms = np.unique(atoms)
    n_unique_atoms = len(unique_atoms)
    n_atoms = len(atoms)
    
    n_colors = 10
    assert n_unique_atoms < n_colors
    cmap = get_cmap('tab10')
    lookup_table = np.zeros(n_atoms)
    lookup_table[unique_atoms] += (np.linspace(0., 1., n_colors)[:n_unique_atoms])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.view_init(0, 0)

    for position, z, s in zip(data_r, atoms, sizes):
        ax.scatter(*(position), marker='o', color=cmap(lookup_table[int(z)]), s=s)

    fig.tight_layout(pad=0)
    ax.margins(0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    vid = np.asarray(canvas.buffer_rgba()).astype(int)
    plt.close()

    # plt.savefig('tmp.png')
    # plt.show()
    # arr1 = img.imread('tmp.png') * 255

    return vid


def get_video(data_r, atoms):
    sizes = get_sizes(data_r[..., -1])  # n_t, n_atom, 3
    data_stream = []
    for data_r_step, size_step in zip(data_r, sizes):
        img = get_image(data_r_step, atoms, size_step)
        data_stream.append(img)
    return np.stack(data_stream, axis=0)


def log_video(data, name, atoms=None):
    n_timesteps = data.shape[0]
    data = data.reshape((n_timesteps, -1, 6))
    data_r = data[:, :, :3]
    atoms = atoms.astype(int)
    data_f = data[:, :, 3:]

    data_video = get_video(data_r, atoms)
    wandb_video = np.transpose(data_video.copy(), (0, 3, 1, 2))
    # wandb_video[:, :, ...] = 0

    wandb.log({name: wandb.Video(wandb_video.astype(int), fps=4)}) # (time, channel, height, width)

    return data_video


def oj(*paths):
    return os.path.join(*paths)


def save_pk(x, path):
    with open(path, 'wb') as f:
        pk.dump(x, f)


def load_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x


def print_dict(dictionary, name=''):
    new_dict = {k: v for k, v in dictionary.items() if type(v) in [int, float]}
    print(f'{name}: {str(new_dict)}')
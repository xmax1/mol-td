import os
import pickle as pk
import wandb

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.image as img
from matplotlib.cm import get_cmap


def get_sizes(data, zlim, new_min=2, new_max=200):
    data_min, data_max = zlim
    sizes = ((data - data_min) / (data_max - data_min)) * (new_max - new_min) + new_min
    return np.clip(sizes, a_min=2, a_max=200)


def get_images(data_rs, atoms, sizes, lims=None):
    imgs = []
    for data_r, size in zip(data_rs, sizes):
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
        if lims is None:
            ax.set_axis_off()
        else:
            ax.set_xlim(lims[0]), ax.set_ylim(lims[1]), ax.set_zlim(lims[2])
        # ax.view_init(0, 0)
        ax.grid(False)

        for position, z, s in zip(data_r, atoms, size):
            ax.scatter(*(position), marker='o', color=cmap(lookup_table[int(z)]), s=s)

        fig.tight_layout(pad=0)
        ax.margins(0)
        canvas = FigureCanvasAgg(fig)  # from png # arr = img.imread('tmp.png') * 255
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba()).astype(int)
        plt.close()
        imgs.append(img)

    return np.stack(imgs, axis=0)  # b, x, y, c


def get_video(data_r, atoms, sizes, lims):
    data_stream = []
    for i in range(data_r.shape[1]):
        img = get_images(data_r[:, i, ...], atoms, sizes[:, i, ...], lims)  # b, 1, x, y, c
        data_stream.append(img[0])  # x, y, c
    return np.stack(data_stream, axis=0)  # t, x, y, c 


def log_wandb_videos_or_images(data, cfg, n_batch=10):
    logs = {}
    for k, arr in data.items():
        n_dim = len(arr.shape)
        if n_dim == 2:
            arr = np.squeeze(arr[:n_batch].reshape((-1, cfg.n_atoms, 6))[..., :3])
            arr = cfg.untransform(arr, -1., 1., new_min=cfg.data_r_min, new_max=cfg.data_r_max, mean=cfg.data_r_mean)
            sizes = get_sizes(arr[..., -1], cfg.data_lims[-1])  # z positions
            media = get_images(arr, cfg.atoms, sizes, cfg.data_lims)
            media = [wandb.Image(m) for m in media]
        elif n_dim == 3:
            arr = np.squeeze(arr[:n_batch].reshape((-1, arr.shape[1], cfg.n_atoms, 6))[..., :3])[None, ...]
            arr = cfg.untransform(arr, -1., 1., new_min=cfg.data_r_min, new_max=cfg.data_r_max, mean=cfg.data_r_mean)
            sizes = get_sizes(arr[..., -1], cfg.data_lims[-1])  # z positions
            media = get_video(arr, cfg.atoms, sizes, cfg.data_lims)
            media = wandb.Video(np.transpose(media, (0, 3, 1, 2)), fps=2)
        else:
            print(f'Media {k} is shape {arr.shape}')
        logs[k] = media

    wandb.log(logs)


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
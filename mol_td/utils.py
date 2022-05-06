import os
import pickle as pk
import wandb
import yaml
import numpy as np
from jax import numpy as jnp
from distutils.util import strtobool
from dataclasses import asdict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.image as img
from matplotlib.cm import get_cmap


def get_directory_leafs(path, target='cfg.pk'):
    subdirs = [x[0] for x in os.walk(path)]
    leafs = []
    for subdir in subdirs:
        files = os.listdir(subdir)
        if target in files:
            leafs += [subdir]
    return leafs


def get_base_folder(path):
    if '.' in path:
        base_folder = '/'.join(path.split('/')[:-1])
    else:
        base_folder = path
    return base_folder


def makedir_to_path(path):
    dir_path = get_base_folder(path)
    print(dir_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'Making path {dir_path}')


def makedir(path):
    if not os.path.exists(path): 
        os.makedirs(path)


def yaml_to_dict(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def load_cfg(path):
    with open(path, 'rb') as f:
        cfg = pk.load(f)
    return cfg


def save_cfg(cfg, path):
    makedir(path)
    cfg = asdict(cfg)
    save_pk(cfg, path  + '/cfg.pk')
    cfg = {k:v for k,v in cfg.items() if type(v) in (str, int, float, tuple, bool)}
    print(cfg)
    with open(path  + '/cfg.yml', 'w') as file:
        yaml.safe_dump(cfg, file)


def save_params(params, path):  # path should already exist
    n_files = len(os.listdir(path))
    path = os.path.join(path, f'best_params{n_files}.pk')
    save_pk(params, path)


def accumulate_signals(storage, signal):
    signal = {k:v for k,v in signal.items() if isinstance(v, jnp.ndarray)}
    for k, v in signal.items():
        if not v.shape in ((1,), ()):
            v = [v]
        if k in storage.keys():
            storage[k] += v
        else:
            storage[k] = v
    return storage


def filter_scalars(signal, n_batch=1., tag='', ignore=()):
    # get the jax arrays
    signal_arr = {k:v for k,v in signal.items() if ((isinstance(v, jnp.ndarray)) and (k not in ignore))}
    # get the scalars
    scalars = {tag+k:float(v)/n_batch for k,v in signal_arr.items() if v.shape in ((1,), ())}
    # signal = {tag+k:(float(v)/n_batch) for k, v in signal.items() if isinstance(v, float)}
    return scalars 


def input_bool(x):
    x = strtobool(x)
    if x: return True
    else: return False

def input_tuple(x, tuple_type='str'):
    if tuple_type == 'str':
        x = str(x)
        x = x.split(',')
        x = tuple([i for i in x])
    return x

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


def md17_log_wandb_videos_or_images(data, cfg, n_batch=10, fps=2):
    logs = {}
    for k, arr in data.items():
        n_dim = len(arr.shape)  # (bs, nt, natom, 3)
        arr = arr[:n_batch]   # (bs, nt, natom, 3)
        arr = cfg.untransform(arr, -1., 1., new_min=cfg.R_min, new_max=cfg.R_max, mean=cfg.R_mean)
        sizes = get_sizes(arr[..., -1], cfg.R_lims[-1])  

        if n_dim == 3:
            media = get_images(arr, cfg.nodes, sizes, cfg.R_lims)
            media = [wandb.Image(m) for m in media]
        
        elif n_dim == 4:
            media = get_video(arr, cfg.nodes, sizes, cfg.R_lims)
            media = wandb.Video(np.transpose(media, (0, 3, 1, 2)), fps=fps)
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


def snapshot_2d(cfg, im):
    ms_base = 10

    unique_species = np.unique(cfg.species)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.grid(False)

    for species in unique_species:
        ms = ms_base * species
        idxs = np.argwhere(species == cfg.species)
        ax.plot(im[idxs, 0], im[idxs, 1], 'o', markersize=ms * 0.5)

    ax.set_xlim([0, cfg.box_size])
    ax.set_ylim([0, cfg.box_size])

    fig.tight_layout(pad=0)
    ax.margins(0)
    canvas = FigureCanvasAgg(fig)  # from png # arr = img.imread('tmp.png') * 255
    canvas.draw()
    im = np.asarray(canvas.buffer_rgba()).astype(int)
    plt.close()
    return im

from math import floor
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def create_animation_2d(cfg, arr, name='test.mp4', dpi=400):
    '''
    arr: (nt, n_nodes, n_dim)

    '''
    arr = cfg.untransform(arr, -1., 1., new_min=cfg.R_min, new_max=cfg.R_max, mean=cfg.R_mean)

    frames = []
    for im in arr:
        im = snapshot_2d(cfg, im)
        frames.append(im)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(frames[0], interpolation='nearest')
    fig.set_size_inches([5,5])
    fig.tight_layout()

    def update_img(frame):
        im.set_data(frame)
        return im

    ani = animation.FuncAnimation(fig, update_img, frames=frames[1:], interval=30)
    # writer = animation.PillowWriter(fps=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save(name,writer=writer,dpi=dpi)
    return ani


def robust_dictionary_append(data, step_data):
    for k, v in step_data.items():
        if k in data.keys():
            data[k] += [v[0]] if type(v) is list else [v]
        else:
            data[k] = [v[0]] if type(v) is list else [v]
    return data








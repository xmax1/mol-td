import os
import pickle as pk
import yaml


DEFAULT_CONFIG_PATH = './configs/default_config.yaml'


def load_config(path=None):
    if path is None: path = DEFAULT_CONFIG_PATH
    with open(oj(path)) as file:
        cfg = yaml.safe_load(file)
    return cfg


def oj(*paths):
    return os.path.join(*paths)


def save_pk(x, path):
    with open(path, 'wb') as f:
        pk.dump(x, f)


def load_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x

import os

def robust_dictionary_append(data, step_data):
    for k, v in step_data.items():
        if k in data.keys():
            data[k] += [v[0]] if type(v) is list else [v]
        else:
            data[k] = [v[0]] if type(v) is list else [v]
    return data



def get_base_folder(path):
    if '.' in path:
        base_folder = '/'.join(path.split('/')[:-1])
    else:
        base_folder = path
    return base_folder


def get_directory_leafs(path, target='cfg.pk'):
    subdirs = [x[0] for x in os.walk(path)]
    leafs = []
    for subdir in subdirs:
        files = os.listdir(subdir)
        if target in files:
            leafs += [subdir]
    return leafs